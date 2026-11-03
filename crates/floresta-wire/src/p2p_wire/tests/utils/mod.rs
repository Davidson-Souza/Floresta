// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bitcoin::Amount;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::CompactTarget;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::ScriptBuf;
use bitcoin::Sequence;
use bitcoin::TxMerkleNode;
use bitcoin::TxOut;
use bitcoin::Witness;
use bitcoin::absolute::LockTime;
use bitcoin::block::Header;
use bitcoin::block::Version as BlockVersion;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::encode;
use bitcoin::consensus::encode::deserialize_hex;
use bitcoin::hashes::Hash;
use bitcoin::hex::FromHex;
use bitcoin::p2p::ServiceFlags;
use bitcoin::transaction::Transaction;
use bitcoin::transaction::TxIn;
use bitcoin::transaction::Version as TransactionVersion;
use derive_more::Constructor;
use floresta_chain::AssumeValidArg;
use floresta_chain::ChainState;
use floresta_chain::FlatChainStore;
use floresta_chain::FlatChainStoreConfig;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_common::Ema;
use floresta_common::service_flags;
use floresta_mempool::Mempool;
use rand::Rng;
use rand::rand_core::UnwrapErr;
use rand::rngs::SysRng;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio::task;
use tokio::time::timeout;
use zstd;

use crate::UtreexoNodeConfig;
use crate::address_man::AddressMan;
use crate::node::ConnectionKind;
use crate::node::InflightRequests;
use crate::node::LocalPeerView;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::PeerStatus;
use crate::node::UtreexoNode;
use crate::node::swift_sync_ctx::SwiftSync;
use crate::node::sync_ctx::SyncNode;
use crate::node_context::NodeContext;
use crate::p2p_wire::block_proof::UtreexoProof;
use crate::p2p_wire::peer::PeerMessages;
use crate::p2p_wire::peer::Version;
use crate::p2p_wire::transport::TransportProtocol;

pub mod mock_chain;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct UtreexoRoots {
    roots: Option<Vec<String>>,
    numleaves: usize,
}

// Nightly Clippy false positive in `Constructor`-generated code:
// https://github.com/rust-lang/rust-clippy/issues/17525
#[allow(clippy::redundant_field_names)]
#[derive(Debug, Constructor)]
pub struct SimulatedPeer {
    headers: Vec<Header>,
    blocks: HashMap<BlockHash, Block>,
    accs: HashMap<BlockHash, Vec<u8>>,
    node_tx: UnboundedSender<NodeNotification>,
    node_rx: UnboundedReceiver<NodeRequest>,
    peer_id: u32,
}

/// Services shared by the simulated peer and the node's local peer record.
fn simulated_peer_services() -> ServiceFlags {
    ServiceFlags::NETWORK
        | service_flags::UTREEXO.into()
        | service_flags::UTREEXO_ARCHIVE.into()
        | ServiceFlags::WITNESS
        | ServiceFlags::COMPACT_FILTERS
}

impl SimulatedPeer {
    pub async fn run(&mut self) {
        let version = Version {
            user_agent: "node_test".to_string(),
            protocol_version: 0,
            blocks: rand::random::<u32>() % 23,
            id: self.peer_id,
            address_id: rand::random::<u64>() as usize,
            services: simulated_peer_services(),
            time_offset: 0,
            kind: ConnectionKind::OutboundFullRelay(service_flags::UTREEXO.into()),
            transport_protocol: TransportProtocol::V2,
        };

        self.node_tx
            .send(NodeNotification::FromPeer(
                self.peer_id,
                PeerMessages::Ready(version),
                Instant::now(),
            ))
            .unwrap();

        loop {
            let req = self.node_rx.recv().await.unwrap();
            let now = Instant::now();

            match req {
                NodeRequest::GetHeaders(hashes) => {
                    let headers = hashes
                        .iter()
                        .filter_map(|h| self.headers.iter().find(|x| x.block_hash() == *h))
                        .copied()
                        .collect();

                    let peer_msg = PeerMessages::Headers(headers);
                    self.node_tx
                        .send(NodeNotification::FromPeer(self.peer_id, peer_msg, now))
                        .unwrap();
                }
                NodeRequest::GetUtreexoState((hash, _)) => {
                    let accs = self.accs.get(&hash).unwrap().clone();

                    let peer_msg = PeerMessages::UtreexoState(accs);
                    self.node_tx
                        .send(NodeNotification::FromPeer(self.peer_id, peer_msg, now))
                        .unwrap();
                }
                NodeRequest::GetBlock(hashes, _) => {
                    for hash in hashes {
                        let block = self.blocks.get(&hash).unwrap().clone();

                        let peer_msg = PeerMessages::Block(block);
                        self.node_tx
                            .send(NodeNotification::FromPeer(self.peer_id, peer_msg, now))
                            .unwrap();
                    }
                }
                NodeRequest::Shutdown => {
                    break;
                }
                NodeRequest::GetBlockProof((block_hash, _, _)) => {
                    let proof = UtreexoProof {
                        block_hash,
                        leaf_data: vec![],
                        targets: vec![],
                        proof_hashes: vec![],
                    };

                    let peer_msg = PeerMessages::UtreexoProof(proof);
                    self.node_tx
                        .send(NodeNotification::FromPeer(self.peer_id, peer_msg, now))
                        .unwrap();
                }
                _ => {}
            }
        }

        self.node_tx
            .send(NodeNotification::FromPeer(
                self.peer_id,
                PeerMessages::Disconnected(self.peer_id as usize),
                Instant::now(),
            ))
            .unwrap();
    }
}

pub fn spawn_peer(
    peer_data: PeerData,
    node_sender: UnboundedSender<NodeNotification>,
    peer_id: u32,
) -> LocalPeerView {
    let (sender, node_rcv) = unbounded_channel();
    let PeerData {
        headers,
        blocks,
        accs,
    } = peer_data;

    let mut peer = SimulatedPeer::new(headers, blocks, accs, node_sender, node_rcv, peer_id);
    task::spawn(async move {
        peer.run().await;
    });

    LocalPeerView {
        message_times: Ema::with_half_life_50(),
        address: "127.0.0.1:8333".parse().unwrap(),
        services: simulated_peer_services(),
        user_agent: "/utreexo:0.1.0/".to_string(),
        height: 0,
        time_offset: 0,
        state: PeerStatus::Ready,
        channel: sender,
        kind: ConnectionKind::OutboundFullRelay(service_flags::UTREEXO.into()),
        banscore: 0,
        _last_message: Instant::now(),
        transport_protocol: TransportProtocol::V2,
    }
}

pub fn get_node_config(
    datadir: impl AsRef<Path>,
    network: Network,
    pow_fraud_proofs: bool,
) -> UtreexoNodeConfig {
    UtreexoNodeConfig {
        network,
        pow_fraud_proofs,
        datadir: datadir.as_ref().into(),
        user_agent: "node_test".to_string(),
        ..Default::default()
    }
}

pub fn serialize(root: UtreexoRoots) -> Vec<u8> {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&(root.numleaves as u64).to_le_bytes());

    for root_hash in root.roots.unwrap() {
        let bytes = Vec::from_hex(&root_hash).unwrap();
        buffer.extend_from_slice(&bytes);
    }

    buffer
}

pub fn create_false_acc(tip: usize) -> Vec<u8> {
    let mut bytes = [0u8; 32];
    let mut rng = UnwrapErr(SysRng);
    rng.fill_bytes(&mut bytes);
    let node_hash = encode::serialize_hex(&bytes);

    let utreexo_root = UtreexoRoots {
        roots: Some(vec![node_hash]),
        numleaves: tip,
    };

    serialize(utreexo_root)
}

/// Returns the first 2016 signet headers
pub fn signet_headers() -> Vec<Header> {
    let mut headers: Vec<Header> = Vec::new();

    let file = include_bytes!("../../../../../floresta-chain/testdata/signet_headers.zst");
    let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
    let mut buffer = uncompressed.as_slice();

    while let Ok(header) = Header::consensus_decode(&mut buffer) {
        headers.push(header);
    }

    headers
}

pub fn mainnet_headers() -> Vec<Header> {
    let mut headers: Vec<Header> = Vec::new();

    let file = include_bytes!("../../../../../floresta-chain/testdata/headers.zst");
    let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
    let mut buffer = uncompressed.as_slice();

    while let Ok(header) = Header::consensus_decode(&mut buffer) {
        headers.push(header);
    }

    headers
}

/// Returns the first 121 signet blocks, including genesis
pub fn signet_blocks() -> HashMap<BlockHash, Block> {
    let file = include_str!("../test_data/blocks.json");
    let entries: Vec<serde_json::Value> = serde_json::from_str(file).unwrap();

    entries
        .iter()
        .map(|e| {
            let str = e["block"].as_str().unwrap();
            let block: Block = deserialize_hex(str).unwrap();
            (block.block_hash(), block)
        })
        .collect()
}

/// Returns the first 120 signet accumulators. The genesis hash doesn't have a value since those
/// coinbase coins are unspendable.
pub fn signet_roots() -> HashMap<BlockHash, Vec<u8>> {
    let file = include_str!("../test_data/roots.json");
    let roots: Vec<UtreexoRoots> = serde_json::from_str(file).unwrap();

    let headers = signet_headers();
    let mut accs = HashMap::new();

    for root in roots.into_iter() {
        // For empty signet blocks numleaves equals the height; the genesis coins are unspendable,
        // so at height 1 we have one leaf, and so on as long as blocks have only one coinbase UTXO
        let height = root.numleaves;

        accs.insert(headers[height].block_hash(), serialize(root));
    }
    accs
}

/// Flips a bit in the first output script, invalidating the block's Merkle root.
pub fn mutate_block(block: &mut Block) {
    block.txdata[0].output[0].script_pubkey.as_mut_bytes()[0] ^= 1;
}

/// Builds a coinbase-only block that isn't a valid block, but passes the
/// mutated-block checks: with a single transaction the merkle root is its txid, and
/// since no transaction uses witness, the witness commitment is optional.
pub fn synthetic_block() -> Block {
    let coinbase = Transaction {
        version: TransactionVersion::TWO,
        lock_time: LockTime::ZERO,
        input: vec![TxIn {
            previous_output: OutPoint::null(),
            script_sig: ScriptBuf::new(),
            sequence: Sequence::MAX,
            witness: Witness::default(),
        }],
        output: vec![TxOut {
            value: Amount::from_sat(50),
            script_pubkey: ScriptBuf::new(),
        }],
    };

    let merkle_root = TxMerkleNode::from_byte_array(coinbase.compute_txid().to_byte_array());

    Block {
        header: Header {
            version: BlockVersion::TWO,
            prev_blockhash: BlockHash::from_byte_array([0; 32]),
            merkle_root,
            time: 0,
            bits: CompactTarget::from_consensus(0x207f_ffff),
            nonce: 0,
        },
        txdata: vec![coinbase],
    }
}

// Nightly Clippy false positive in `Constructor`-generated code:
// https://github.com/rust-lang/rust-clippy/issues/17525
#[allow(clippy::redundant_field_names)]
#[derive(Clone, Constructor)]
/// The chain data that our simulated peer will have
pub struct PeerData {
    headers: Vec<Header>,
    blocks: HashMap<BlockHash, Block>,
    accs: HashMap<BlockHash, Vec<u8>>,
}

// Nightly Clippy false positive in `Constructor`-generated code:
// https://github.com/rust-lang/rust-clippy/issues/17525
#[allow(clippy::redundant_field_names)]
#[derive(Constructor)]
/// The arguments needed to set up the test `UtreexoNode`
pub struct SetupNodeArgs {
    peers: Vec<PeerData>,
    pow_fraud_proofs: bool,
    network: Network,
    datadir: String,
    num_blocks: usize,
}

type Chain = Arc<ChainState<FlatChainStore>>;

/// Builds and returns a node with spawned simulated peers. The caller can run its event loop.
pub fn setup_node<T>(args: SetupNodeArgs) -> UtreexoNode<Chain, T>
where
    T: 'static + Default + NodeContext,
{
    setup_node_with_assume_valid(args, AssumeValidArg::Disabled)
}

/// Builds a test node with the given AssumeValid setting.
pub fn setup_node_with_assume_valid<T>(
    args: SetupNodeArgs,
    assume_valid: AssumeValidArg,
) -> UtreexoNode<Chain, T>
where
    T: 'static + Default + NodeContext,
{
    let net = args.network;
    let datadir = args.datadir;

    // Small test databases avoid exhausting disk space on Windows
    let config = FlatChainStoreConfig {
        block_index_size: Some(32_768),
        headers_file_size: Some(32_768),
        ..FlatChainStoreConfig::new(&datadir)
    };
    // Create `ChainState` and add headers to it
    let chainstore = FlatChainStore::new(config).unwrap();
    let chain = Arc::new(ChainState::open(chainstore, net, assume_valid).unwrap());

    let headers = match net {
        Network::Signet => signet_headers(),
        Network::Bitcoin => mainnet_headers(),
        Network::Regtest => Vec::new(),
        _ => panic!("unavailable headers for net: {net}"),
    };
    for header in headers.into_iter().skip(1).take(args.num_blocks) {
        chain.accept_header(header).unwrap();
    }

    // Create `UtreexoNode` and spawn the simulated peers
    let config = get_node_config(datadir, net, args.pow_fraud_proofs);
    let mempool = Arc::new(Mutex::new(Mempool::new(1000)));
    let kill_signal = Arc::new(RwLock::new(false));
    let addr_man = AddressMan::new(None, &[]);
    let mut node = UtreexoNode::new(config, chain, mempool, None, kill_signal, addr_man).unwrap();

    for (i, peer_data) in args.peers.into_iter().enumerate() {
        let peer_id = i as u32;
        let peer = spawn_peer(peer_data, node.node_tx.clone(), peer_id);

        // Add a fixed peer to avoid opening real P2P connections
        if i == 0 {
            node.fixed_peers = vec![peer.address.clone()];
        }

        node.peers.insert(peer_id, peer);

        // Populate the peer services too
        for service in [
            service_flags::UTREEXO.into(),
            ServiceFlags::COMPACT_FILTERS,
            ServiceFlags::NETWORK,
        ] {
            node.peer_by_service
                .entry(service)
                .or_default()
                .push(peer_id);
        }

        // This allows the node to properly assign a message time for the peer
        node.inflight.insert(
            InflightRequests::Connect(peer_id),
            (peer_id, Instant::now()),
        );
    }

    node
}

const NODE_TIMEOUT: Duration = Duration::from_secs(100);

pub async fn setup_sync_node(args: SetupNodeArgs) -> Arc<ChainState<FlatChainStore>> {
    let node = setup_node::<SyncNode>(args);
    let chain = node.chain.clone();

    timeout(NODE_TIMEOUT, node.run(|_| {})).await.unwrap();

    chain
}

pub async fn setup_swiftsync(
    args: SetupNodeArgs,
    assume_valid: AssumeValidArg,
) -> Arc<ChainState<FlatChainStore>> {
    let node = setup_node_with_assume_valid::<SwiftSync>(args, assume_valid);
    let chain = node.chain.clone();

    timeout(NODE_TIMEOUT, node.run(|_| {}))
        .await
        .unwrap()
        .unwrap();

    chain
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;
    use std::time::Instant;

    use bitcoin::BlockHash;
    use bitcoin::Network;
    use bitcoin::consensus::deserialize;
    use bitcoin::hashes::Hash;
    use bitcoin::p2p::ServiceFlags;
    use floresta_chain::AssumeValidArg;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;
    use floresta_chain::FlatChainStoreConfig;
    use floresta_common::Ema;
    use floresta_common::bhash;
    use floresta_common::service_flags;
    use floresta_mempool::Mempool;
    use hintsfile::Hintsfile;
    use tokio::sync::Mutex;
    use tokio::sync::RwLock;
    use tokio::sync::mpsc::unbounded_channel;

    use super::TransportProtocol;
    use super::get_node_config;
    use super::mutate_block;
    use super::signet_blocks;
    use super::signet_headers;
    use super::signet_roots;
    use crate::p2p_wire::address_man::AddressMan;
    use crate::p2p_wire::address_man::LocalAddress;
    use crate::p2p_wire::node::ConnectionKind;
    use crate::p2p_wire::node::LocalPeerView;
    use crate::p2p_wire::node::PeerStatus;
    use crate::p2p_wire::node::UtreexoNode;
    use crate::p2p_wire::node::sync_ctx::SyncNode;

    fn load_test_hints() -> Hintsfile {
        let mut file = File::open("./src/p2p_wire/tests/test_data/bitcoin.hints").unwrap();
        Hintsfile::from_reader(&mut file).unwrap()
    }

    #[test]
    #[should_panic]
    fn test_hints_file_genesis() {
        let hints = load_test_hints();
        hints.indices_at_height(0).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_hints_file_after_stop_height() {
        let hints = load_test_hints();
        hints.indices_at_height(176).unwrap();
    }

    #[test]
    fn test_hints_file_shape() {
        let hints = load_test_hints();
        assert_eq!(hints.stop_height(), 175);

        for height in 1..=175 {
            let unspent_indices = match height {
                9 => Vec::new(),      // The single UTXO in this block is spent later
                170 => vec![0, 1, 2], // Contains the transaction spending the height-9 UTXO
                _ => vec![0],         // Other blocks have just a coinbase output (here unspent)
            };

            assert_eq!(hints.indices_at_height(height).unwrap(), unspent_indices);
        }
    }

    #[test]
    fn test_get_headers_and_blocks() {
        let headers = signet_headers();
        let blocks = signet_blocks();

        assert_eq!(headers.len(), 2016);
        assert_eq!(blocks.len(), 121); // including genesis, up to height 120

        // Sanity check
        let mut prev_hash = BlockHash::all_zeros();
        for (i, header) in headers.iter().enumerate() {
            let hash = header.block_hash();

            let Some(block) = blocks.get(&hash) else {
                if i < 121 {
                    panic!("We should have a block at height {i}");
                }
                break;
            };

            assert_eq!(*header, block.header, "hashmap links to the correct block");
            assert!(block.check_merkle_root(), "valid txdata");
            assert_eq!(header.prev_blockhash, prev_hash, "valid hash chain");
            prev_hash = hash;
        }
    }

    #[test]
    fn test_get_mutated_block() {
        let hash = bhash!("000002c45c8ea9e553d4b0ee5d50324e56fc76f13019873fe707ff44fc56183f");
        let blocks = signet_blocks();

        let mut block_25 = blocks.get(&hash).unwrap().clone();
        mutate_block(&mut block_25);

        assert!(!block_25.txdata.is_empty(), "at least one tx");
        assert!(
            !block_25.check_merkle_root(),
            "invalid merkle root (txdata was tampered with)",
        );

        let headers = signet_headers();
        assert_eq!(
            block_25.header.prev_blockhash,
            headers[24].block_hash(),
            "block is at height 25",
        );
    }

    #[test]
    fn test_get_accs() {
        let accs = signet_roots();
        assert_eq!(accs.len(), 120, "we have roots starting from height 1");

        for (i, header) in signet_headers().iter().enumerate().skip(1).take(120) {
            let acc = accs.get(&header.block_hash()).unwrap();

            let leaves: u64 = deserialize(acc.clone().drain(0..8).as_slice()).unwrap();
            assert_eq!(i as u64, leaves, "one leaf added per block");
        }
    }

    #[tokio::test]
    async fn test_save_utreexo_peers_node_level() {
        let tempdir = tempfile::tempdir().unwrap();
        let config = FlatChainStoreConfig::new(tempdir.path());
        let chainstore = FlatChainStore::new(config).unwrap();
        let mempool = Arc::new(Mutex::new(Mempool::new(1000)));
        let chain =
            ChainState::open(chainstore, Network::Regtest, AssumeValidArg::Disabled).unwrap();
        let chain = Arc::new(chain);

        let config = get_node_config(tempdir.path(), Network::Regtest, false);
        let kill_signal = Arc::new(RwLock::new(false));
        let mut node = UtreexoNode::<Arc<ChainState<FlatChainStore>>, SyncNode>::new(
            config,
            chain,
            mempool,
            None,
            kill_signal,
            AddressMan::new(None, &[]),
        )
        .unwrap();

        fn make_peer(addr: &str, state: PeerStatus, services: ServiceFlags) -> LocalPeerView {
            let (sender, _receiver) = unbounded_channel();
            let mut address: LocalAddress = addr.parse().unwrap();
            address.set_services(services);
            LocalPeerView {
                message_times: Ema::with_half_life_50(),
                address,
                services,
                user_agent: "/utreexo:0.1.0/".to_string(),
                height: 0,
                time_offset: 0,
                state,
                channel: sender,
                kind: ConnectionKind::OutboundFullRelay(services),
                banscore: 0,
                _last_message: Instant::now(),
                transport_protocol: TransportProtocol::V2,
            }
        }

        // 1. Peer 0: Ready Utreexo peer -> Should be saved
        let p0 = make_peer(
            "127.0.0.1:8333",
            PeerStatus::Ready,
            service_flags::UTREEXO.into(),
        );
        let p0_addr = p0.address.clone();
        node.peers.insert(0, p0);

        // 2. Peer 1: Awaiting (unhandshaked) Utreexo peer -> Should be ignored
        let p1 = make_peer(
            "127.0.0.2:8333",
            PeerStatus::Awaiting,
            service_flags::UTREEXO.into(),
        );
        node.peers.insert(1, p1);

        // 3. Peer 2: Banned Utreexo peer -> Should be ignored
        let p2 = make_peer(
            "127.0.0.3:8333",
            PeerStatus::Banned,
            service_flags::UTREEXO.into(),
        );
        node.peers.insert(2, p2);

        // 4. Peer 3: Ready non-Utreexo peer -> Should be ignored
        let p3 = make_peer("127.0.0.4:8333", PeerStatus::Ready, ServiceFlags::NETWORK);
        node.peers.insert(3, p3);

        // Execute save_utreexo_peers on the node
        node.save_utreexo_peers().unwrap();

        // Read anchors.json back using start_addr_man
        let loaded_anchors = node.address_man.start_addr_man(tempdir.path());

        // Assert that ONLY the ready Utreexo peer (Peer 0) was saved
        assert_eq!(loaded_anchors.len(), 1);
        assert_eq!(
            loaded_anchors[0].get_socket_addr(),
            p0_addr.get_socket_addr()
        );
        assert!(
            loaded_anchors[0]
                .get_services()
                .has(service_flags::UTREEXO.into())
        );
    }
}

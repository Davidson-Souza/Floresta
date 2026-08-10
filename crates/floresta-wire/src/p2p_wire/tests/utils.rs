// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::block::Header;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::encode;
use bitcoin::consensus::encode::deserialize_hex;
use bitcoin::hex::FromHex;
use bitcoin::p2p::ServiceFlags;
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
use crate::node::sync_ctx::SyncNode;
use crate::p2p_wire::block_proof::UtreexoProof;
use crate::p2p_wire::peer::PeerMessages;
use crate::p2p_wire::peer::Version;
use crate::p2p_wire::transport::TransportProtocol;

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

impl SimulatedPeer {
    pub async fn run(&mut self) {
        let version = Version {
            user_agent: "node_test".to_string(),
            protocol_version: 0,
            blocks: rand::random::<u32>() % 23,
            id: self.peer_id,
            address_id: rand::random::<u64>() as usize,
            services: ServiceFlags::NETWORK
                | service_flags::UTREEXO.into()
                | service_flags::UTREEXO_ARCHIVE.into()
                | ServiceFlags::WITNESS
                | ServiceFlags::COMPACT_FILTERS,
            time_offset: 0,
            kind: ConnectionKind::Regular(service_flags::UTREEXO.into()),
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
                NodeRequest::GetBlock(hashes) => {
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

pub fn create_peer(
    headers: Vec<Header>,
    blocks: HashMap<BlockHash, Block>,
    accs: HashMap<BlockHash, Vec<u8>>,
    node_sender: UnboundedSender<NodeNotification>,
    sender: UnboundedSender<NodeRequest>,
    node_rcv: UnboundedReceiver<NodeRequest>,
    peer_id: u32,
) -> LocalPeerView {
    let mut peer = SimulatedPeer::new(headers, blocks, accs, node_sender, node_rcv, peer_id);
    task::spawn(async move {
        peer.run().await;
    });

    LocalPeerView {
        message_times: Ema::with_half_life_50(),
        address: "127.0.0.1:8333".parse().unwrap(),
        services: service_flags::UTREEXO.into(),
        user_agent: "/utreexo:0.1.0/".to_string(),
        height: 0,
        time_offset: 0,
        state: PeerStatus::Ready,
        channel: sender,
        kind: ConnectionKind::Regular(service_flags::UTREEXO.into()),
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
        max_tip_age_secs: u32::MAX,
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

    let file = include_bytes!("../../../../floresta-chain/testdata/signet_headers.zst");
    let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
    let mut buffer = uncompressed.as_slice();

    while let Ok(header) = Header::consensus_decode(&mut buffer) {
        headers.push(header);
    }

    headers
}

/// Returns the first 121 signet blocks, including genesis
pub fn signet_blocks() -> HashMap<BlockHash, Block> {
    let file = include_str!("./test_data/blocks.json");
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
    let file = include_str!("./test_data/roots.json");
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

/// Returns a mutated signet block at height 7
pub fn mutated_block_h7() -> Block {
    deserialize_hex(
        "00000020daf3b60d374b19476461f97540498dcfa2eb7016238ec6b1d022f82fb60100007a7ae65b53cb988c2ec92d2384996713821d5645ffe61c9acea60da75cd5edfa1a944d5fae77031e9dbb050001010000000001010000000000000000000000000000000000000000000000000000000000000000ffffffff025751feffffff0200f2052a01000000160014ef2dceae02e35f8137de76768ae3345d99ca68860000000000000000776a24aa21a9ede2f61c3f71d1defd3fa999dfa36953755c690689799962b48bebd836974e8cf94c4fecc7daa2490047304402202b3f946d6447f9bf17d00f3696cede7ee70b785495e5498274ee682a493befd5022045fc0bcf9331073168b5d35507175f9f374a8eba2336873885d12aada67ea5f601000120000000000000000000000000000000000000000000000000000000000000000000000000"
    ).unwrap()
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

pub async fn setup_node(
    peers: Vec<PeerData>,
    pow_fraud_proofs: bool,
    network: Network,
    datadir: impl AsRef<Path>,
    num_blocks: usize,
) -> Arc<ChainState<FlatChainStore>> {
    let config = FlatChainStoreConfig::new(&datadir);

    let chainstore = FlatChainStore::new(config).unwrap();
    let mempool = Arc::new(Mutex::new(Mempool::new(1000)));
    let chain = ChainState::open(chainstore, network, AssumeValidArg::Disabled).unwrap();
    let chain = Arc::new(chain);

    let mut headers = signet_headers();
    headers.remove(0);
    headers.truncate(num_blocks);
    for header in headers {
        chain.accept_header(header).unwrap();
    }

    let config = get_node_config(&datadir, network, pow_fraud_proofs);
    let kill_signal = Arc::new(RwLock::new(false));
    let mut node = UtreexoNode::<Arc<ChainState<FlatChainStore>>, SyncNode>::new(
        config,
        chain.clone(),
        mempool,
        None,
        kill_signal.clone(),
        AddressMan::new(None, &[]),
    )
    .unwrap();

    for (i, peer) in peers.into_iter().enumerate() {
        let (sender, receiver) = unbounded_channel();
        let peer_id = i as u32;

        let peer = create_peer(
            peer.headers,
            peer.blocks,
            peer.accs,
            node.node_tx.clone(),
            sender.clone(),
            receiver,
            peer_id,
        );

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

    timeout(Duration::from_secs(100), node.run(|_| {}))
        .await
        .unwrap();

    chain
}

#[cfg(test)]
mod tests {
    use bitcoin::BlockHash;
    use bitcoin::consensus::deserialize;
    use bitcoin::hashes::Hash;

    use super::mutated_block_h7;
    use super::signet_blocks;
    use super::signet_headers;
    use super::signet_roots;

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
        let mutated_block = mutated_block_h7();
        assert!(!mutated_block.txdata.is_empty(), "at least one tx");

        assert!(!mutated_block.check_merkle_root(), "invalid merkle root");

        let headers = signet_headers();
        assert_eq!(
            mutated_block.header.prev_blockhash,
            headers[6].block_hash(),
            "invalid block is at height 7",
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
}

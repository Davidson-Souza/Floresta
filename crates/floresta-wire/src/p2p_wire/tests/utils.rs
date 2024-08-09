use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::Cursor;
use std::io::Read;
use std::str::FromStr;
use std::time::Instant;

use bitcoin::block::Header;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::Decodable;
use bitcoin::hex::FromHex;
use bitcoin::p2p::ServiceFlags;
use bitcoin::BlockHash;
use floresta_chain::UtreexoBlock;
use hex;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task;
use zstd;

use crate::node::LocalPeerView;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::PeerStatus;
use crate::p2p_wire::peer::PeerMessages;
use crate::p2p_wire::peer::Version;
use crate::UtreexoNodeConfig;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct UtreexoRoots {
    roots: Option<Vec<String>>,
    numleaves: usize,
}

#[derive(Deserialize, Debug)]
struct Block {
    block: String,
}

#[derive(Debug)]
pub struct TestPeer {
    headers: Vec<Header>,
    blocks: HashMap<BlockHash, UtreexoBlock>,
    filters: HashMap<BlockHash, Vec<u8>>,
    node_tx: UnboundedSender<NodeNotification>,
    node_rx: UnboundedReceiver<NodeRequest>,
    peer_id: u32,
}

impl TestPeer {
    pub fn new(
        node_tx: UnboundedSender<NodeNotification>,
        headers: Vec<Header>,
        blocks: HashMap<BlockHash, UtreexoBlock>,
        filters: HashMap<BlockHash, Vec<u8>>,
        node_rx: UnboundedReceiver<NodeRequest>,
        peer_id: u32,
    ) -> Self {
        TestPeer {
            headers,
            blocks,
            filters,
            node_tx,
            node_rx,
            peer_id,
        }
    }

    pub async fn run(&mut self) {
        let version = Version {
            user_agent: "node_test".to_string(),
            protocol_version: 0,
            blocks: rand::random::<u32>() % 23,
            id: self.peer_id as u32,
            address_id: rand::random::<usize>(),
            services: ServiceFlags::UTREEXO
                | ServiceFlags::NETWORK
                | ServiceFlags::WITNESS
                | ServiceFlags::COMPACT_FILTERS
                | ServiceFlags::from(1 << 25),
            feeler: false,
        };

        self.node_tx
            .send(NodeNotification::FromPeer(
                self.peer_id,
                PeerMessages::Ready(version),
            ))
            .unwrap();

        loop {
            let req = self.node_rx.recv().await.unwrap();

            match req {
                NodeRequest::GetBlock((hashes, _)) => {
                    for hash in hashes {
                        let block = self.blocks.get(&hash).unwrap().clone();
                        self.node_tx
                            .send(NodeNotification::FromPeer(
                                self.peer_id,
                                PeerMessages::Block(block),
                            ))
                            .unwrap();
                    }
                }
                NodeRequest::Shutdown => {
                    break;
                }
                _ => {}
            }
        }

        self.node_tx
            .send(NodeNotification::FromPeer(
                self.peer_id,
                PeerMessages::Disconnected(self.peer_id as usize),
            ))
            .unwrap();
    }
}

pub fn create_peer(
    headers: Vec<Header>,
    blocks: HashMap<BlockHash, UtreexoBlock>,
    filters: HashMap<BlockHash, Vec<u8>>,
    node_sender: UnboundedSender<NodeNotification>,
    sender: UnboundedSender<NodeRequest>,
    node_rcv: UnboundedReceiver<NodeRequest>,
    peer_id: u32,
) -> LocalPeerView {
    let mut peer = TestPeer::new(node_sender, headers, blocks, filters, node_rcv, peer_id);
    task::spawn(async move {
        peer.run().await;
    });

    LocalPeerView {
        address: "127.0.0.1".parse().unwrap(),
        services: ServiceFlags::UTREEXO,
        user_agent: "/utreexo:0.1.0/".to_string(),
        height: 0,
        state: PeerStatus::Ready,
        channel: sender,
        port: 8333,
        feeler: false,
        banscore: 0,
        address_id: 0,
        _last_message: Instant::now(),
    }
}

pub fn get_node_config(
    datadir: String,
    network: floresta_chain::Network,
    pow_fraud_proofs: bool,
) -> UtreexoNodeConfig {
    UtreexoNodeConfig {
        network: network.into(),
        pow_fraud_proofs,
        compact_filters: false,
        fixed_peer: None,
        max_banscore: 100,
        max_outbound: 8,
        max_inflight: 10,
        datadir,
        proxy: None,
        assume_utreexo: None,
        backfill: false,
        filter_start_height: None,
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
    OsRng.fill_bytes(&mut bytes);
    let node_hash = hex::encode(bytes);

    let utreexo_root = UtreexoRoots {
        roots: Some(vec![node_hash]),
        numleaves: tip,
    };

    serialize(utreexo_root)
}

pub fn get_test_headers() -> Vec<Header> {
    let file =
        include_bytes!("../../../../floresta-chain/src/pruned_utreexo/testdata/signet_headers.zst");

    let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
    let mut cursor = Cursor::new(uncompressed);
    let mut headers: Vec<Header> = Vec::new();
    while let Ok(header) = Header::consensus_decode(&mut cursor) {
        headers.push(header);
    }
    headers
}

pub fn get_test_blocks() -> io::Result<HashMap<BlockHash, UtreexoBlock>> {
    let dir = "./src/p2p_wire/tests/test_data/blocks.json";
    let mut contents = String::new();
    File::open(dir)?.read_to_string(&mut contents)?;

    let blocks: Vec<Block> = serde_json::from_str(&contents).unwrap();
    let mut u_blocks = HashMap::new();

    for block_str in blocks {
        let block = Vec::from_hex(&block_str.block).unwrap();
        let block: UtreexoBlock = deserialize(&block).unwrap();
        u_blocks.insert(block.block.block_hash(), block);
    }

    Ok(u_blocks)
}

pub fn get_test_filters() -> io::Result<HashMap<BlockHash, Vec<u8>>> {
    let mut contents = String::new();
    File::open("./src/p2p_wire/tests/test_data/roots.json")
        .unwrap()
        .read_to_string(&mut contents)
        .unwrap();
    let roots: Vec<UtreexoRoots> = serde_json::from_str(&contents).unwrap();
    let headers = get_test_headers();
    let mut filters = HashMap::new();

    for root in roots.into_iter() {
        let buffer = serialize(root.clone());

        // Insert the serialised Utreexo-Root along with its corresponding BlockHash in the HashMap
        filters.insert(headers[root.numleaves].block_hash(), buffer);
    }
    Ok(filters)
}

pub fn generate_invalid_block() -> UtreexoBlock {
    let invalid_block_str = "00000020daf3b60d374b19476461f97540498dcfa2eb7016238ec6b1d022f82fb60100007a7ae65b53cb988c2ec92d2384996713821d5645ffe61c9acea60da75cd5edfa1a944d5fae77031e9dbb050001010000000001010000000000000000000000000000000000000000000000000000000000000000ffffffff025751feffffff0200f2052a01000000160014ef2dceae02e35f8137de76768ae3345d99ca68860000000000000000776a24aa21a9ede2f61c3f71d1defd3fa999dfa36953755c690689799962b48bebd836974e8cf94c4fecc7daa2490047304402202b3f946d6447f9bf17d00f3696cede7ee70b785495e5498274ee682a493befd5022045fc0bcf9331073168b5d35507175f9f374a8eba2336873885d12aada67ea5f60100012000000000000000000000000000000000000000000000000000000000000000000000000000000000";

    let block = Vec::from_hex(&invalid_block_str).unwrap();
    let block: UtreexoBlock = deserialize(&block).unwrap();

    block
}

pub fn get_essentials() -> (
    Vec<Header>,
    HashMap<BlockHash, UtreexoBlock>,
    HashMap<BlockHash, Vec<u8>>,
    BlockHash,
    UtreexoBlock,
) {
    let headers = get_test_headers();
    let blocks = get_test_blocks().unwrap();
    let true_filters = get_test_filters().unwrap();
    let invalid_block = generate_invalid_block();

    // BlockHash of chain_tip: 0000035f0e5513b26bba7cead874fdf06241a934e4bc4cf7a0381c60e4cdd2bb (119)
    let tip_hash =
        BlockHash::from_str("0000035f0e5513b26bba7cead874fdf06241a934e4bc4cf7a0381c60e4cdd2bb")
            .unwrap();

    (headers, blocks, true_filters, tip_hash, invalid_block)
}

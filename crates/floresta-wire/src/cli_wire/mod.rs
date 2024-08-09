use std::{
    collections::HashMap,
    io::BufReader,
    net::TcpStream,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

use bitcoin::{
    consensus::{deserialize, deserialize_partial, Encodable},
    hashes::{
        hex::{FromHex, ToHex},
        sha256, Hash,
    },
    Block, BlockHash, BlockHeader, OutPoint,
};
use btcd_rpc::{
    client::{BTCDClient, BtcdRpc},
    json_types::{blockchain::GetUtreexoProofResult, VerbosityOutput},
};
use log::{error, info};
use rustreexo::accumulator::proof::Proof;
use serde::Deserialize;

use super::{
    chain_state::ChainState, chainstore::KvChainStore, error::BlockchainError, udata::LeafData,
    BlockchainInterface, Result, UpdatableChainstate,
};
use crate::try_and_log;

pub struct UtreexodBackend {
    pub use_batch_sync: bool,
    pub batch_sync_hostname: String,
    pub rpc: Arc<BTCDClient>,
    pub chainstate: Arc<ChainState<KvChainStore>>,
    pub term: Arc<AtomicBool>,
}

impl UtreexodBackend {
    fn _get_block_hash(&self, height: u32) -> Result<bitcoin::BlockHash> {
        Ok(BlockHash::from_hex(
            self.rpc.getblockhash(height as usize)?.as_str(),
        )?)
    }
    fn get_height(&self) -> Result<u32> {
        let block = self.rpc.getbestblock()?;
        Ok(block.height as u32)
    }

    fn broadcast(&self, tx: &bitcoin::Transaction) -> Result<()> {
        let mut writer = Vec::new();
        let _ = tx
            .consensus_encode(&mut writer)
            .expect("Should be a valid transaction");
        if self.rpc.sendrawtransaction(writer.to_hex()).is_err() {
            error!("Error while broadcasting tx {}", tx.txid());
        }
        Ok(())
    }

    fn _estimate_fee(&self, target: usize) -> Result<f64> {
        let feerate = self.rpc.estimatefee(target as u32)?;
        Ok(feerate)
    }
    pub fn get_block(&self, height: u32) -> Result<Block> {
        let hash = self.rpc.getblockhash(height as usize)?;
        let block = self.rpc.getblock(hash, false)?;
        if let VerbosityOutput::Simple(hex) = block {
            let block = Vec::from_hex(hex.as_str())?;
            let (block, _): (Block, usize) = deserialize_partial(&block).unwrap();
            let validation = block.header.validate_pow(&block.header.target());
            assert!(validation.is_ok());
            return Ok(block);
        }
        Err(BlockchainError::BlockNotPresent)
    }
    pub fn get_proof<T: BtcdRpc>(
        rpc: &T,
        hash: &String,
    ) -> Result<(Proof, Vec<sha256::Hash>, Vec<LeafData>)> {
        let proof = rpc.getutreexoproof(hash.to_string(), true)?.get_verbose();
        Self::process_proof(proof)
    }
    fn process_proof(
        proof: GetUtreexoProofResult,
    ) -> Result<(Proof, Vec<sha256::Hash>, Vec<LeafData>)> {
        let preimages: Vec<_> = proof
            .target_preimages
            .iter()
            .map(|preimage| {
                deserialize_partial::<LeafData>(&Vec::from_hex(preimage).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let proof_hashes: Vec<_> = proof
            .proofhashes
            .iter()
            .map(|hash| sha256::Hash::from_hex(hash).unwrap())
            .collect();
        let targets = proof.prooftargets;

        let targethashes: Vec<_> = proof
            .targethashes
            .iter()
            .map(|hash| sha256::Hash::from_hex(hash).unwrap())
            .collect();
        let proof = Proof::new(targets, proof_hashes);

        Ok((proof, targethashes, preimages))
    }

    pub fn handle_broadcast(&self) -> Result<()> {
        let tx_list = self.chainstate.get_unbroadcasted();
        for tx in tx_list {
            self.broadcast(&tx)?;
        }
        Ok(())
    }
    pub async fn handle_tip_update(&self) -> Result<()> {
        let height = self.get_height()?;
        let local_best = self.chainstate.get_best_block().unwrap().0;
        if height > local_best {
            self.get_headers()?;
            self.download_blocks().await?;
        }
        Ok(())
    }
    fn process_block(&self, block_height: u32) -> Result<()> {
        let block = self.get_block(block_height)?;
        let (proof, del_hashes, leaf_data) =
            Self::get_proof(&*self.rpc, &block.block_hash().to_string())?;
        let mut inputs = HashMap::new();
        for tx in block.txdata.iter() {
            for (vout, out) in tx.output.iter().enumerate() {
                inputs.insert(
                    OutPoint {
                        txid: tx.txid(),
                        vout: vout as u32,
                    },
                    out.clone(),
                );
            }
        }
        for leaf in leaf_data {
            inputs.insert(leaf.prevout, leaf.utxo);
        }
        self.chainstate
            .connect_block(&block, proof, inputs, del_hashes)?;
        Ok(())
    }
    async fn download_blocks(&self) -> Result<()> {
        let height = self.get_height()?;
        let current = self.chainstate.get_validation_index()?;
        // We don't download genesis, because utreexod will error out if we try to fetch
        // proof for it.
        let current = if current == 0 { 1 } else { current };
        if self.chainstate.is_in_idb() {
            info!("Start Initial Block Download at height {current} of {height}");
        }
        for block_height in current..=height {
            if self.is_shutting_down() {
                return Ok(());
            }

            if block_height % 10_000 == 0 {
                info!("Sync at block {block_height}");
                if block_height % 100_000 == 0 {
                    self.chainstate.flush()?;
                }
            }
            self.process_block(block_height)?;
        }
        if self.chainstate.is_in_idb() {
            info!("Leaving Initial Block Download at height {height}");
        } else {
            info!("New tip: {height}");
        }
        self.chainstate.toggle_ibd(false);
        self.chainstate.flush()?;

        Ok(())
    }
    async fn process_batch_block(&self) -> Result<()> {
        let socket = TcpStream::connect(self.batch_sync_hostname.to_owned().as_str())?;

        let height = self.get_height()?;
        let current = self.chainstate.get_validation_index()?;
        let mut reader = BufReader::new(socket);
        for _ in (current + 1)..=height {
            if self.is_shutting_down() {
                return Ok(());
            }
            let block_data = rmp_serde::decode::from_read::<_, BlockData>(&mut reader);
            if block_data.is_err() {
                error!("{:?}", block_data);
                break;
            }
            let block_data = block_data.unwrap();
            let (proof, del_hashes, leaf_data) = Self::process_proof(block_data.proof)?;
            let mut inputs = HashMap::new();
            for tx in block_data.block.txdata.iter() {
                for (vout, out) in tx.output.iter().enumerate() {
                    inputs.insert(
                        OutPoint {
                            txid: tx.txid(),
                            vout: vout as u32,
                        },
                        out.clone(),
                    );
                }
            }
            if block_data.height % 10_000 == 0 {
                info!("Sync at block {}", block_data.height);
                self.chainstate.flush()?;
            }
            for leaf in leaf_data {
                inputs.insert(leaf.prevout, leaf.utxo);
            }
            self.chainstate
                .connect_block(&block_data.block, proof, inputs, del_hashes)?;
        }
        info!("Leaving ibd");
        self.chainstate.toggle_ibd(false);
        self.chainstate.flush()?;

        Ok(())
    }
    fn get_headers(&self) -> Result<()> {
        let tip = self.get_height()?;
        let current = self.chainstate.get_best_block()?.0;
        info!("Downloading headers");
        for i in (current / 2_000)..((tip / 2_000) + 1) {
            if self.is_shutting_down() {
                return Ok(());
            }
            let locator = self.chainstate.get_block_locator()?;
            let locator = locator
                .iter()
                .map(|hash| hash.to_string())
                .collect::<Vec<_>>();

            let headers = self
                .rpc
                .getheaders(locator, BlockHash::all_zeros().to_string())?;

            let headers = headers
                .iter()
                .map(|header| {
                    let header = Vec::from_hex(header).unwrap();
                    deserialize::<BlockHeader>(&header).unwrap()
                })
                .collect::<Vec<_>>();

            for header in headers {
                self.chainstate.accept_header(header)?;
            }
            self.chainstate.flush()?;
            info!(
                "Downloading headers...{:02.0}%",
                ((i * 2_000) as f64) / (tip as f64) * 100.00
            );
        }
        Ok(())
    }

    pub async fn run(self) {
        try_and_log!(self.get_headers());
        if self.is_shutting_down() {
            info!("Shuting blockchain down");
            try_and_log!(self.chainstate.flush());
            return;
        }
        if self.use_batch_sync {
            try_and_log!(self.process_batch_block().await);
        } else {
            try_and_log!(self.download_blocks().await);
        }
        self.chainstate.toggle_ibd(false);
        loop {
            std::thread::sleep(Duration::from_secs(1)).await;
            if self.is_shutting_down() {
                info!("Shuting blockchain down");
                try_and_log!(self.chainstate.flush());
                return;
            }
            try_and_log!(self.handle_broadcast());
            try_and_log!(self.handle_tip_update().await);
            try_and_log!(self.chainstate.flush());
        }
    }
    fn is_shutting_down(&self) -> bool {
        self.term
            .fetch_and(true, std::sync::atomic::Ordering::AcqRel)
    }
}
#[macro_export]
/// Attempt to execute a function, and logs the error, if any.
macro_rules! try_and_log {
    ($what: expr) => {
        let result = $what;
        if let Err(error) = result {
            log::error!("{:?}", error);
        }
    };
}
#[allow(unused)]
#[derive(Debug, Deserialize)]
struct BlockData {
    height: u32,
    block: Block,
    proof: GetUtreexoProofResult,
}

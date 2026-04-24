// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node that downloads and validates the blockchain, but skips utreexo proofs as they aren't
//! needed to validate the UTXO set with the SwiftSync method.

use std::collections::HashSet;
use std::fs::File;
use std::process::exit;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bitcoin::block::Header as BlockHeader;
use bitcoin::p2p::ServiceFlags;
use bitcoin::Amount;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::Transaction;
use bitcoin::TxOut;
use floresta_chain::pruned_utreexo::consensus::Consensus;
use floresta_chain::swift_sync_agg::SipHashKeys;
use floresta_chain::swift_sync_agg::SwiftSyncAgg;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use floresta_chain::ThreadSafeChain;
use floresta_common::service_flags;
use hintsfile::Hintsfile;
use rand::rngs::OsRng;
use rand::RngCore;
use rustreexo::stump::Stump;
use tokio::time;
use tokio::time::MissedTickBehavior;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::node::periodic_job;
use crate::node::try_and_log;
use crate::node::ConnectionKind;
use crate::node::InflightBlock;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::UtreexoNode;
use crate::node::WorkerResult;
use crate::node_context::LoopControl;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::p2p_wire::coinscache::CoinsCache;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::peer::PeerMessages;

/// [`SwiftSync`] is a node that downloads and validates the blockchain but skips utreexo
/// proofs by using SwiftSync.
///
/// This node implements:
///     - `NodeContext`
///     - `UtreexoNode<SwiftSync, Chain>`
#[derive(Default)]
pub struct SwiftSync {
    /// The `TxOut` aggregator.
    agg: SwiftSyncAgg,

    /// The secret salt used to compute the aggregator element hashes.
    salt: Arc<SipHashKeys>,

    /// The total unspent amount. Once we reach the SwiftSync stop height, this must be less or
    /// equal than the theoretical supply limit at that height.
    supply: Amount,

    /// The target height for the currently used SwiftSync hints.
    stop_height: u32,

    /// Height at which SwiftSync was aborted, if any.
    ///
    /// We abort when either the hints are found to be invalid or the current chain is invalid (we
    /// may find an invalid block or, at the end, a violation of the maximum supply limit).
    abort_height: Option<u32>,

    coins_cache: CoinsCache,
}

impl NodeContext for SwiftSync {
    fn get_required_services(&self) -> bitcoin::p2p::ServiceFlags {
        ServiceFlags::WITNESS | service_flags::UTREEXO.into() | ServiceFlags::NETWORK
    }

    const TRY_NEW_CONNECTION: u64 = 15; // We want to be well-connected early on
    const REQUEST_TIMEOUT: u64 = 2 * 60; // 2 minutes (5 blocks should reach us much faster)
    const MAX_INFLIGHT_REQUESTS: usize = 100; // double the default
    const MAX_OUTGOING_PEERS: usize = 30;
    const MAX_CONCURRENT_GETDATA: usize = 40; // 40 * 5 = 200 blocks in parallel
    const ASSUME_STALE: u64 = 2 * 60; // Two minutes without blocks while in IBD is very suspicious

    // A more conservative value than the default of 1 second, since we'll have many peer messages
    const MAINTENANCE_TICK: Duration = Duration::from_secs(5);
}

// This is more than enough to avoid CPU from ever becoming a bottleneck
const MAX_PARALLEL_WORKERS: usize = 6;

/// Node methods for a [`UtreexoNode`] where its Context is [`SwiftSync`].
/// See [node](crates/floresta-wire/src/p2p_wire/node.rs) for more information.
impl<Chain> UtreexoNode<Chain, SwiftSync>
where
    Chain: ThreadSafeChain,
    WireError: From<Chain::Error>,
{
    /// Parses the SwiftSync hints file and returns an in-memory [`Hintsfile`] representation.
    fn parse_hints_file(datadir: &str, network: Network) -> Hintsfile {
        let path = format!("{datadir}/{network}.hints");

        let mut file = File::open(path).expect("invalid hints file path");
        Hintsfile::from_reader(&mut file).expect("couldn't read hints file")
    }

    /// Generates a random salt for this SwiftSync session.
    fn generate_salt() -> Arc<SipHashKeys> {
        let mut rng = OsRng;
        Arc::new(SipHashKeys::new(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ))
    }

    /// Returns `true` if SwiftSync failed, due to the hints being invalid or the current chain
    /// being invalid (below the SwiftSync stop height).
    pub(crate) fn was_aborted(&self) -> bool {
        self.context.abort_height.is_some()
    }

    /// Computes the next blocks to request, and sends a GETDATA request, advancing
    /// `last_block_request` up to the SwiftSync hints `stop_height`.
    fn get_blocks_to_download(&mut self) {
        // If this request would make our inflight queue too long, postpone it
        if !self.can_request_more_blocks() || self.was_aborted() {
            return;
        }

        let prev_last_request = self.last_block_request;
        let mut blocks = Vec::with_capacity(SwiftSync::BLOCKS_PER_GETDATA);

        for _ in 0..SwiftSync::BLOCKS_PER_GETDATA {
            let next_height = self.last_block_request + 1;

            if next_height > self.context.stop_height {
                // We need to reach it but not exceed it
                break;
            }

            let Ok(next_block) = self.chain.get_block_hash(next_height) else {
                // Likely end of chain (e.g., `BlockNotPresent`)
                break;
            };

            blocks.push(next_block);
            self.last_block_request += 1;
        }

        if let Err(e) = self.request_blocks(blocks) {
            // Rollback so we can retry the same heights next time.
            error!("Failed to request blocks: {e:?}");
            self.last_block_request = prev_last_request;
        }
        // If `request_blocks` succeeds, we will keep track of the requests in `self.inflight`,
        // so even if the remote peer disconnects, we can still re-request them.
    }

    /// Starts SwiftSync processing for up to `MAX_PARALLEL_WORKERS` pending blocks.
    fn pump_swiftsync(&mut self, hints: &mut Hintsfile) -> Result<(), WireError> {
        let processing = self
            .blocks
            .values()
            .filter(|b| b.processing_since.is_some())
            .count();

        let free = MAX_PARALLEL_WORKERS.saturating_sub(processing);
        if free == 0 {
            return Ok(());
        }

        // Collect hashes first (can't mutate the map while iterating it)
        let to_process: Vec<BlockHash> = self
            .blocks
            .iter()
            .filter(|(_, b)| b.processing_since.is_none())
            .take(free) // We don't exceed MAX_PARALLEL_WORKERS
            .map(|(h, _)| *h)
            .collect();

        for hash in to_process {
            // Prefer storing height in the entry to avoid repeated chain lookups
            let height = self
                .chain
                .get_block_height(&hash)?
                // NOTE: if a previous block was invalid, we will get this error
                .ok_or(BlockchainError::OrphanOrInvalidBlock)?;

            self.start_processing_swiftsync(hash, height, hints)?;
        }

        Ok(())
    }

    /// Spawns a blocking task to process a block with the provided SwiftSync hints.
    fn start_processing_swiftsync(
        &mut self,
        block_hash: BlockHash,
        block_height: u32,
        hints: &mut Hintsfile,
    ) -> Result<(), WireError> {
        debug!("processing block {block_hash}");
        let entry = self
            .blocks
            .get_mut(&block_hash)
            .ok_or(WireError::BlockNotFound)?;

        if entry.processing_since.is_some() {
            return Ok(()); // already being processed
        }

        let Some(block_hints) = hints.take_indices(block_height) else {
            error!("We tried processing block {block_height} but its hints are missing");
            return Ok(());
        };
        let unspent_indexes: HashSet<u32> = block_hints.into_iter().collect();

        // Start the processing timer
        entry.processing_since = Some(Instant::now());

        let block = Arc::clone(&entry.block);
        let consensus = Consensus::from(self.network);
        let salt = Arc::clone(&self.context.salt);

        // If we find a very cheap block (e.g., ~10μs), it's faster to process it directly
        if block.txdata.len() == 1 {
            let result =
                consensus.process_block_swiftsync(&block, block_height, unspent_indexes, &salt);

            self.handle_worker_notification(result, block_hash, block_height, hints)?;
            return Ok(());
        }

        let node_sender = self.node_tx.clone();
        tokio::task::spawn_blocking(move || {
            let result =
                consensus.process_block_swiftsync(&block, block_height, unspent_indexes, &salt);

            let notification = NodeNotification::FromWorker((result, block_hash, block_height));
            let _ = node_sender.send(notification);
        });

        Ok(())
    }

    /// Starts the SwiftSync node by updating the last block requested and starting the main loop.
    /// This loop to the following tasks, in order:
    ///   - Receives messages from our peers through the node_tx channel, and handles them.
    ///   - Checks if the kill signal is set, and if so breaks the loop.
    ///   - Checks if we have downloaded and processed all blocks, and verifies that the aggregator
    ///     is zero. If so, we are done.
    ///   - Checks if our last validation update was long ago and creates an extra connection.
    ///   - Handles timeouts for inflight requests.
    ///   - If we are low on inflights, requests new blocks to validate.
    pub async fn run(mut self, done_cb: impl FnOnce(&Chain)) -> Self {
        info!("Starting SwiftSync node...");
        self.last_block_request = self.chain.get_validation_index().unwrap();
        assert_eq!(self.last_block_request, 0);

        // Parse the hints file and randomly fill the SwiftSync salt for this session
        let mut hints = Self::parse_hints_file(&self.datadir, self.network);

        self.context.stop_height = hints.stop_height();
        self.context.salt = Self::generate_salt();

        info!("Performing SwiftSync up to height {}", hints.stop_height());

        let mut ticker = time::interval(SwiftSync::MAINTENANCE_TICK);
        // If we fall behind, don't "catch up" by running maintenance repeatedly
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;

                // Maintenance runs only on tick but has priority
                _ = ticker.tick() => match self.maintenance_tick(&mut hints).await {
                    LoopControl::Continue => {},
                    LoopControl::Break => break,
                },

                // Handle messages as soon as we find any, otherwise sleep until maintenance
                msg = self.node_rx.recv() => {
                    let Some(msg) = msg else {
                        break;
                    };
                    // We only update the aggregator when reading responses from the workers
                    try_and_log!(self.handle_message(msg, &mut hints).await);

                    // Drain all queued messages
                    while let Ok(msg) = self.node_rx.try_recv() {
                        try_and_log!(self.handle_message(msg, &mut hints).await);
                    }
                    if *self.kill_signal.read().await {
                        break;
                    }
                }
            }
        }

        done_cb(&self.chain);
        self
    }

    /// Performs the periodic maintenance tasks, including checking for the cancel signal, peer
    /// connections, and inflight request timeouts.
    ///
    /// Returns `LoopControl::Break` if we need to break the main `SwiftSync` loop, which may
    /// happen if the kill signal was set, we successfully finished SwiftSync, or we need to abort
    /// operation due to a validation error.
    async fn maintenance_tick(&mut self, hints: &mut Hintsfile) -> LoopControl {
        if *self.kill_signal.read().await {
            return LoopControl::Break;
        }

        if let Some(invalid_h) = self.context.abort_height {
            // All our progress is lost since the hints refer to an invalid chain, and we don't
            // know if the current UTXO set is correct. We need to start from genesis.
            error!("Aborting SwiftSync: the most PoW chain is invalid at height {invalid_h}");
            return LoopControl::Break;
        }

        // If we have reached the SwiftSync stop height, we aren't waiting for inflight requested
        // blocks, and there's no in-memory block being processed, we have finished.
        let finished_requesting = self.last_block_request == self.context.stop_height;

        if finished_requesting && self.unprocessed_blocks() == 0 {
            self.handle_stop_height_reached();
            return LoopControl::Break;
        }

        // Checks if we need to open a new connection
        periodic_job!(
            self.last_connection => self.maybe_open_connection(ServiceFlags::NETWORK),
            SwiftSync::TRY_NEW_CONNECTION,
        );

        // Open new feeler connection periodically
        periodic_job!(
            self.last_feeler => self.open_feeler_connection(),
            SwiftSync::FEELER_INTERVAL,
        );

        // Re-request blocks that haven't arrived in `SwiftSync::REQUEST_TIMEOUT` seconds
        try_and_log!(self.check_for_timeout());

        let assume_stale = Instant::now()
            .duration_since(self.common.last_tip_update)
            .as_secs()
            > SwiftSync::ASSUME_STALE;

        if assume_stale {
            try_and_log!(self.create_connection(ConnectionKind::Extra));
            self.last_tip_update = Instant::now();
            return LoopControl::Continue;
        }

        try_and_log!(self.pump_swiftsync(hints));

        self.get_blocks_to_download();
        LoopControl::Continue
    }

    /// Called when we process the last SwiftSync block. Verifies that the produced aggregator is
    /// zero and supply is correct. On success marks the chain assumed and exits IBD.
    ///
    /// If one of the two invariants fails, it sets the `abort_height` field.
    fn handle_stop_height_reached(&mut self) {
        let stop_height = self.context.stop_height;
        let final_agg = self.context.agg;
        let final_supply = self.context.supply;

        if !final_agg.is_zero() {
            error!("SwiftSync failed with the provided hints file; end aggregator is not zero");

            self.context.abort_height = Some(stop_height);
            return;
        }

        let consensus = Consensus::from(self.network);
        if final_supply > consensus.max_supply_at_height(stop_height) {
            error!("Aborting SwiftSync: most PoW chain has excess supply ({final_supply})");

            self.context.abort_height = Some(stop_height);
            return;
        }

        info!("SwiftSync is finished, switching to normal operation mode");
        let tip_hash = self.chain.get_block_hash(stop_height).unwrap();

        self.chain
            .mark_chain_as_assumed(Stump::new(), tip_hash)
            .unwrap();
        self.chain.toggle_ibd(false);
    }

    fn process_spends(&mut self, tx: &Transaction) -> Vec<TxOut> {
        if tx.is_coinbase() {
            return vec![];
        }

        tx.input
            .iter()
            .filter_map(|input| self.context.coins_cache.spend(&input.previous_output))
            .collect()
    }

    fn process_block_cache(&mut self, block: &Block) {
        let inputs: usize = block
            .txdata
            .iter()
            .map(|tx| {
                if tx.is_coinbase() {
                    return 0;
                }

                tx.input.iter().count()
            }).sum();

        let hits = block
            .txdata
            .iter()
            .map(|tx| {
                let txid = tx.compute_txid();
                tx.output.iter().enumerate().for_each(|(vout, out)| {
                    let outpoint = OutPoint {
                        txid,
                        vout: vout as u32,
                    };

                    self.context.coins_cache.create(outpoint, out.clone())
                });

                self.process_spends(tx)
            })
            .flatten()
            .collect::<Vec<_>>();

        /*if hits.len() != inputs {
            println!(
                "block {} stats: hits = {} total = {} cache_size={}",
                block.block_hash(),
                hits.len(),
                inputs,
                self.context.coins_cache.size(),
            );
        }*/
    }

    /// Process a message from a peer and handle it accordingly between the variants of [`PeerMessages`].
    async fn handle_message(
        &mut self,
        msg: NodeNotification,
        hints: &mut Hintsfile,
    ) -> Result<(), WireError> {
        match msg {
            NodeNotification::FromUser(request, responder) => {
                self.perform_user_request(request, responder).await;
            }

            NodeNotification::DnsSeedAddresses(addresses) => {
                self.address_man.push_addresses(&addresses);
            }

            NodeNotification::FromPeer(peer, notification, time) => {
                self.register_message_time(&notification, peer, time);

                let Some(unhandled) = self.handle_peer_msg_common(notification, peer)? else {
                    return Ok(());
                };

                match unhandled {
                    PeerMessages::Block(block) => {
                        let hash = block.block_hash();
                        if self.blocks.contains_key(&hash) {
                            debug!(
                                "Received block {hash} from peer {peer}, but we already have it"
                            );
                            return Ok(());
                        }

                        let Some(_) = self.inflight.remove(&InflightRequests::Blocks(hash)) else {
                            warn!("Received block {hash}, but we didn't ask for it");
                            self.increase_banscore(peer, 5)?;

                            return Ok(());
                        };

                        // Reply and return early if it's a user-requested block. Else continue handling it.
                        let Some(block) = self.check_is_user_block_and_reply(block)? else {
                            return Ok(());
                        };

                        self.process_block_cache(&block);

                        let inflight_block = InflightBlock {
                            peer,
                            block: Arc::new(block),
                            // Since this is AV-SwiftSync, we don't need proofs nor leaves (UTXOs)
                            // TODO: once we implement full validation we'll need the spent UTXOs
                            aux_data: None,
                            processing_since: None,
                        };
                        self.blocks.insert(hash, inflight_block);

                        self.pump_swiftsync(hints)?;
                        self.get_blocks_to_download();
                    }

                    PeerMessages::Ready(version) => {
                        try_and_log!(self.handle_peer_ready(peer, &version));
                    }

                    PeerMessages::Disconnected(idx) => {
                        try_and_log!(self.handle_disconnection(peer, idx));
                    }

                    PeerMessages::UtreexoProof(_) => {
                        warn!("Utreexo proof received from peer {peer}, but we didn't ask (SwiftSync)");
                        self.increase_banscore(peer, 5)?;
                    }

                    _ => {}
                }
            }

            NodeNotification::FromWorker((result, block_hash, height)) => {
                self.handle_worker_notification(result, block_hash, height, hints)?;
            }
        }

        Ok(())
    }

    fn handle_worker_notification(
        &mut self,
        result: WorkerResult,
        block_hash: BlockHash,
        height: u32,
        hints: &mut Hintsfile,
    ) -> Result<(), WireError> {
        // This block has already been processed: open space for a new worker
        let block = self
            .blocks
            .remove(&block_hash)
            .ok_or(WireError::BlockNotFound)?;

        // Immediately replace the finished worker with a new one
        self.pump_swiftsync(hints)?;

        match result {
            Ok((agg_re, unspent_amount)) => {
                self.context.agg += agg_re;
                self.context.supply += unspent_amount;
                self.handle_valid_worker_block(block_hash, height, block);
            }
            Err(e) => {
                let header = block.block.header;
                self.handle_invalid_block(e, header, height, block.peer)?
            }
        };
        Ok(())
    }

    fn handle_invalid_block(
        &mut self,
        chain_err: BlockchainError,
        header: BlockHeader,
        height: u32,
        peer: PeerId,
    ) -> Result<(), WireError> {
        error!("Invalid block {header:?} received by peer {peer} reason: {chain_err:?}");
        let block_hash = header.block_hash();

        // Return early if the error is not from block validation (e.g., a database error)
        let Some(e) = Self::block_validation_err(chain_err) else {
            return Ok(());
        };

        match e {
            // Abort SwiftSync if the block is truly invalid
            BlockValidationErrors::InvalidCoinbase(_)
            | BlockValidationErrors::UtxoNotFound(_)
            | BlockValidationErrors::ScriptValidationError(_)
            | BlockValidationErrors::NullPrevOut
            | BlockValidationErrors::EmptyInputs
            | BlockValidationErrors::EmptyOutputs
            | BlockValidationErrors::ScriptError
            | BlockValidationErrors::BlockTooBig
            | BlockValidationErrors::NotEnoughPow
            | BlockValidationErrors::TooManyCoins
            | BlockValidationErrors::NotEnoughMoney
            | BlockValidationErrors::FirstTxIsNotCoinbase
            | BlockValidationErrors::BadCoinbaseOutValue
            | BlockValidationErrors::EmptyBlock
            | BlockValidationErrors::BadBip34
            | BlockValidationErrors::BIP94TimeWarp
            | BlockValidationErrors::UnspendableUTXO
            | BlockValidationErrors::CoinbaseNotMatured => {
                self.context.abort_height = Some(height);
                try_and_log!(self.chain.invalidate_block(block_hash));
            }

            // This block's txdata doesn't match the txid or wtxid merkle root. This can be a
            // mutated block, so we can't invalidate it since the original txdata may be valid.
            BlockValidationErrors::BadMerkleRoot | BlockValidationErrors::BadWitnessCommitment => {}

            // No proofs involved in SwiftSync
            BlockValidationErrors::InvalidProof => {}

            BlockValidationErrors::BlockExtendsAnOrphanChain
            | BlockValidationErrors::BlockDoesntExtendTip => {
                // The SwiftSync blocks are from our best chain, so this should never happen.
                error!("BUG: block {block_hash} from peer {peer} returned: {e:?}");
                return Ok(());
            }
        }

        warn!("Block {block_hash} from peer {peer} is invalid, banning peer");
        self.disconnect_and_ban(peer)?;

        Err(WireError::PeerMisbehaving)
    }

    /// This method is currently just about updating metrics, but may be changed to persist the
    /// SwiftSync progress.
    fn handle_valid_worker_block(
        &mut self,
        block_hash: BlockHash,
        height: u32,
        block: InflightBlock,
    ) {
        // TODO should we update header and block index (similar to `self.chain.update_view`)?
        info!(
            "SwiftSync block: block_hash={block_hash} height={height} tx_count={}",
            block.block.txdata.len(),
        );

        // TODO should we flush on SwiftSync?
        // TODO notify the block
        self.last_tip_update = Instant::now();

        // Update metrics
        let elapsed = block
            .processing_since
            .expect("Block was processed, this field is `Some`")
            .elapsed()
            .as_secs_f64();

        self.block_sync_avg.add(elapsed);

        #[cfg(feature = "metrics")]
        {
            use metrics::get_metrics;

            let avg = self.block_sync_avg.value().expect("at least one sample");
            let metrics = get_metrics();
            metrics.block_height.set(height.into());
            metrics.avg_block_processing_time.set(avg);
        }
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node that downloads and validates the blockchain, but skips utreexo proofs as they aren't
//! needed to validate the UTXO set with the SwiftSync method.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bitcoin::Amount;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::block::Header as BlockHeader;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use floresta_chain::ThreadSafeChain;
use floresta_chain::pruned_utreexo::IBDState;
use floresta_chain::pruned_utreexo::consensus::Consensus;
use floresta_chain::pruned_utreexo::consensus::SwiftSyncDelta;
use floresta_chain::swift_sync_agg::SipHashKeys;
use floresta_chain::swift_sync_agg::SwiftSyncAgg;
use hintsfile::Hintsfile;
use rand::Rng;
use rustreexo::node_hash::BitcoinNodeHash;
use rustreexo::stump::Stump;
use tokio::time;
use tokio::time::MissedTickBehavior;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::node::ConnectionKind;
use crate::node::InflightBlock;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::UtreexoNode;
use crate::node::WitnessMode;
use crate::node::WorkerResult;
use crate::node::oneshot::error::TryRecvError;
use crate::node::periodic_job;
use crate::node::try_and_log;
use crate::node_context::LoopControl;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::node_handle::UserRequest;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::peer::PeerMessages;
use crate::p2p_wire::stump_updater::SparseUtreexoAdds;
use crate::p2p_wire::stump_updater::StumpUpdater;
use crate::p2p_wire::stump_updater::StumpUpdaterHandle;

/// Worker count used when the available CPU parallelism cannot be determined.
const DEFAULT_WORKERS: usize = 4;

/// [`SwiftSync`] is a node that downloads and validates the blockchain but skips utreexo
/// proofs by using SwiftSync.
///
/// This node implements:
///     - `NodeContext`
///     - `UtreexoNode<SwiftSync, Chain>`
#[derive(Default)]
pub struct SwiftSync {
    stump_updater: Option<StumpUpdaterHandle>,

    /// Maximum number of concurrent block-validation tasks for this session.
    max_workers: usize,

    /// The `TxOut` aggregator.
    agg: SwiftSyncAgg,

    /// The secret salt used to compute the aggregator element hashes.
    salt: Arc<SipHashKeys>,

    /// The total unspent amount. Once we reach the SwiftSync stop height, this must be less or
    /// equal than the theoretical supply limit at that height.
    supply: Amount,

    /// The target height for the currently used SwiftSync hints.
    stop_height: u32,

    /// Number of distinct blocks processed successfully in this SwiftSync session.
    processed_blocks: u32,

    /// Height at which SwiftSync was aborted, if any.
    ///
    /// We abort when either the hints are found to be invalid or the current chain is invalid (we
    /// may find an invalid block or, at the end, a violation of the maximum supply limit).
    abort_height: Option<u32>,
}

impl NodeContext for SwiftSync {
    const TRY_NEW_CONNECTION: u64 = 15; // We want to be well-connected early on
    const REQUEST_TIMEOUT: u64 = 2 * 60; // 2 minutes (5 blocks should reach us much faster)
    const MAX_INFLIGHT_REQUESTS: usize = 100; // double the default
    const MAX_OUTGOING_PEERS: usize = 30;
    const MAX_CONCURRENT_GETDATA: usize = 40; // 40 * 5 = 200 blocks in parallel
    const ASSUME_STALE: u64 = 2 * 60; // Two minutes without blocks while in IBD is very suspicious

    // A more conservative value than the default of 1 second, since we'll have many peer messages
    const MAINTENANCE_TICK: Duration = Duration::from_secs(5);
}

/// Node methods for a [`UtreexoNode`] where its Context is [`SwiftSync`].
/// See [node](crates/floresta-wire/src/p2p_wire/node.rs) for more information.
impl<Chain> UtreexoNode<Chain, SwiftSync>
where
    Chain: ThreadSafeChain,
    WireError: From<Chain::Error>,
{
    /// Parses the SwiftSync hints file and returns an in-memory [`Hintsfile`] representation.
    /// Returns `None` if the file is missing or cannot be read or parsed.
    fn parse_hints_file(datadir: impl AsRef<Path>, network: Network) -> Option<Hintsfile> {
        let path = datadir.as_ref().join(format!("{network}.hints"));

        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    warn!("Could not open SwiftSync hints {}: {e}", path.display());
                }
                return None;
            }
        };

        match Hintsfile::from_reader(&mut file) {
            Ok(hints) => Some(hints),
            Err(e) => {
                warn!("Could not read SwiftSync hints {}: {e}", path.display());
                None
            }
        }
    }

    /// Generates a random salt for this SwiftSync session.
    fn generate_salt() -> Arc<SipHashKeys> {
        let mut rng = rand::rng();

        Arc::new(SipHashKeys::new(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ))
    }

    /// Returns `true` if SwiftSync failed, due to the hints being invalid or the current chain
    /// being invalid (below the SwiftSync stop height).
    pub fn was_aborted(&self) -> bool {
        self.context.abort_height.is_some()
    }

    /// Computes the next blocks to request, and sends a GETDATA request, advancing
    /// `last_block_request` up to the SwiftSync hints `stop_height`.
    fn refill_block_requests(&mut self) {
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

    fn check_connections(&mut self) -> Result<(), WireError> {
        if self.has_fixed_peers() {
            return self.maybe_open_connection(ServiceFlags::NETWORK);
        }

        if self.connected_peers() >= SwiftSync::MAX_OUTGOING_PEERS {
            self.maybe_disconnect_slowest_peer(&[], false)?;
        }

        self.maybe_open_connection(ServiceFlags::NETWORK)
    }

    /// Starts pending blocks without exceeding the session's worker limit.
    fn pump_swiftsync(&mut self, hints: &mut Hintsfile) -> Result<(), WireError> {
        if self.was_aborted() {
            return Ok(());
        }

        let processing = self
            .blocks
            .values()
            .filter(|b| b.processing_since.is_some())
            .count();

        let free = self.context.max_workers.saturating_sub(processing);
        if free == 0 {
            return Ok(());
        }

        // Collect hashes first (can't mutate the map while iterating it)
        let to_process: Vec<BlockHash> = self
            .blocks
            .iter()
            .filter(|(_, b)| b.processing_since.is_none())
            .take(free) // We don't exceed `max_workers`
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

        // Hints are removed only after processing succeeds
        let block_hints = hints
            .indices_at_height(block_height)
            .expect("unprocessed SwiftSync blocks must retain their hints");

        let unspent_indexes: HashSet<u32> = block_hints.into_iter().collect();

        // Start the processing timer
        entry.processing_since = Some(Instant::now());

        let block = Arc::clone(&entry.block);
        let consensus = Consensus::from(self.network);
        let salt = Arc::clone(&self.context.salt);

        let node_sender = self.node_tx.clone();
        tokio::task::spawn_blocking(move || {
            let result =
                consensus.process_block_swiftsync(&block, block_height, &unspent_indexes, &salt);

            let _ = node_sender.send(NodeNotification::FromWorker {
                result,
                block_hash,
                block_height,
            });
        });

        Ok(())
    }

    /// Runs SwiftSync up to the hints' stop height, processing downloaded blocks in parallel
    /// while the stump updater applies their additions in height order. Maintains peer
    /// connections and retries timed-out requests until completion, abort or shutdown.
    ///
    /// Skips SwiftSync unless the configured AssumeValid block is on the best chain
    /// and covers the hints' stop height.
    ///
    /// User block requests are cancelled while SwiftSync runs.
    ///
    /// After processing all blocks, checks that the final aggregator is zero (i.e., the
    /// multiset of hinted-as-spent output `OutPoints` matches the input multiset) and that
    /// supply doesn't exceed the maximum cap at the given height.
    ///
    /// # Errors
    ///
    /// Returns chainstate read or commit errors.
    ///
    /// An invalid chain or failed final check aborts SwiftSync without returning an error.
    /// Use [`Self::was_aborted`] to check whether SwiftSync aborted.
    pub async fn run(mut self, done_cb: impl FnOnce(&Chain)) -> Result<Self, WireError> {
        let Some(mut hints) = Self::parse_hints_file(&self.datadir, self.network) else {
            return Ok(self);
        };

        // We can't safely proceed with an unreadable validation index
        let validation_idx = self.chain.get_validation_index()?;
        if validation_idx >= hints.stop_height() {
            return Ok(self);
        }

        if validation_idx != 0 {
            info!("Skipping SwiftSync: proof sync has already reached height {validation_idx}.");
            return Ok(self);
        }

        // SwiftSync skips spent-coin validation, so the entire range must be covered by AssumeValid
        let Some(assume_valid_height) = self.chain.get_assume_valid_height()? else {
            info!("Skipping SwiftSync: no AssumeValid block on the best chain.");
            return Ok(self);
        };
        if hints.stop_height() > assume_valid_height {
            info!(
                "Skipping SwiftSync: hints stop at {}, beyond AssumeValid height {assume_valid_height}.",
                hints.stop_height(),
            );
            return Ok(self);
        }

        // Pending user block replies must not consume blocks needed by SwiftSync.
        self.inflight_user_requests
            .retain(|request, _| !matches!(request, UserRequest::Block(_)));

        // Allow as many worker threads as this machine supports. This is helpful for very high
        // bandwidth connections (e.g., > 1 Gbps). Defaults to 4 workers for unknown CPUs.
        self.context.max_workers =
            std::thread::available_parallelism().map_or(DEFAULT_WORKERS, |n| n.get());

        self.witness_mode = WitnessMode::Witnessless; // enable witnessless sync
        self.context.stop_height = hints.stop_height();

        self.last_block_request = 0;
        self.context.processed_blocks = 0;
        self.chain.update_ibd(IBDState::SwiftSync {
            processed_blocks: 0,
            total_blocks: hints.stop_height(),
        });

        // Initialize the accumulator updater task that will work in parallel to block validation
        self.context.stump_updater =
            Some(StumpUpdater::spawn(Stump::new(), 0, hints.stop_height()));

        // Generate the random salt and kick off SwiftSync!
        self.context.salt = Self::generate_salt();

        info!(
            "Performing SwiftSync up to height {} with up to {} validation workers",
            hints.stop_height(),
            self.context.max_workers,
        );

        let mut ticker = time::interval(SwiftSync::MAINTENANCE_TICK);
        // If we fall behind, don't "catch up" by running maintenance repeatedly
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;

                // Maintenance runs only on tick but has priority
                _ = ticker.tick() => match self.maintenance_tick(&mut hints).await? {
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

        // Release pending accumulator updates when SwiftSync stops or aborts.
        self.context.stump_updater = None;

        // Ordinary proof sync needs witnesses, including after aborting or stopping SwiftSync.
        self.witness_mode = WitnessMode::Full;
        done_cb(&self.chain);
        Ok(self)
    }

    /// Performs the periodic maintenance tasks, including checking for the cancel signal, peer
    /// connections, and inflight request timeouts.
    ///
    /// Returns `LoopControl::Break` if we need to break the main `SwiftSync` loop, which may
    /// happen if the kill signal was set, we successfully finished SwiftSync, or we need to abort
    /// operation due to a validation error.
    async fn maintenance_tick(&mut self, hints: &mut Hintsfile) -> Result<LoopControl, WireError> {
        if *self.kill_signal.read().await {
            return Ok(LoopControl::Break);
        }

        if let Some(invalid_h) = self.context.abort_height {
            // All our progress is lost since the hints refer to an invalid chain, and we don't
            // know if the current UTXO set is correct. We need to start from genesis.
            error!("Aborting SwiftSync: the most PoW chain is invalid at height {invalid_h}");
            return Ok(LoopControl::Break);
        }

        // If we have reached the SwiftSync stop height, and we have added all the utreexo leaves
        // to the accumulator, we have finished.
        if let Some(final_acc) = self.swift_sync_finished() {
            self.handle_stop_height_reached(final_acc)?;
            return Ok(LoopControl::Break);
        }

        // Checks if we need to open a new connection
        periodic_job!(
            self.last_connection => self.check_connections(),
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
            return Ok(LoopControl::Continue);
        }

        try_and_log!(self.pump_swiftsync(hints));

        self.refill_block_requests();
        Ok(LoopControl::Continue)
    }

    /// Returns true if we have requested all blocks up to the stop height, we have received and
    /// processed all of them, and we have added all utreexo leaves to the accumulator.
    fn swift_sync_finished(&mut self) -> Option<Stump> {
        let requesting_blocks = self.last_block_request != self.context.stop_height;

        // We are still requesting or processing blocks
        if requesting_blocks || self.unprocessed_blocks() != 0 {
            return None;
        }

        // Try to get the result from the stump builder task, else keep waiting for it to finish
        let updater = self.context.stump_updater.as_mut().expect("initialized");

        match updater.done.try_recv() {
            Ok(stump) => Some(stump),

            // All blocks have been processed, but the stump builder task hasn't finished yet
            Err(TryRecvError::Empty) => None,

            // These should never happen!
            Err(TryRecvError::Closed) => panic!("Stump builder task was closed without result"),
        }
    }

    /// Called when we process the last SwiftSync block. Verifies that the produced aggregator is
    /// zero and supply is correct. On success marks the chain assumed and exits IBD.
    ///
    /// If one of the two invariants fails, it sets the `abort_height` field.
    fn handle_stop_height_reached(&mut self, final_acc: Stump) -> Result<(), WireError> {
        let stop_height = self.context.stop_height;
        let final_agg = self.context.agg;
        let final_supply = self.context.supply;

        if !final_agg.is_zero() {
            error!("SwiftSync failed with the provided hints file; end aggregator is not zero");

            self.context.abort_height = Some(stop_height);
            return Ok(());
        }

        let consensus = Consensus::from(self.network);
        if final_supply > consensus.max_supply_at_height(stop_height) {
            error!("Aborting SwiftSync: most PoW chain has excess supply ({final_supply})");

            self.context.abort_height = Some(stop_height);
            return Ok(());
        }

        let tip_hash = self.chain.get_block_hash(stop_height)?;

        info!("SwiftSync produced the following accumulator for {tip_hash}: \n{final_acc:?}");
        self.chain.mark_chain_as_assumed(final_acc, tip_hash)?;

        info!("SwiftSync is finished, switching to Utreexo-proof mode");
        Ok(())
    }

    /// Process a message from a peer and handle it accordingly between the variants of [`PeerMessages`].
    async fn handle_message(
        &mut self,
        msg: NodeNotification,
        hints: &mut Hintsfile,
    ) -> Result<(), WireError> {
        match msg {
            NodeNotification::FromUser(UserRequest::Block(_), _) => {
                // User replies must not consume matching sync blocks and stall sync, nor receive
                // witnessless blocks when full blocks were requested. Until replies are routed
                // independently, dropping the responder rejects these requests.
            }

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
                            self.punish_unrequested_block(peer, hash)?;

                            return Ok(());
                        };

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
                        self.refill_block_requests();
                    }

                    PeerMessages::Ready(version) => {
                        try_and_log!(self.handle_peer_ready(peer, version));
                    }

                    PeerMessages::Disconnected(idx) => {
                        try_and_log!(self.handle_disconnection(peer, idx));
                    }

                    PeerMessages::UtreexoProof(_) => {
                        warn!(
                            "Utreexo proof received from peer {peer}, but we didn't ask (SwiftSync)"
                        );
                        self.increase_banscore(peer, 5)?;
                    }

                    _ => {}
                }
            }

            NodeNotification::FromWorker {
                result,
                block_hash,
                block_height,
            } => {
                self.handle_worker_notification(result, block_hash, block_height, hints)?;
                self.pump_swiftsync(hints)?;
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
        // Ignore remaining worker results after SwiftSync has aborted.
        if self.was_aborted() {
            return Ok(());
        }

        // This block has already been processed: open space for a new worker
        let block = self
            .blocks
            .remove(&block_hash)
            .ok_or(WireError::BlockNotFound)?;

        match result {
            Ok(SwiftSyncDelta {
                aggregator_delta,
                unspent_amount,
                utreexo_adds,
            }) => {
                self.context.agg += aggregator_delta;
                // Saturating add to avoid panics on overflow (protection against bad hints)
                self.context.supply = self
                    .context
                    .supply
                    .checked_add(unspent_amount)
                    .unwrap_or(Amount::MAX);
                self.pump_utreexo_adds(height, utreexo_adds);

                // Block is valid and not mutated, we can drop these hints from memory
                hints.take_indices(height);
                self.handle_valid_worker_block(block_hash, height, block);

                assert!(self.context.processed_blocks < self.context.stop_height);
                self.context.processed_blocks += 1;

                // Expose the current SwiftSync progress through the FFI API
                self.chain.update_ibd(IBDState::SwiftSync {
                    processed_blocks: self.context.processed_blocks,
                    total_blocks: self.context.stop_height,
                });

                // Refill request capacity freed by processing this block
                self.refill_block_requests();
            }
            Err(e) => {
                let header = block.block.header;
                self.handle_invalid_block(e, header, height, block.peer)?;
            }
        };
        Ok(())
    }

    /// Handles sending new utreexo leaves to add. This should only be called when we know the
    /// stump builder task is running (i.e., when there are still blocks to process).
    fn pump_utreexo_adds(&self, height: u32, adds: Vec<BitcoinNodeHash>) {
        let updater = self.context.stump_updater.as_ref().expect("initialized");

        updater
            .tx
            .send((height, SparseUtreexoAdds::new(adds)))
            .expect("addition-only doesn't fail (proofless), updater should be alive");
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
            | BlockValidationErrors::InvalidSignetBlockSolution
            | BlockValidationErrors::UtxoNotFound(_)
            | BlockValidationErrors::ScriptValidationError(_)
            | BlockValidationErrors::NullPrevOut
            | BlockValidationErrors::EmptyInputs
            | BlockValidationErrors::EmptyOutputs
            | BlockValidationErrors::ScriptError
            | BlockValidationErrors::BlockTooBig
            | BlockValidationErrors::TooManySigops
            | BlockValidationErrors::NotEnoughPow
            | BlockValidationErrors::BadDiffBits
            | BlockValidationErrors::BadVersion
            | BlockValidationErrors::TooManyCoins
            | BlockValidationErrors::NotEnoughMoney
            | BlockValidationErrors::FirstTxIsNotCoinbase
            | BlockValidationErrors::BadCoinbaseOutValue
            | BlockValidationErrors::EmptyBlock
            | BlockValidationErrors::BadBip34
            | BlockValidationErrors::BIP94TimeWarp
            | BlockValidationErrors::UnspendableUTXO
            | BlockValidationErrors::NonFinalTransaction
            | BlockValidationErrors::CoinbaseNotMatured
            | BlockValidationErrors::BadRelativeLockTime
            | BlockValidationErrors::DuplicateInput => {
                self.context.abort_height = Some(height);
                try_and_log!(self.chain.invalidate_block(block_hash));
            }

            // This block's txdata doesn't match the txid or wtxid merkle root. This can be a
            // mutated block, so we can't invalidate it since the original txdata may be valid.
            BlockValidationErrors::BadMerkleRoot | BlockValidationErrors::BadWitnessCommitment => {
                // Re-insert the block request so that we can retry it after banning this peer
                self.inflight
                    .insert(InflightRequests::Blocks(block_hash), (peer, Instant::now()));
            }

            // No proofs involved in SwiftSync (we use implicit deletion instead)
            BlockValidationErrors::InvalidUtreexoProof => {}

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
        info!(
            "SwiftSync block: block_hash={block_hash} height={height} tx_count={}",
            block.block.txdata.len(),
        );

        // TODO(@davidson) implement a way to index/notify this block
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
            use floresta_metrics::get_metrics;

            let avg = self.block_sync_avg.value().expect("at least one sample");
            let metrics = get_metrics();
            metrics.block_height.set(height.into());
            metrics.avg_block_processing_time.set(avg);
        }
    }
}

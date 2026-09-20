// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node that downloads and validates the blockchain, but skips utreexo proofs as they aren't
//! needed to validate the UTXO set with the SwiftSync method.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bitcoin::Amount;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::TxIn;
use bitcoin::TxOut;
use bitcoin::block::Header as BlockHeader;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::serialize;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use floresta_chain::CompactLeafData;
use floresta_chain::DatabaseError;
use floresta_chain::ScriptPubKeyKind;
use floresta_chain::ThreadSafeChain;
use floresta_chain::extensions::Bip30UnspendableExt;
use floresta_chain::extensions::HeaderExt;
use floresta_chain::proof_util;
use floresta_chain::pruned_utreexo::IBDState;
use floresta_chain::pruned_utreexo::consensus::Consensus;
use floresta_chain::pruned_utreexo::utxo_data::UtxoData;
use floresta_chain::swift_sync_agg::SipHashKeys;
use floresta_chain::swift_sync_agg::SwiftSyncAgg;
use floresta_common::service_flags;
use floresta_db::Config as DatabaseConfig;
use floresta_db::Database;
use floresta_db::Mode;
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
use crate::node::IndexWorkerResult;
use crate::node::InflightBlock;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::PeerStatus;
use crate::node::UtreexoNode;
use crate::node::ValidationTimings;
use crate::node::WitnessMode;
use crate::node::download_window::DownloadWindow;
use crate::node::oneshot::error::TryRecvError;
use crate::node::periodic_job;
use crate::node::try_and_log;
use crate::node_context::LoopControl;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::p2p_wire::error::WireError;
use crate::p2p_wire::peer::PeerMessages;
use crate::p2p_wire::stump_updater::SparseUtreexoAdds;
use crate::p2p_wire::stump_updater::StumpUpdater;
use crate::p2p_wire::stump_updater::StumpUpdaterHandle;

#[derive(Debug)]
struct SwiftSyncDatabaseError(String);

impl fmt::Display for SwiftSyncDatabaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl DatabaseError for SwiftSyncDatabaseError {}

struct SwiftSyncUtxoDb {
    database: Option<Database>,
    path: PathBuf,
}

const UTXO_KEY_SIZE: usize = 16;
const SCRIPT_COMMITMENT_SIZE: usize = 12;
const UTXO_METADATA_SIZE: usize = 16;
const RECLAIM_WATERMARK: usize = 16 * 1024;
const PACKED_AMOUNT_BITS: u32 = 37;
const PACKED_AMOUNT_MASK: u64 = (1 << PACKED_AMOUNT_BITS) - 1;
const PACKED_SPK_SHIFT: u32 = PACKED_AMOUNT_BITS;
const PACKED_COINBASE_SHIFT: u32 = PACKED_SPK_SHIFT + 3;
const PACKED_HEIGHT_SHIFT: u32 = PACKED_COINBASE_SHIFT + 1;
const PACKED_HEIGHT_MAX: u32 = (1 << 23) - 1;

struct LoadedInputs {
    utxos: HashMap<OutPoint, UtxoData>,
}

impl SwiftSyncUtxoDb {
    fn create(datadir: &Path) -> Result<Arc<Self>, BlockchainError> {
        let path = datadir.join(format!(".swiftsync-utxos-{}", rand::rng().next_u64()));
        let mut config = DatabaseConfig::new(Mode::Map, 1 << 25);
        // Sparse mappings: these are address-space maxima, not eager disk allocations.
        config.body_capacity = 256 << 30;
        config.blob_capacity = 256 << 30;
        let database = Database::create(&path, config)
            .map_err(|err| SwiftSyncDatabaseError(err.to_string()))?;
        Ok(Arc::new(Self {
            database: Some(database),
            path,
        }))
    }

    /// Packs the 96 least-significant txid bits and 32-bit vout into one 128-bit key.
    fn outpoint_key(outpoint: OutPoint) -> [u8; UTXO_KEY_SIZE] {
        let txid = outpoint.txid.to_byte_array();
        let mut key = [0_u8; UTXO_KEY_SIZE];
        // Txid's internal byte order is the reverse of its display order, so this prefix is the
        // numeric least-significant end.
        key[..12].copy_from_slice(&txid[..12]);
        key[12..].copy_from_slice(&outpoint.vout.to_le_bytes());
        key
    }

    fn encode_utxo(utxo: &UtxoData) -> Vec<u8> {
        let script_kind = proof_util::get_script_type(&utxo.txout.script_pubkey);
        let amount = utxo.txout.value.to_sat();
        if let Some(encoded_kind) = Self::packed_script_kind(&script_kind)
            && utxo.creation_height <= PACKED_HEIGHT_MAX
            && amount <= PACKED_AMOUNT_MASK
        {
            let packed = amount
                | encoded_kind << PACKED_SPK_SHIFT
                | u64::from(utxo.is_coinbase) << PACKED_COINBASE_SHIFT
                | u64::from(utxo.creation_height) << PACKED_HEIGHT_SHIFT;
            return packed.to_le_bytes().to_vec();
        }

        let header_code = (utxo.creation_height << 1) | u32::from(utxo.is_coinbase);
        let encoded_kind = serialize(&script_kind);
        let reconstructable = !matches!(&script_kind, ScriptPubKeyKind::Other(_));
        let mut value = Vec::with_capacity(
            UTXO_METADATA_SIZE
                + encoded_kind.len()
                + usize::from(reconstructable) * SCRIPT_COMMITMENT_SIZE,
        );
        value.extend_from_slice(&header_code.to_le_bytes());
        value.extend_from_slice(&utxo.creation_time.to_le_bytes());
        value.extend_from_slice(&amount.to_le_bytes());
        value.extend_from_slice(&encoded_kind);
        if reconstructable {
            let commitment = sha256::Hash::hash(utxo.txout.script_pubkey.as_bytes());
            value.extend_from_slice(&commitment.to_byte_array()[..SCRIPT_COMMITMENT_SIZE]);
        }
        value
    }

    fn packed_script_kind(script_kind: &ScriptPubKeyKind) -> Option<u64> {
        match script_kind {
            ScriptPubKeyKind::PubKeyHash => Some(1),
            ScriptPubKeyKind::WitnessV0PubKeyHash => Some(2),
            ScriptPubKeyKind::ScriptHash => Some(3),
            ScriptPubKeyKind::WitnessV0ScriptHash => Some(4),
            ScriptPubKeyKind::Other(_) => None,
        }
    }

    fn unpacked_script_kind(encoded: u64) -> Result<ScriptPubKeyKind, BlockchainError> {
        match encoded {
            1 => Ok(ScriptPubKeyKind::PubKeyHash),
            2 => Ok(ScriptPubKeyKind::WitnessV0PubKeyHash),
            3 => Ok(ScriptPubKeyKind::ScriptHash),
            4 => Ok(ScriptPubKeyKind::WitnessV0ScriptHash),
            _ => Err(SwiftSyncDatabaseError("invalid packed script type".to_owned()).into()),
        }
    }

    fn decode_utxo(value: &[u8], input: &TxIn) -> Result<UtxoData, BlockchainError> {
        let (header_code, creation_time, amount, script_kind, commitment) = if let Ok(packed) =
            <[u8; 8]>::try_from(value)
        {
            let packed = u64::from_le_bytes(packed);
            let amount = packed & PACKED_AMOUNT_MASK;
            let encoded_kind = (packed >> PACKED_SPK_SHIFT) & 0b111;
            let is_coinbase = (packed >> PACKED_COINBASE_SHIFT) & 1;
            let height = packed >> PACKED_HEIGHT_SHIFT;
            (
                (u32::try_from(height)
                    .map_err(|_| SwiftSyncDatabaseError("packed height overflow".to_owned()))?
                    << 1)
                    | u32::try_from(is_coinbase).map_err(|_| {
                        SwiftSyncDatabaseError("packed coinbase flag overflow".to_owned())
                    })?,
                0,
                amount,
                Self::unpacked_script_kind(encoded_kind)?,
                None,
            )
        } else {
            if value.len() <= UTXO_METADATA_SIZE {
                return Err(SwiftSyncDatabaseError("truncated SwiftSync UTXO".to_owned()).into());
            }
            let header_code =
                u32::from_le_bytes(value[..4].try_into().expect("four-byte header code"));
            let creation_time =
                u32::from_le_bytes(value[4..8].try_into().expect("four-byte creation time"));
            let amount = u64::from_le_bytes(value[8..16].try_into().expect("eight-byte amount"));
            let encoded_script = &value[UTXO_METADATA_SIZE..];
            let script_kind: ScriptPubKeyKind = if encoded_script[0] == 0 {
                deserialize(encoded_script)
                    .map_err(|err| SwiftSyncDatabaseError(err.to_string()))?
            } else {
                if encoded_script.len() != 1 + SCRIPT_COMMITMENT_SIZE {
                    return Err(SwiftSyncDatabaseError(
                        "invalid reconstructable script encoding".to_owned(),
                    )
                    .into());
                }
                deserialize(&encoded_script[..1])
                    .map_err(|err| SwiftSyncDatabaseError(err.to_string()))?
            };
            let commitment = if encoded_script[0] == 0 {
                None
            } else {
                Some(&encoded_script[1..1 + SCRIPT_COMMITMENT_SIZE])
            };
            (header_code, creation_time, amount, script_kind, commitment)
        };
        let leaf = CompactLeafData {
            header_code,
            amount,
            spk_ty: script_kind,
        };
        let script_pubkey = proof_util::reconstruct_script_pubkey(&leaf, input).map_err(|err| {
            BlockValidationErrors::ScriptValidationError(format!(
                "could not reconstruct prevout script: {err:?}"
            ))
        })?;
        if let Some(expected_commitment) = commitment {
            let commitment = sha256::Hash::hash(script_pubkey.as_bytes()).to_byte_array();
            if commitment[..SCRIPT_COMMITMENT_SIZE] != *expected_commitment {
                return Err(BlockValidationErrors::ScriptValidationError(
                    "reconstructed prevout script does not match its indexed commitment".to_owned(),
                )
                .into());
            }
        }
        Ok(UtxoData {
            txout: TxOut {
                value: Amount::from_sat(amount),
                script_pubkey,
            },
            is_coinbase: header_code & 1 != 0,
            creation_height: header_code >> 1,
            creation_time,
        })
    }

    fn index_block(
        &self,
        block: &bitcoin::Block,
        height: u32,
        unspent_indexes: &HashSet<u32>,
        creation_time: u32,
    ) -> Result<(), BlockchainError> {
        let mut output_index = 0_u32;
        let mut entries = Vec::new();
        let spent_in_block: HashSet<OutPoint> = block
            .txdata
            .iter()
            .skip(1)
            .flat_map(|transaction| transaction.input.iter())
            .map(|input| input.previous_output)
            .collect();

        for (transaction_index, transaction) in block.txdata.iter().enumerate() {
            if transaction_index == 0 && block.is_bip30_unspendable(height) {
                continue;
            }
            let txid = transaction.compute_txid();
            for (vout, txout) in transaction.output.iter().enumerate() {
                if Consensus::is_unspendable(&txout.script_pubkey) {
                    continue;
                }
                let is_unspent = unspent_indexes.contains(&output_index);
                output_index += 1;
                if is_unspent {
                    continue;
                }
                let outpoint = OutPoint::new(txid, vout as u32);
                if spent_in_block.contains(&outpoint) {
                    continue;
                }
                let key = Self::outpoint_key(outpoint);
                let value = Self::encode_utxo(&UtxoData {
                    txout: txout.clone(),
                    is_coinbase: transaction_index == 0,
                    creation_height: height,
                    creation_time,
                });
                entries.push((key, value));
            }
        }

        let writer = self
            .database
            .as_ref()
            .expect("database exists until drop")
            .write_only()
            .map_err(|err| SwiftSyncDatabaseError(err.to_string()))?;
        writer
            .put_batch(
                entries
                    .iter()
                    .map(|(key, value)| (key.as_slice(), value.as_slice())),
            )
            .map_err(|err| SwiftSyncDatabaseError(err.to_string()))?;
        Ok(())
    }

    fn load_inputs(
        &self,
        block: &bitcoin::Block,
        height: u32,
        creation_time: u32,
    ) -> Result<LoadedInputs, BlockchainError> {
        let txids: Vec<_> = block
            .txdata
            .iter()
            .map(|transaction| transaction.compute_txid())
            .collect();
        let block_outputs: HashSet<OutPoint> = block
            .txdata
            .iter()
            .zip(&txids)
            .flat_map(|(transaction, txid)| {
                transaction
                    .output
                    .iter()
                    .enumerate()
                    .map(move |(vout, _)| OutPoint::new(*txid, vout as u32))
            })
            .collect();
        let spent_in_block: HashSet<OutPoint> = block
            .txdata
            .iter()
            .skip(1)
            .flat_map(|transaction| transaction.input.iter())
            .map(|input| input.previous_output)
            .filter(|outpoint| block_outputs.contains(outpoint))
            .collect();

        let mut local_outputs = HashMap::with_capacity(spent_in_block.len());
        let mut inputs = HashMap::new();
        let mut external_inputs = Vec::new();
        let mut seen_inputs = HashSet::new();
        for (transaction_index, (transaction, txid)) in block.txdata.iter().zip(txids).enumerate() {
            if transaction_index != 0 {
                for input in &transaction.input {
                    let outpoint = input.previous_output;
                    if !seen_inputs.insert(outpoint) {
                        return Err(BlockValidationErrors::UtxoNotFound(outpoint).into());
                    }
                    if let Some(utxo) = local_outputs.remove(&outpoint) {
                        inputs.insert(outpoint, utxo);
                    } else if block_outputs.contains(&outpoint) {
                        // The referenced output belongs to this block but has not been created yet,
                        // or was already consumed by an earlier transaction.
                        return Err(BlockValidationErrors::UtxoNotFound(outpoint).into());
                    } else {
                        external_inputs.push((outpoint, input));
                    }
                }
            }

            for (vout, txout) in transaction.output.iter().enumerate() {
                let outpoint = OutPoint::new(txid, vout as u32);
                if spent_in_block.contains(&outpoint) {
                    local_outputs.insert(
                        outpoint,
                        UtxoData {
                            txout: txout.clone(),
                            is_coinbase: transaction_index == 0,
                            creation_height: height,
                            creation_time,
                        },
                    );
                }
            }
        }

        let keys: Vec<(OutPoint, [u8; UTXO_KEY_SIZE])> = external_inputs
            .iter()
            .map(|(outpoint, _)| (*outpoint, Self::outpoint_key(*outpoint)))
            .collect();
        let values = self.pop_inputs(&keys)?;
        inputs.reserve(external_inputs.len());
        for ((outpoint, input), value) in external_inputs.into_iter().zip(values) {
            inputs.insert(outpoint, Self::decode_utxo(&value, input)?);
        }
        Ok(LoadedInputs { utxos: inputs })
    }

    /// Destructively loads inputs in bounded batches so body and blob reclamation never retains
    /// more than [`RECLAIM_WATERMARK`] detached entries at once.
    fn pop_inputs(
        &self,
        keys: &[(OutPoint, [u8; UTXO_KEY_SIZE])],
    ) -> Result<Vec<Vec<u8>>, BlockchainError> {
        let database = self.database.as_ref().expect("database exists until drop");
        let mut values = Vec::with_capacity(keys.len());
        for chunk in keys.chunks(RECLAIM_WATERMARK) {
            let popped = database
                .batch_pop(chunk.iter().map(|(_, key)| key.as_slice()))
                .map_err(|err| SwiftSyncDatabaseError(err.to_string()))?;
            for ((outpoint, _), value) in chunk.iter().zip(popped) {
                values.push(value.ok_or(BlockValidationErrors::UtxoNotFound(*outpoint))?);
            }
        }
        Ok(values)
    }
}

impl Drop for SwiftSyncUtxoDb {
    fn drop(&mut self) {
        drop(self.database.take());
        if let Err(err) = fs::remove_dir_all(&self.path) {
            warn!("Failed to remove temporary SwiftSync UTXO index: {err}");
        }
    }
}

/// [`SwiftSync`] is a node that downloads and validates the blockchain but skips utreexo
/// proofs by using SwiftSync.
///
/// This node implements:
///     - `NodeContext`
///     - `UtreexoNode<SwiftSync, Chain>`
#[derive(Default)]
pub struct SwiftSync {
    /// Temporary historical prevout index shared by indexing and validation workers.
    utxo_db: Option<Arc<SwiftSyncUtxoDb>>,

    /// Indexed blocks retained until contextual validation completes.
    indexed_blocks: BTreeMap<u32, InflightBlock>,

    /// Serialized bytes held by all downloaded blocks until validation completes.
    downloaded_block_bytes: usize,

    /// Indexed heights whose predecessors have all been indexed.
    validation_queue: VecDeque<u32>,

    /// Heights currently undergoing contextual validation.
    validating: HashSet<u32>,

    /// Highest contiguous indexed height.
    indexed_prefix: u32,

    stump_updater: Option<StumpUpdaterHandle>,

    /// Rolling byte limit for requested and not-yet-processed blocks.
    download_window: DownloadWindow,

    /// Interval counters for logging only; never used to select peers or adjust the window.
    diagnostics: DownloadDiagnostics,

    /// Thirty-second slot peaks and queue-pressure timing, used only for logging.
    worker_occupancy: WorkerOccupancy,

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

const BLOCK_TIMEOUT_MULTIPLIER: u32 = 10;
const BLOCK_TIMEOUT_SAFETY_MARGIN: Duration = Duration::from_secs(2);
const MIN_BLOCK_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);
const MAX_BLOCK_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const FALLBACK_PEER_RESPONSE_TIME: Duration = Duration::from_secs(2);

impl NodeContext for SwiftSync {
    fn get_required_services(&self) -> bitcoin::p2p::ServiceFlags {
        ServiceFlags::WITNESS | service_flags::UTREEXO.into() | ServiceFlags::NETWORK
    }

    fn block_download_window(&self) -> usize {
        MAX_BUFFERED_BLOCKS
    }

    fn block_download_window_bytes(&self) -> Option<usize> {
        Some(self.download_window.limit_bytes())
    }

    fn requested_block_bytes(&self) -> usize {
        self.download_window.estimated_block_bytes()
    }

    fn retained_processing_blocks(&self) -> usize {
        self.indexed_blocks.len()
    }

    fn downloaded_processing_bytes(&self) -> Option<usize> {
        Some(self.downloaded_block_bytes)
    }

    fn block_request_timeout(&self, average_peer_time: Option<Duration>) -> Duration {
        average_peer_time
            .unwrap_or(FALLBACK_PEER_RESPONSE_TIME)
            .saturating_mul(BLOCK_TIMEOUT_MULTIPLIER)
            .saturating_add(BLOCK_TIMEOUT_SAFETY_MARGIN)
            .clamp(
                MIN_BLOCK_REQUEST_TIMEOUT,
                MAX_BLOCK_REQUEST_TIMEOUT.min(Duration::from_secs(Self::REQUEST_TIMEOUT)),
            )
    }

    const TRY_NEW_CONNECTION: u64 = 15; // We want to be well-connected early on
    const NEW_CONNECTIONS_BATCH_SIZE: usize = 12;
    const REQUEST_TIMEOUT: u64 = 2 * 60; // Upper bound and timeout for non-block requests
    const MAX_INFLIGHT_REQUESTS: usize = 100; // double the default
    const MAX_OUTGOING_PEERS: usize = 30;
    // Probing download capacity can produce late responses without peer misbehavior.
    const PENALIZE_BLOCK_TIMEOUT: bool = false;
    const ASSUME_STALE: u64 = 2 * 60; // Two minutes without blocks while in IBD is very suspicious

    // A more conservative value than the default of 1 second, since we'll have many peer messages
    const MAINTENANCE_TICK: Duration = Duration::from_secs(5);
}

// This is more than enough to avoid CPU from ever becoming a bottleneck
const MAX_PARALLEL_WORKERS: usize = 16;
const MAX_BUFFERED_BLOCKS: usize = 2_000;

/// Slot peaks and time with all slots occupied while a downloaded block awaits dispatch.
/// Slots include dispatch/result-queue delay and inline processing, not just CPU execution.
/// Queued blocks have been accepted by the node; messages still in `node_rx` do not count.
struct WorkerOccupancy {
    since: Instant,
    peak: usize,
    last_update: Instant,
    all_busy_with_queue: bool,
    all_busy_with_queue_time: Duration,
}

impl Default for WorkerOccupancy {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            since: now,
            peak: 0,
            last_update: now,
            all_busy_with_queue: false,
            all_busy_with_queue_time: Duration::ZERO,
        }
    }
}

impl WorkerOccupancy {
    /// Accounts for the previous state before recording a queue or slot change.
    fn record(&mut self, now: Instant, occupied: usize, queued: usize) {
        if self.all_busy_with_queue {
            self.all_busy_with_queue_time += now.duration_since(self.last_update);
        }
        self.last_update = now;
        self.all_busy_with_queue = occupied == MAX_PARALLEL_WORKERS && queued > 0;
        self.peak = self.peak.max(occupied);
    }

    /// Reports every thirty seconds, or the final partial interval when `finished` is true.
    fn take(
        &mut self,
        now: Instant,
        occupied: usize,
        queued: usize,
        finished: bool,
    ) -> Option<(Duration, usize, Duration)> {
        let interval = now.duration_since(self.since);
        if interval.is_zero() || (!finished && interval < Duration::from_secs(30)) {
            return None;
        }

        self.record(now, occupied, queued);
        let sample = (interval, self.peak, self.all_busy_with_queue_time);
        // Carry the live state across the boundary, but start fresh interval totals.
        self.peak = occupied;
        self.all_busy_with_queue_time = Duration::ZERO;
        self.since = now;
        Some(sample)
    }
}

/// Traffic handled since the previous diagnostic tick, including unneeded block copies.
#[derive(Default)]
struct PeerDownloadDiagnostics {
    received_blocks: u64,
    received_bytes: u64,
    ignored_blocks: u64,
    ignored_bytes: u64,
    /// Needed replies from a different peer than the currently assigned retry recipient.
    other_peer_replies: u64,
}

/// Independent wall-clock counters; unlike probe measurements, these include draining/settling.
struct DownloadDiagnostics {
    since: Instant,
    peers: BTreeMap<PeerId, PeerDownloadDiagnostics>,
    processed_blocks: u64,
    processed_bytes: u64,
    /// Delay from completed message decoding to handling by this node, not network latency.
    max_message_delay: Duration,
    worker_results: u64,
    worker_turnaround: Duration,
    max_worker_turnaround: Duration,
    validation_results: u64,
    prevout_fetch_time: Duration,
    prevout_delete_time: Duration,
    consensus_validation_time: Duration,
}

impl Default for DownloadDiagnostics {
    fn default() -> Self {
        Self {
            since: Instant::now(),
            peers: BTreeMap::new(),
            processed_blocks: 0,
            processed_bytes: 0,
            max_message_delay: Duration::ZERO,
            worker_results: 0,
            worker_turnaround: Duration::ZERO,
            max_worker_turnaround: Duration::ZERO,
            validation_results: 0,
            prevout_fetch_time: Duration::ZERO,
            prevout_delete_time: Duration::ZERO,
            consensus_validation_time: Duration::ZERO,
        }
    }
}

impl DownloadDiagnostics {
    /// Counts decoded block payloads when handled, not raw socket traffic.
    /// Node backlogs can shift this rate between intervals; message delay exposes that lag.
    fn received_block(
        &mut self,
        peer: PeerId,
        bytes: usize,
        ignored: bool,
        other_peer: bool,
        message_delay: Duration,
    ) {
        let traffic = self.peers.entry(peer).or_default();
        traffic.received_blocks += 1;
        traffic.received_bytes += bytes as u64;
        if ignored {
            traffic.ignored_blocks += 1;
            traffic.ignored_bytes += bytes as u64;
        } else if other_peer {
            traffic.other_peer_replies += 1;
        }
        self.max_message_delay = self.max_message_delay.max(message_delay);
    }
}

#[derive(Default)]
struct ProcessMemory {
    rss_kib: u64,
    anonymous_kib: u64,
    file_kib: u64,
    swap_kib: u64,
}

#[cfg(target_os = "linux")]
fn process_memory() -> ProcessMemory {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return ProcessMemory::default();
    };
    let value = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    };
    ProcessMemory {
        rss_kib: value("VmRSS:"),
        anonymous_kib: value("RssAnon:"),
        file_kib: value("RssFile:"),
        swap_kib: value("VmSwap:"),
    }
}

#[cfg(not(target_os = "linux"))]
fn process_memory() -> ProcessMemory {
    ProcessMemory::default()
}

/// Node methods for a [`UtreexoNode`] where its Context is [`SwiftSync`].
/// See [node](crates/floresta-wire/src/p2p_wire/node.rs) for more information.
impl<Chain> UtreexoNode<Chain, SwiftSync>
where
    Chain: ThreadSafeChain,
    WireError: From<Chain::Error>,
{
    /// Parses the SwiftSync hints file and returns an in-memory [`Hintsfile`] representation.
    fn parse_hints_file(datadir: impl AsRef<Path>, network: Network) -> Option<Hintsfile> {
        let path = datadir.as_ref().join(format!("{network}.hints"));

        let mut file = File::open(path).ok()?;
        Some(Hintsfile::from_reader(&mut file).expect("couldn't read hints file"))
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
    pub(crate) fn was_aborted(&self) -> bool {
        self.context.abort_height.is_some()
    }

    /// Computes the next blocks to request, and sends a GETDATA request, advancing
    /// `last_block_request` up to the SwiftSync hints `stop_height`.
    fn get_blocks_to_download(&mut self) {
        // Fill new headroom immediately, including after the controller raises its limit.
        while !self.was_aborted() && self.can_request_more_blocks() {
            let prev_last_request = self.last_block_request;
            let mut blocks = Vec::with_capacity(SwiftSync::BLOCKS_PER_GETDATA);

            for _ in 0..SwiftSync::BLOCKS_PER_GETDATA {
                if self.last_block_request >= self.context.stop_height {
                    break;
                }

                let next_height = self.last_block_request + 1;
                let Ok(next_block) = self.chain.get_block_hash(next_height) else {
                    break;
                };

                blocks.push(next_block);
                self.last_block_request = next_height;
            }

            if blocks.is_empty() {
                break;
            }

            if let Err(err) = self.request_blocks(blocks) {
                // Roll back this batch so missing peers don't make us skip heights.
                self.last_block_request = prev_last_request;
                if !matches!(err, WireError::NoPeersAvailable) {
                    error!("Failed to request blocks: {err:?}");
                }
                break;
            }
        }
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

    fn indexing_workers(&self) -> usize {
        self.blocks
            .values()
            .filter(|block| block.processing_since.is_some())
            .count()
    }

    /// Slots stay occupied until the node handles their results, even if execution has finished.
    fn busy_workers(&self) -> usize {
        self.indexing_workers() + self.context.validating.len()
    }

    /// Call after each queue/slot change, including completion before any recursive inline work.
    fn record_worker_occupancy(&mut self) {
        let occupied = self.busy_workers();
        let queued_indexing = self.blocks.len().saturating_sub(self.indexing_workers());
        let queued = queued_indexing + self.context.validation_queue.len();
        self.context
            .worker_occupancy
            .record(Instant::now(), occupied, queued);
    }

    /// Starts eligible validation first, then fills remaining slots with unordered indexing work.
    fn pump_swiftsync(&mut self, hints: &mut Hintsfile) -> Result<(), WireError> {
        while self.busy_workers() < MAX_PARALLEL_WORKERS {
            let Some(height) = self.context.validation_queue.pop_front() else {
                break;
            };
            if let Err(error) = self.start_validating_swiftsync(height) {
                error!(
                    "Aborting SwiftSync: failed to start validation at height {height}: {error}"
                );
                self.context.abort_height = Some(height);
                self.record_worker_occupancy();
                return Ok(());
            }
        }

        let free = MAX_PARALLEL_WORKERS.saturating_sub(self.busy_workers());
        let to_index: Vec<BlockHash> = self
            .blocks
            .iter()
            .filter(|(_, block)| block.processing_since.is_none())
            .take(free)
            .map(|(hash, _)| *hash)
            .collect();

        for hash in to_index {
            let height = match self.chain.get_block_height(&hash) {
                Ok(Some(height)) => height,
                Ok(None) => {
                    let abort_height = self.context.indexed_prefix.saturating_add(1);
                    error!(
                        "Aborting SwiftSync: downloaded block {hash} is orphaned or marked invalid \
                         in the header index; indexed_prefix={} abort_height={abort_height}",
                        self.context.indexed_prefix
                    );
                    self.context.abort_height = Some(abort_height);
                    break;
                }
                Err(error) => {
                    let abort_height = self.context.indexed_prefix.saturating_add(1);
                    error!(
                        "Aborting SwiftSync: failed to look up downloaded block {hash} in the \
                         header index: {error}; indexed_prefix={} abort_height={abort_height}",
                        self.context.indexed_prefix
                    );
                    self.context.abort_height = Some(abort_height);
                    break;
                }
            };
            if let Err(error) = self.start_processing_swiftsync(hash, height, hints) {
                error!(
                    "Aborting SwiftSync: failed to start indexing block {hash} at height \
                     {height}: {error}"
                );
                self.context.abort_height = Some(height);
                break;
            }
        }
        self.record_worker_occupancy();
        Ok(())
    }

    fn previous_median_time_past(&self, header: &BlockHeader) -> Result<u32, Chain::Error> {
        let previous = self.chain.get_block_header(&header.prev_blockhash)?;
        previous
            .median_time_past_with(|current| self.chain.get_block_header(&current.prev_blockhash))
    }

    /// Spawns an unordered indexing task for a downloaded block.
    fn start_processing_swiftsync(
        &mut self,
        block_hash: BlockHash,
        block_height: u32,
        hints: &mut Hintsfile,
    ) -> Result<(), WireError> {
        debug!("indexing block {block_hash}");
        let entry = self
            .blocks
            .get_mut(&block_hash)
            .ok_or(WireError::BlockNotFound)?;
        if entry.processing_since.is_some() {
            return Ok(());
        }
        let Some(block_hints) = hints.indices_at_height(block_height) else {
            error!("We tried indexing block {block_height} but its hints are missing");
            return Ok(());
        };
        let unspent_indexes: HashSet<u32> = block_hints.into_iter().collect();
        entry.processing_since = Some(Instant::now());

        let block = Arc::clone(&entry.block);
        let creation_time = self.previous_median_time_past(&block.header)?;
        let consensus = Consensus::from(self.network);
        let salt = Arc::clone(&self.context.salt);
        let database = Arc::clone(self.context.utxo_db.as_ref().expect("initialized"));
        let node_sender = self.node_tx.clone();
        tokio::task::spawn_blocking(move || {
            let result = consensus
                .process_block_swiftsync(&block, block_height, &unspent_indexes, &salt)
                .and_then(|result| {
                    database
                        .index_block(&block, block_height, &unspent_indexes, creation_time)
                        .map(|()| result)
                });
            let _ = node_sender.send(NodeNotification::FromWorker((
                result,
                block_hash,
                block_height,
            )));
        });
        Ok(())
    }

    fn start_validating_swiftsync(&mut self, height: u32) -> Result<(), WireError> {
        let block = Arc::clone(
            &self
                .context
                .indexed_blocks
                .get(&height)
                .ok_or(WireError::BlockNotFound)?
                .block,
        );
        let block_hash = block.block_hash();
        let previous_median_time_past = self.previous_median_time_past(&block.header)?;
        let database = Arc::clone(self.context.utxo_db.as_ref().expect("initialized"));
        let consensus = Consensus::from(self.network);
        let node_sender = self.node_tx.clone();
        assert!(self.context.validating.insert(height));

        tokio::task::spawn_blocking(move || {
            let pop_started = Instant::now();
            let loaded = database.load_inputs(&block, height, previous_median_time_past);
            let prevout_pop_time = pop_started.elapsed();
            let (result, timings) = match loaded {
                Ok(LoadedInputs { utxos }) => {
                    let validation_started = Instant::now();
                    let result = consensus.validate_indexed_swiftsync_block(
                        &block,
                        height,
                        previous_median_time_past,
                        utxos,
                    );
                    (
                        result,
                        ValidationTimings {
                            prevout_fetch: prevout_pop_time,
                            prevout_delete: Duration::ZERO,
                            consensus: validation_started.elapsed(),
                        },
                    )
                }
                Err(error) => (
                    Err(error),
                    ValidationTimings {
                        prevout_fetch: prevout_pop_time,
                        prevout_delete: Duration::ZERO,
                        consensus: Duration::ZERO,
                    },
                ),
            };
            let _ = node_sender.send(NodeNotification::FromValidationWorker((
                result, block_hash, height, timings,
            )));
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
        let Some(mut hints) = Self::parse_hints_file(&self.datadir, self.network) else {
            return self;
        };

        let validation_idx = self.chain.get_validation_index().unwrap();
        if validation_idx >= hints.stop_height() {
            return self;
        }

        // Full witnesses are required for non-assumevalid script execution.
        self.witness_mode = WitnessMode::Full;
        self.context.stop_height = hints.stop_height();

        assert_eq!(
            validation_idx, 0,
            "Validation index should be 0 at the start of SwiftSync"
        );
        self.last_block_request = 0;
        self.context.processed_blocks = 0;
        self.context.indexed_prefix = 0;
        self.context.indexed_blocks.clear();
        self.context.downloaded_block_bytes = 0;
        self.context.validation_queue.clear();
        self.context.validating.clear();
        self.context.utxo_db = match SwiftSyncUtxoDb::create(&self.datadir) {
            Ok(database) => Some(database),
            Err(err) => {
                error!("Failed to create SwiftSync UTXO index: {err}");
                self.context.abort_height = Some(0);
                return self;
            }
        };
        self.context.download_window = DownloadWindow::default();
        // Existing connections may have header-sync traffic; start a fresh socket-read interval.
        self.socket_reads.set_observing(true);
        self.context.diagnostics = DownloadDiagnostics::default();
        self.context.worker_occupancy = WorkerOccupancy::default();
        self.record_worker_occupancy();
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
            "Performing SwiftSync up to height {}, download window={} MiB",
            hints.stop_height(),
            self.context.download_window.limit_bytes() / (1 << 20),
        );

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

        self.socket_reads.set_observing(false);
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
            // A prior diagnostic names the specific validation, indexing, or chain lookup error.
            // The temporary UTXO state cannot be trusted after any such failure.
            error!("Aborting SwiftSync after failure at height {invalid_h}");
            return LoopControl::Break;
        }

        // If we have reached the SwiftSync stop height, and we have added all the utreexo leaves
        // to the accumulator, we have finished.
        if let Some(final_acc) = self.swift_sync_finished() {
            self.log_download_diagnostics();
            self.log_worker_occupancy(true);
            self.handle_stop_height_reached(final_acc);
            return LoopControl::Break;
        }

        // Snapshot queues before maintenance retries or disconnects change their owners/ages.
        self.log_download_diagnostics();
        self.log_worker_occupancy(false);

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
        self.update_download_window();

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

    /// Probe useful throughput periodically, without treating the final drain as a slowdown.
    fn update_download_window(&mut self) {
        let now = Instant::now();

        // The median prevents one stalled peer from stretching every observation period.
        let response_ms = self.median_peer_latency(ServiceFlags::NETWORK);
        let can_probe = self.last_block_request < self.context.stop_height && response_ms.is_some();
        let response_ms = response_ms
            .unwrap_or(1_000.0)
            .clamp(0.0, SwiftSync::REQUEST_TIMEOUT as f64 * 1_000.0);
        let queued_blocks = self.blocks.len().saturating_sub(self.indexing_workers())
            + self.context.validation_queue.len();
        let growth_allowed = self.busy_workers() < MAX_PARALLEL_WORKERS || queued_blocks == 0;
        self.context
            .download_window
            .set_growth_allowed(growth_allowed);
        let waiting_bytes = self.unprocessed_block_bytes();
        let waiting_window_units = DownloadWindow::units_for_bytes(waiting_bytes);
        if let Some(sample) = self.context.download_window.update(
            now,
            Duration::from_secs_f64(response_ms / 1_000.0),
            can_probe,
            waiting_window_units,
        ) {
            info!(
                "SwiftSync download: window_mib={} waiting_mib={} waiting_blocks={} max_buffered_blocks={} growth_allowed={} estimated_block_bytes={} connected_peers={} sampled_window_mib={} useful_throughput={:.2} Mbps action={} sample_secs={:.3} sample_bytes={} baseline_window_mib={:?} baseline_mbps={:?} median_response_ms={:.1}",
                self.context.download_window.limit_bytes() / (1 << 20),
                waiting_window_units,
                self.unprocessed_blocks(),
                MAX_BUFFERED_BLOCKS,
                growth_allowed,
                self.context.download_window.estimated_block_bytes(),
                self.connected_peers(),
                sample.measured_limit,
                sample.bytes_per_second * 8.0 / 1_000_000.0,
                sample.action,
                sample.sample_duration.as_secs_f64(),
                sample.sample_bytes,
                sample.baseline_limit,
                sample
                    .baseline_bytes_per_second
                    .map(|rate| rate * 8.0 / 1_000_000.0),
                response_ms,
            );
        }
    }

    /// Logs slot peaks and the percentage of wall time queued blocks had no free worker slot.
    fn log_worker_occupancy(&mut self, finished: bool) {
        let occupied = self.busy_workers();
        let queued = self.blocks.len().saturating_sub(self.indexing_workers())
            + self.context.validation_queue.len();
        if let Some((interval, peak, all_busy_with_queue_time)) = self
            .context
            .worker_occupancy
            .take(Instant::now(), occupied, queued, finished)
        {
            info!(
                "SwiftSync workers: interval_secs={:.3} max_busy_workers={} worker_capacity={} all_busy_with_queue_pct={:.2}",
                interval.as_secs_f64(),
                peak,
                MAX_PARALLEL_WORKERS,
                100.0 * all_busy_with_queue_time.as_secs_f64() / interval.as_secs_f64(),
            );
        }
    }

    /// Logs independent interval rates and queue snapshots without affecting download decisions.
    /// Socket reads include partial messages and transport/control bytes, not TCP/IP overhead.
    /// Decoded block rates remain separate; neither measures decryption/decoding time.
    fn log_download_diagnostics(&mut self) {
        let now = Instant::now();
        let interval = now.duration_since(self.context.diagnostics.since);
        // The interval timer's immediate first tick has no useful rate to report.
        if interval < Duration::from_millis(1) {
            return;
        }
        let elapsed = interval.as_secs_f64();
        let diagnostics = std::mem::replace(
            &mut self.context.diagnostics,
            DownloadDiagnostics {
                since: now,
                ..Default::default()
            },
        );
        let mbps = |bytes: u64| bytes as f64 * 8.0 / elapsed / 1_000_000.0;
        let socket_reads = self.socket_reads.take();
        let socket_read_bytes: u64 = socket_reads.values().sum();
        // All connections count, including feelers/handshakes and peers that left this interval.
        let socket_read_peers = socket_reads.values().filter(|bytes| **bytes > 0).count();

        // Include idle eligible peers and departed peers with traffic or unreassigned requests.
        let mut queues: BTreeMap<PeerId, (usize, f64)> = BTreeMap::new();
        let mut eligible_peers = 0;
        for (&id, peer) in &self.peers {
            if peer.state == PeerStatus::Ready
                && peer.is_long_lived()
                && peer.services.has(ServiceFlags::NETWORK)
            {
                eligible_peers += 1;
                queues.entry(id).or_default();
            }
        }
        for id in diagnostics.peers.keys() {
            queues.entry(*id).or_default();
        }
        let mut inflight_blocks = 0;
        let mut ages = Vec::new();
        for (request, (peer, sent_at)) in &self.inflight {
            if matches!(request, InflightRequests::Blocks(_)) {
                let age = now.saturating_duration_since(*sent_at).as_secs_f64();
                let (count, oldest) = queues.entry(*peer).or_default();
                *count += 1;
                *oldest = oldest.max(age);
                inflight_blocks += 1;
                ages.push(age);
            }
        }
        ages.sort_by(f64::total_cmp);
        let median_age = ages.get(ages.len() / 2).copied().unwrap_or(0.0);
        let oldest_age = ages.last().copied().unwrap_or(0.0);
        let busy_workers = self.busy_workers();
        let oldest_worker = self
            .blocks
            .values()
            .chain(self.context.indexed_blocks.values())
            .filter_map(|block| block.processing_since)
            .map(|start| now.saturating_duration_since(start))
            .max()
            .unwrap_or_default();
        let received_blocks: u64 = diagnostics.peers.values().map(|p| p.received_blocks).sum();
        let received_bytes: u64 = diagnostics.peers.values().map(|p| p.received_bytes).sum();
        let ignored_blocks: u64 = diagnostics.peers.values().map(|p| p.ignored_blocks).sum();
        let ignored_bytes: u64 = diagnostics.peers.values().map(|p| p.ignored_bytes).sum();
        let other_peer_replies: u64 = diagnostics
            .peers
            .values()
            .map(|p| p.other_peer_replies)
            .sum();
        let window_bytes = self.context.download_window.limit_bytes();
        let pending_bytes = self.unprocessed_block_bytes();
        let pending_blocks =
            inflight_blocks + self.blocks.len() + self.context.indexed_blocks.len();
        let memory = process_memory();

        info!(
            "SwiftSync diagnostics: interval_secs={:.3} phase={} window_mib={} headroom_mib={} pending_mib={} pending_blocks={} rss_mib={} rss_anon_mib={} rss_file_mib={} swap_mib={} processed_total={} requested_height={} received_block_mbps={:.2} processed_block_mbps={:.2} ignored_block_mbps={:.2} received_blocks={} processed_blocks={} ignored_blocks={} other_peer_replies={} avg_received_block_bytes={} estimated_block_bytes={} inflight_blocks={} buffered_blocks={} queued_blocks={} busy_workers={} oldest_worker_secs={:.3} avg_worker_turnaround_ms={:.3} max_worker_turnaround_ms={:.3} validated_blocks={} avg_prevout_fetch_ms={:.3} avg_prevout_delete_ms={:.3} avg_consensus_validation_ms={:.3} node_queue_messages={} max_block_message_delay_ms={:.3} eligible_peers={} inflight_peers={} delivering_peers={} max_peer_inflight={} median_request_age_secs={:.1} oldest_request_age_secs={:.1} requests_age_ge_10s={} requests_age_ge_30s={} requests_age_ge_120s={} socket_read_mbps={:.2} socket_read_peers={}",
            elapsed,
            self.context.download_window.phase(now),
            window_bytes / (1 << 20),
            window_bytes.saturating_sub(pending_bytes) / (1 << 20),
            DownloadWindow::units_for_bytes(pending_bytes),
            pending_blocks,
            memory.rss_kib / 1_024,
            memory.anonymous_kib / 1_024,
            memory.file_kib / 1_024,
            memory.swap_kib / 1_024,
            self.context.processed_blocks,
            self.last_block_request,
            mbps(received_bytes),
            mbps(diagnostics.processed_bytes),
            mbps(ignored_bytes),
            received_blocks,
            diagnostics.processed_blocks,
            ignored_blocks,
            other_peer_replies,
            received_bytes.checked_div(received_blocks).unwrap_or(0),
            self.context.download_window.estimated_block_bytes(),
            inflight_blocks,
            self.blocks.len() + self.context.indexed_blocks.len(),
            self.blocks.len().saturating_sub(self.indexing_workers())
                + self.context.validation_queue.len(),
            busy_workers,
            oldest_worker.as_secs_f64(),
            diagnostics.worker_turnaround.as_secs_f64() * 1_000.0
                / diagnostics.worker_results.max(1) as f64,
            diagnostics.max_worker_turnaround.as_secs_f64() * 1_000.0,
            diagnostics.validation_results,
            diagnostics.prevout_fetch_time.as_secs_f64() * 1_000.0
                / diagnostics.validation_results.max(1) as f64,
            diagnostics.prevout_delete_time.as_secs_f64() * 1_000.0
                / diagnostics.validation_results.max(1) as f64,
            diagnostics.consensus_validation_time.as_secs_f64() * 1_000.0
                / diagnostics.validation_results.max(1) as f64,
            self.node_rx.len(),
            diagnostics.max_message_delay.as_secs_f64() * 1_000.0,
            eligible_peers,
            queues.values().filter(|(count, _)| *count > 0).count(),
            diagnostics
                .peers
                .values()
                .filter(|p| p.received_blocks > 0)
                .count(),
            queues.values().map(|(count, _)| *count).max().unwrap_or(0),
            median_age,
            oldest_age,
            ages.iter().filter(|age| **age >= 10.0).count(),
            ages.iter().filter(|age| **age >= 30.0).count(),
            ages.iter().filter(|age| **age >= 120.0).count(),
            mbps(socket_read_bytes),
            socket_read_peers,
        );

        // One compact line per tick avoids per-block logging on the hot path.
        let peers = queues
            .iter()
            .map(|(id, (pending, age))| {
                let traffic = diagnostics.peers.get(id);
                let received = traffic.map_or(0, |p| p.received_bytes);
                let ignored = traffic.map_or(0, |p| p.ignored_bytes);
                let latency = self
                    .peers
                    .get(id)
                    .and_then(|p| p.message_times.value())
                    .map_or_else(|| "-".to_owned(), |ms| format!("{ms:.1}"));
                format!(
                    "{id}/{pending}/{age:.1}/{:.2}/{:.2}/{latency}",
                    mbps(received),
                    mbps(ignored)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        info!(
            "SwiftSync peers: interval_secs={elapsed:.3} columns=peer/inflight/oldest_secs/received_mbps/ignored_mbps/general_latency_ms peers=[{peers}]"
        );
        // Keep the existing per-peer format intact for previous benchmark analysis scripts.
        let sockets = socket_reads
            .iter()
            .map(|(id, bytes)| format!("{id}/{:.2}", mbps(*bytes)))
            .collect::<Vec<_>>()
            .join(",");
        info!(
            "SwiftSync socket reads: interval_secs={elapsed:.3} columns=peer/socket_read_mbps peers=[{sockets}]"
        );
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
    fn handle_stop_height_reached(&mut self, final_acc: Stump) {
        let stop_height = self.context.stop_height;
        let final_agg = self.context.agg;
        let final_supply = self.context.supply;

        // Disable witnessless mode, since we are done with SwiftSync. UtreexoSync requires the
        // witness data in order to reconstruct P2WPKH and P2WSH outputs.
        self.witness_mode = WitnessMode::Full;

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

        info!("SwiftSync produced the following accumulator for {tip_hash}: \n{final_acc:?}");

        self.chain
            .mark_chain_as_assumed(final_acc, tip_hash)
            .unwrap();
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
                        let block_size = block.total_size();
                        let already_buffered = self.blocks.contains_key(&hash);
                        let assigned_peer = self
                            .inflight
                            .get(&InflightRequests::Blocks(hash))
                            .map(|(peer, _)| *peer);
                        self.context.diagnostics.received_block(
                            peer,
                            block_size,
                            already_buffered || assigned_peer.is_none(),
                            assigned_peer.is_some_and(|assigned| assigned != peer),
                            Instant::now().saturating_duration_since(time),
                        );
                        if already_buffered {
                            debug!(
                                "Received block {hash} from peer {peer}, but we already have it"
                            );
                            return Ok(());
                        }

                        let Some(_) = self.inflight.remove(&InflightRequests::Blocks(hash)) else {
                            // Retries cannot cancel old transfers; timeouts already hurt the score.
                            debug!("Ignoring unneeded block {hash} from peer {peer}");
                            return Ok(());
                        };

                        // Reply and return early if it's a user-requested block. Else continue handling it.
                        let Some(block) = self.check_is_user_block_and_reply(block)? else {
                            return Ok(());
                        };
                        self.context.download_window.observe_block_size(block_size);
                        self.context.downloaded_block_bytes = self
                            .context
                            .downloaded_block_bytes
                            .saturating_add(block_size);

                        let inflight_block = InflightBlock {
                            peer,
                            block: Arc::new(block),
                            // SwiftSync obtains prevouts from its temporary UTXO index.
                            aux_data: None,
                            processing_since: None,
                        };
                        self.blocks.insert(hash, inflight_block);
                        self.record_worker_occupancy();

                        self.pump_swiftsync(hints)?;
                        self.get_blocks_to_download();
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

            NodeNotification::FromWorker((result, block_hash, height)) => {
                self.handle_worker_notification(result, block_hash, height, hints)?;
            }

            NodeNotification::FromValidationWorker((result, block_hash, height, timings)) => {
                self.handle_validation_notification(result, block_hash, height, hints, timings)?;
            }
        }

        Ok(())
    }

    fn handle_worker_notification(
        &mut self,
        result: IndexWorkerResult,
        block_hash: BlockHash,
        height: u32,
        hints: &mut Hintsfile,
    ) -> Result<(), WireError> {
        let block = self
            .blocks
            .remove(&block_hash)
            .ok_or(WireError::BlockNotFound)?;

        match result {
            Ok((agg_re, unspent_amount, utreexo_adds)) => {
                self.context.agg += agg_re;
                self.context.supply += unspent_amount;
                self.pump_utreexo_adds(height, utreexo_adds);
                hints.take_indices(height);
                // Moving into `indexed_blocks` retains the existing byte charge.
                assert!(self.context.indexed_blocks.insert(height, block).is_none());

                while self
                    .context
                    .indexed_blocks
                    .contains_key(&(self.context.indexed_prefix + 1))
                {
                    self.context.indexed_prefix += 1;
                    self.context
                        .validation_queue
                        .push_back(self.context.indexed_prefix);
                }
                self.pump_swiftsync(hints)?;
                self.get_blocks_to_download();
            }
            Err(error) => {
                self.context.downloaded_block_bytes = self
                    .context
                    .downloaded_block_bytes
                    .checked_sub(block.block.total_size())
                    .expect("downloaded byte accounting includes indexing workers");
                self.handle_invalid_block(error, block.block.header, height, block.peer)?;
            }
        }
        self.record_worker_occupancy();
        Ok(())
    }

    fn handle_validation_notification(
        &mut self,
        result: Result<(), BlockchainError>,
        block_hash: BlockHash,
        height: u32,
        hints: &mut Hintsfile,
        timings: ValidationTimings,
    ) -> Result<(), WireError> {
        assert!(self.context.validating.remove(&height));
        let block = self
            .context
            .indexed_blocks
            .remove(&height)
            .ok_or(WireError::BlockNotFound)?;
        self.context.downloaded_block_bytes = self
            .context
            .downloaded_block_bytes
            .checked_sub(block.block.total_size())
            .expect("downloaded byte accounting includes indexed blocks");
        debug_assert_eq!(block.block.block_hash(), block_hash);
        if let Some(start) = block.processing_since {
            let elapsed = start.elapsed();
            let diagnostics = &mut self.context.diagnostics;
            diagnostics.worker_results += 1;
            diagnostics.worker_turnaround += elapsed;
            diagnostics.max_worker_turnaround = diagnostics.max_worker_turnaround.max(elapsed);
        }
        self.context.diagnostics.validation_results += 1;
        self.context.diagnostics.prevout_fetch_time += timings.prevout_fetch;
        self.context.diagnostics.prevout_delete_time += timings.prevout_delete;
        self.context.diagnostics.consensus_validation_time += timings.consensus;

        match result {
            Ok(()) => {
                let bytes = block.block.total_size();
                self.context
                    .download_window
                    .record_bytes(bytes, Instant::now());
                self.context.diagnostics.processed_blocks += 1;
                self.context.diagnostics.processed_bytes += bytes as u64;
                self.handle_valid_worker_block(block_hash, height, block);
                assert!(self.context.processed_blocks < self.context.stop_height);
                self.context.processed_blocks += 1;
                self.chain.update_ibd(IBDState::SwiftSync {
                    processed_blocks: self.context.processed_blocks,
                    total_blocks: self.context.stop_height,
                });
                self.pump_swiftsync(hints)?;
                self.get_blocks_to_download();
            }
            Err(error) => {
                self.handle_invalid_block(error, block.block.header, height, block.peer)?;
            }
        }
        self.record_worker_occupancy();
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

        // Local storage failures are not peer faults, but continuing cannot make progress.
        let Some(e) = Self::block_validation_err(chain_err) else {
            self.context.abort_height = Some(height);
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
            | BlockValidationErrors::NonFinalTransaction
            | BlockValidationErrors::CoinbaseNotMatured
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
        // TODO should we update header and block index (similar to `self.chain.update_view`)?
        if height % 1000 == 0 {
            info!(
                "SwiftSync block: block_hash={block_hash} height={height} tx_count={}",
                block.block.txdata.len(),
            );
        }

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
            use floresta_metrics::get_metrics;

            let avg = self.block_sync_avg.value().expect("at least one sample");
            let metrics = get_metrics();
            metrics.block_height.set(height.into());
            metrics.avg_block_processing_time.set(avg);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bitcoin::Block;
    use bitcoin::consensus::encode::deserialize_hex;
    use floresta_chain::BlockchainInterface;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;

    use super::*;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::SetupNodeArgs;
    use crate::p2p_wire::tests::utils::setup_node;

    #[test]
    fn compact_outpoint_key_uses_txid_lsb_and_vout() {
        let mut txid_bytes = [0_u8; 32];
        for (byte, value) in txid_bytes.iter_mut().zip(0_u8..) {
            *byte = value;
        }
        let outpoint = OutPoint::new(bitcoin::Txid::from_byte_array(txid_bytes), 0x4433_2211);
        let key = SwiftSyncUtxoDb::outpoint_key(outpoint);
        assert_eq!(&key[..12], &txid_bytes[..12]);
        assert_eq!(&key[12..], &0x4433_2211_u32.to_le_bytes());
    }

    #[test]
    fn reconstructable_utxo_packs_into_inline_value() {
        let public_key = [2_u8; 33];
        let pubkey_hash = bitcoin::WPubkeyHash::hash(&public_key);
        let utxo = UtxoData {
            txout: TxOut {
                value: Amount::from_sat(42_000),
                script_pubkey: bitcoin::ScriptBuf::new_p2wpkh(&pubkey_hash),
            },
            is_coinbase: true,
            creation_height: 840_000,
            creation_time: 1_700_000_000,
        };
        let input = TxIn {
            previous_output: OutPoint::null(),
            script_sig: bitcoin::ScriptBuf::new(),
            sequence: bitcoin::Sequence::MAX,
            witness: bitcoin::Witness::from_slice(&[public_key]),
        };

        let encoded = SwiftSyncUtxoDb::encode_utxo(&utxo);
        assert_eq!(encoded.len(), 8);
        let packed = u64::from_le_bytes(encoded.clone().try_into().unwrap());
        assert_eq!(packed & PACKED_AMOUNT_MASK, 42_000);
        assert_eq!((packed >> PACKED_SPK_SHIFT) & 0b111, 2);
        assert_eq!((packed >> PACKED_COINBASE_SHIFT) & 1, 1);
        assert_eq!(packed >> PACKED_HEIGHT_SHIFT, 840_000);
        assert_eq!(packed >> 63, 0);
        assert_eq!(
            SwiftSyncUtxoDb::decode_utxo(&encoded, &input).unwrap(),
            UtxoData {
                creation_time: 0,
                ..utxo.clone()
            }
        );

        let fallback = UtxoData {
            txout: TxOut {
                value: Amount::from_sat(PACKED_AMOUNT_MASK + 1),
                ..utxo.txout.clone()
            },
            ..utxo
        };
        let encoded = SwiftSyncUtxoDb::encode_utxo(&fallback);
        assert_eq!(
            encoded.len(),
            UTXO_METADATA_SIZE + 1 + SCRIPT_COMMITMENT_SIZE
        );
        let mut wrong_input = input;
        wrong_input.witness = bitcoin::Witness::from_slice(&[[3_u8; 33]]);
        assert!(SwiftSyncUtxoDb::decode_utxo(&encoded, &wrong_input).is_err());
    }

    #[test]
    fn non_reconstructable_fallback_decodes_without_commitment() {
        let utxo = UtxoData {
            txout: TxOut {
                value: Amount::from_sat(42_000),
                script_pubkey: bitcoin::ScriptBuf::from_bytes(vec![0x51]),
            },
            is_coinbase: false,
            creation_height: 840_000,
            creation_time: 1_700_000_000,
        };
        let input = TxIn {
            previous_output: OutPoint::null(),
            script_sig: bitcoin::ScriptBuf::new(),
            sequence: bitcoin::Sequence::MAX,
            witness: bitcoin::Witness::new(),
        };

        let encoded = SwiftSyncUtxoDb::encode_utxo(&utxo);
        assert!(encoded.len() > UTXO_METADATA_SIZE);
        assert_eq!(
            SwiftSyncUtxoDb::decode_utxo(&encoded, &input).unwrap(),
            utxo
        );
    }

    fn transaction_spending(previous_output: OutPoint) -> bitcoin::Transaction {
        bitcoin::Transaction {
            version: bitcoin::transaction::Version::ONE,
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![TxIn {
                previous_output,
                script_sig: bitcoin::ScriptBuf::new(),
                sequence: bitcoin::Sequence::MAX,
                witness: bitcoin::Witness::new(),
            }],
            output: vec![TxOut {
                value: Amount::from_sat(1),
                script_pubkey: bitcoin::ScriptBuf::new(),
            }],
        }
    }

    fn block_with_transactions(transactions: Vec<bitcoin::Transaction>) -> Block {
        let mut block: Block = deserialize_hex(
            include_str!("../../../../floresta-chain/testdata/mainnet_blocks.txt")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap();
        block.txdata = transactions;
        block
    }

    #[test]
    fn same_block_spends_bypass_the_database() {
        let database = SwiftSyncUtxoDb::create(&std::env::temp_dir()).unwrap();
        let coinbase = transaction_spending(OutPoint::null());
        let coinbase_outpoint = OutPoint::new(coinbase.compute_txid(), 0);
        let spender = transaction_spending(coinbase_outpoint);
        let block = block_with_transactions(vec![coinbase.clone(), spender]);

        database
            .index_block(&block, 1, &HashSet::new(), 123)
            .unwrap();
        let key = SwiftSyncUtxoDb::outpoint_key(coinbase_outpoint);
        assert_eq!(
            database
                .database
                .as_ref()
                .unwrap()
                .batch_fetch([key.as_slice()])
                .unwrap(),
            vec![None]
        );

        let loaded = database.load_inputs(&block, 1, 123).unwrap();
        assert_eq!(
            loaded.utxos[&coinbase_outpoint],
            UtxoData {
                txout: coinbase.output[0].clone(),
                is_coinbase: true,
                creation_height: 1,
                creation_time: 123,
            }
        );
    }

    #[test]
    fn same_block_overlay_rejects_backward_and_duplicate_spends() {
        let database = SwiftSyncUtxoDb::create(&std::env::temp_dir()).unwrap();
        let coinbase = transaction_spending(OutPoint::null());
        let external = OutPoint::new(bitcoin::Txid::from_byte_array([9_u8; 32]), 0);
        let creator = transaction_spending(external);
        let creator_outpoint = OutPoint::new(creator.compute_txid(), 0);
        let backward_spender = transaction_spending(creator_outpoint);
        let backward = block_with_transactions(vec![coinbase.clone(), backward_spender, creator]);
        assert!(matches!(
            database.load_inputs(&backward, 1, 123),
            Err(BlockchainError::BlockValidation(
                BlockValidationErrors::UtxoNotFound(outpoint)
            )) if outpoint == creator_outpoint
        ));

        let first_spender = transaction_spending(external);
        let second_spender = transaction_spending(external);
        let duplicate = block_with_transactions(vec![coinbase, first_spender, second_spender]);
        assert!(matches!(
            database.load_inputs(&duplicate, 1, 123),
            Err(BlockchainError::BlockValidation(
                BlockValidationErrors::UtxoNotFound(outpoint)
            )) if outpoint == external
        ));
    }

    #[tokio::test]
    async fn pop_inputs_returns_and_reclaims_each_indexed_prevout_once() {
        let node = node();
        let database = node.context.utxo_db.as_ref().unwrap();
        let outpoint = OutPoint::new(bitcoin::Txid::from_byte_array([7_u8; 32]), 3);
        let key = SwiftSyncUtxoDb::outpoint_key(outpoint);
        database
            .database
            .as_ref()
            .unwrap()
            .write_only()
            .unwrap()
            .put_batch([(key.as_slice(), b"prevout".as_slice())])
            .unwrap();

        assert_eq!(
            database.pop_inputs(&[(outpoint, key)]).unwrap(),
            vec![b"prevout".to_vec()]
        );
        assert_eq!(
            database
                .database
                .as_ref()
                .unwrap()
                .batch_fetch([key.as_slice()])
                .unwrap(),
            vec![None]
        );
        assert!(database.pop_inputs(&[(outpoint, key)]).is_err());
    }

    #[test]
    fn worker_peak_retains_brief_full_occupancy_and_carries_over_busy_slots() {
        let mut usage = WorkerOccupancy::default();
        let start = usage.since;
        for occupied in [1, 4, 6, 2] {
            usage.record(start, occupied, 0);
        }
        assert!(
            usage
                .take(start + Duration::from_secs(29), 2, 0, false)
                .is_none()
        );
        assert_eq!(usage.since, start);
        assert_eq!(usage.peak, 6);

        assert_eq!(
            usage.take(start + Duration::from_secs(30), 2, 0, false),
            Some((Duration::from_secs(30), 6, Duration::ZERO)),
        );
        // Both remaining slots can become idle without any new dispatches.
        usage.record(start + Duration::from_secs(35), 0, 0);
        assert_eq!(
            usage.take(start + Duration::from_secs(60), 0, 0, false),
            Some((Duration::from_secs(30), 2, Duration::ZERO)),
        );
        assert_eq!(
            usage.take(start + Duration::from_secs(90), 0, 0, false),
            Some((Duration::from_secs(30), 0, Duration::ZERO)),
        );
    }

    #[test]
    fn worker_peak_reports_the_final_partial_interval() {
        let mut usage = WorkerOccupancy::default();
        let start = usage.since;
        usage.record(start, 3, 0);
        let finish = start + Duration::from_secs(7);
        usage.record(finish, 0, 0);
        assert!(usage.take(finish, 0, 0, false).is_none());
        assert_eq!(
            usage.take(finish, 0, 0, true),
            Some((Duration::from_secs(7), 3, Duration::ZERO))
        );
        assert!(usage.take(finish, 0, 0, true).is_none());
    }

    #[test]
    fn worker_pressure_requires_full_slots_and_a_queue() {
        let mut usage = WorkerOccupancy::default();
        let start = usage.since;
        // Neither full occupancy alone nor a queue with a free slot counts.
        usage.record(start, MAX_PARALLEL_WORKERS, 0);
        usage.record(start + Duration::from_secs(5), MAX_PARALLEL_WORKERS - 1, 1);
        usage.record(start + Duration::from_secs(10), MAX_PARALLEL_WORKERS, 2);
        usage.record(start + Duration::from_secs(13), MAX_PARALLEL_WORKERS, 1);
        usage.record(start + Duration::from_secs(19), MAX_PARALLEL_WORKERS - 1, 1);
        usage.record(start + Duration::from_secs(20), MAX_PARALLEL_WORKERS, 0);
        assert_eq!(
            usage.take(
                start + Duration::from_secs(30),
                MAX_PARALLEL_WORKERS,
                0,
                false
            ),
            // Nine seconds out of thirty: all_busy_with_queue_pct=30.00.
            Some((
                Duration::from_secs(30),
                MAX_PARALLEL_WORKERS,
                Duration::from_secs(9),
            )),
        );
    }

    #[test]
    fn worker_pressure_spans_reporting_boundaries_without_double_counting() {
        let mut usage = WorkerOccupancy::default();
        let start = usage.since;
        usage.record(start + Duration::from_secs(20), MAX_PARALLEL_WORKERS, 1);
        assert!(
            usage
                .take(
                    start + Duration::from_secs(25),
                    MAX_PARALLEL_WORKERS,
                    1,
                    false
                )
                .is_none()
        );
        assert_eq!(
            usage.take(
                start + Duration::from_secs(30),
                MAX_PARALLEL_WORKERS,
                1,
                false
            ),
            Some((
                Duration::from_secs(30),
                MAX_PARALLEL_WORKERS,
                Duration::from_secs(10)
            )),
        );
        assert_eq!(
            usage.take(
                start + Duration::from_secs(60),
                MAX_PARALLEL_WORKERS,
                1,
                false
            ),
            Some((
                Duration::from_secs(30),
                MAX_PARALLEL_WORKERS,
                Duration::from_secs(30)
            )),
        );
        // Final partial interval includes pressure up to completion, then idle time.
        usage.record(start + Duration::from_secs(63), 0, 0);
        let finish = start + Duration::from_secs(67);
        assert_eq!(
            usage.take(finish, 0, 0, true),
            Some((
                Duration::from_secs(7),
                MAX_PARALLEL_WORKERS,
                Duration::from_secs(3)
            )),
        );
        assert!(usage.take(finish, 0, 0, true).is_none());
    }

    fn node() -> UtreexoNode<Arc<ChainState<FlatChainStore>>, SwiftSync> {
        node_with_blocks(175)
    }

    fn node_with_blocks(
        num_blocks: usize,
    ) -> UtreexoNode<Arc<ChainState<FlatChainStore>>, SwiftSync> {
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new());
        let args = SetupNodeArgs::new(
            vec![peer; 2],
            false,
            Network::Bitcoin,
            format!("./tmp-db/{}.download_window", rand::random::<u32>()),
            num_blocks,
        );
        let mut node = setup_node::<SwiftSync>(args);
        for peer in node.peers.values_mut() {
            peer.services |= ServiceFlags::NETWORK;
            peer.message_times.add(1.0);
        }
        node.fixed_peers.clear();
        node.inflight.clear();
        node.context.stop_height = num_blocks as u32;
        node.context.utxo_db = Some(SwiftSyncUtxoDb::create(&node.datadir).unwrap());
        node
    }

    #[tokio::test]
    async fn byte_window_keeps_network_inflight_bounded() {
        let mut node = node_with_blocks(SwiftSync::MAX_INFLIGHT_REQUESTS * 2);
        node.get_blocks_to_download();
        assert_eq!(node.inflight.len(), SwiftSync::MAX_INFLIGHT_REQUESTS);
        assert_eq!(
            node.unprocessed_block_bytes(),
            SwiftSync::MAX_INFLIGHT_REQUESTS * node.context.download_window.estimated_block_bytes()
        );
        assert!(!node.can_request_more_blocks());
    }

    #[tokio::test]
    async fn byte_window_keeps_buffered_block_count_bounded() {
        let mut node = node();
        let block: Block = deserialize_hex(
            include_str!("../../../../floresta-chain/testdata/mainnet_blocks.txt")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap();
        let block = Arc::new(block);
        for height in 0..MAX_BUFFERED_BLOCKS as u32 {
            node.context.indexed_blocks.insert(
                height,
                InflightBlock {
                    peer: 0,
                    block: Arc::clone(&block),
                    aux_data: None,
                    processing_since: None,
                },
            );
        }
        assert_eq!(node.unprocessed_blocks(), MAX_BUFFERED_BLOCKS);
        assert!(!node.can_request_more_blocks());

        for height in 0..SwiftSync::BLOCKS_PER_GETDATA as u32 {
            node.context.indexed_blocks.remove(&height);
        }
        assert!(node.can_request_more_blocks());
    }

    #[tokio::test]
    async fn fills_the_window_and_waits_for_a_shrunken_window_to_drain() {
        let mut node = node_with_blocks(SwiftSync::MAX_INFLIGHT_REQUESTS * 2);
        assert_eq!(node.context.download_window.limit_bytes(), 1 << 30);
        node.context.download_window.set_limit_for_test(100);
        let initial_window = node.context.download_window.limit();
        let initial_blocks =
            initial_window / SwiftSync::BLOCKS_PER_GETDATA * SwiftSync::BLOCKS_PER_GETDATA;
        // One available peer can fill the global window without a separate per-peer limit.
        node.peers.retain(|&id, _| id == 0);
        node.get_blocks_to_download();
        assert_eq!(node.unprocessed_blocks(), initial_blocks);
        assert_eq!(
            node.unprocessed_block_bytes(),
            initial_blocks * node.context.download_window.estimated_block_bytes()
        );
        assert_eq!(node.last_block_request as usize, initial_blocks);
        assert!(!node.can_request_more_blocks());

        // Free one batch while the rest of the window is still outstanding.
        let completed: Vec<_> = node.inflight.keys().take(5).cloned().collect();
        for request in completed {
            node.inflight.remove(&request);
        }
        node.get_blocks_to_download();
        assert_eq!(node.unprocessed_blocks(), initial_blocks);
        assert_eq!(node.last_block_request as usize, initial_blocks + 5);

        let now = Instant::now() + Duration::from_secs(30);
        node.context.download_window.record_bytes(3_000, now);
        node.context.download_window.probe_down_next();
        node.context
            .download_window
            .update(now, Duration::ZERO, true, initial_blocks)
            .unwrap();
        let smaller_window = node.context.download_window.limit();
        let smaller_blocks =
            smaller_window / SwiftSync::BLOCKS_PER_GETDATA * SwiftSync::BLOCKS_PER_GETDATA;
        assert!(smaller_window < initial_window);
        // The first probe shrinks the window without cancelling existing downloads.
        node.get_blocks_to_download();
        assert_eq!(node.unprocessed_blocks(), initial_blocks);
        assert_eq!(node.last_block_request as usize, initial_blocks + 5);

        // No sample is produced while the old requests still exceed the cap.
        let now = now + Duration::from_secs(60);
        assert!(
            node.context
                .download_window
                .update(now, Duration::ZERO, true, initial_blocks)
                .is_none()
        );
        let completed: Vec<_> = node
            .inflight
            .keys()
            .take(initial_blocks - smaller_blocks + 5)
            .cloned()
            .collect();
        for request in completed {
            node.inflight.remove(&request);
        }
        node.get_blocks_to_download();
        assert_eq!(node.unprocessed_blocks(), smaller_blocks);
        assert_eq!(node.last_block_request as usize, initial_blocks + 10);

        // Draining starts settling; only the following full sample can reject the trial.
        assert!(
            node.context
                .download_window
                .update(now, Duration::ZERO, true, smaller_blocks)
                .is_none()
        );
        let now = now + Duration::from_secs(35);
        node.context
            .download_window
            .update(now, Duration::ZERO, true, smaller_blocks)
            .unwrap();
        assert_eq!(node.context.download_window.limit(), initial_window);
        node.get_blocks_to_download();
        assert_eq!(node.unprocessed_blocks(), initial_blocks);
        assert_eq!(
            node.last_block_request as usize,
            initial_blocks + 10 + initial_blocks - smaller_blocks
        );
    }

    #[tokio::test]
    async fn refill_stops_at_the_stop_height_or_when_no_peers_are_available() {
        let mut node = node();
        node.peers.clear();
        node.get_blocks_to_download();
        assert_eq!(node.last_block_request, 0);
        assert!(node.inflight.is_empty());

        let mut node = self::node();
        node.context.stop_height = 17;
        node.get_blocks_to_download();
        assert_eq!(node.last_block_request, 17);
        assert_eq!(node.unprocessed_blocks(), 17);
        node.get_blocks_to_download();
        assert_eq!(node.unprocessed_blocks(), 17);
    }

    #[tokio::test]
    async fn diagnostic_snapshots_reset_only_logging_counters() {
        let mut node = node();
        let socket_bytes = node.socket_reads.register(0);
        node.socket_reads.set_observing(true);
        socket_bytes.fetch_add(1_000, std::sync::atomic::Ordering::Relaxed);
        node.context.worker_occupancy.record(Instant::now(), 6, 0);
        let worker_interval_start = node.context.worker_occupancy.since;
        node.get_blocks_to_download();
        let inflight = node.inflight.clone();
        let window = node.context.download_window.limit();
        let requested_height = node.last_block_request;
        let latency = node.peers[&0].message_times.value();
        let diagnostics = &mut node.context.diagnostics;
        diagnostics.since = Instant::now() - Duration::from_secs(5);
        diagnostics.received_block(0, 100, false, false, Duration::from_millis(10));
        diagnostics.processed_blocks = 1;
        diagnostics.processed_bytes = 100;
        diagnostics.worker_results = 1;
        diagnostics.worker_turnaround = Duration::from_millis(2);
        diagnostics.max_worker_turnaround = Duration::from_millis(2);
        diagnostics.validation_results = 1;
        diagnostics.prevout_fetch_time = Duration::from_millis(3);
        diagnostics.consensus_validation_time = Duration::from_millis(5);
        diagnostics.prevout_delete_time = Duration::from_millis(4);

        node.log_download_diagnostics();

        assert_eq!(socket_bytes.load(std::sync::atomic::Ordering::Relaxed), 0);
        // Five-second diagnostics must not reset the thirty-second worker peak.
        assert_eq!(node.context.worker_occupancy.peak, 6);
        assert_eq!(node.context.worker_occupancy.since, worker_interval_start);
        node.context.worker_occupancy.since = Instant::now() - Duration::from_secs(30);
        node.log_worker_occupancy(false);
        assert_eq!(node.context.worker_occupancy.peak, 0);
        assert_eq!(node.inflight, inflight);
        assert_eq!(node.context.download_window.limit(), window);
        assert_eq!(node.last_block_request, requested_height);
        assert_eq!(node.context.processed_blocks, 0);
        assert!(node.blocks.is_empty());
        assert_eq!(node.peers[&0].message_times.value(), latency);
        assert_eq!(node.peers[&0].banscore, 0);
        let diagnostics = &node.context.diagnostics;
        assert!(diagnostics.peers.is_empty());
        assert_eq!(diagnostics.processed_blocks, 0);
        assert_eq!(diagnostics.processed_bytes, 0);
        assert_eq!(diagnostics.max_message_delay, Duration::ZERO);
        assert_eq!(diagnostics.worker_results, 0);
        assert_eq!(diagnostics.worker_turnaround, Duration::ZERO);
        assert_eq!(diagnostics.max_worker_turnaround, Duration::ZERO);
        assert_eq!(diagnostics.validation_results, 0);
        assert_eq!(diagnostics.prevout_fetch_time, Duration::ZERO);
        assert_eq!(diagnostics.prevout_delete_time, Duration::ZERO);
        assert_eq!(diagnostics.consensus_validation_time, Duration::ZERO);
    }

    #[test]
    fn dynamic_block_timeout_has_safety_bounds() {
        let context = SwiftSync::default();
        assert_eq!(context.block_request_timeout(None), Duration::from_secs(10));
        assert_eq!(
            context.block_request_timeout(Some(Duration::from_millis(100))),
            Duration::from_millis(2_400)
        );
        assert_eq!(
            context.block_request_timeout(Some(Duration::from_secs(2))),
            Duration::from_secs(10)
        );
        assert_eq!(
            context.block_request_timeout(Some(Duration::from_secs(20))),
            Duration::from_secs(30)
        );
    }

    #[tokio::test]
    async fn timed_out_requests_are_retried_without_banning_and_unneeded_blocks_are_ignored() {
        let mut node = node();
        let block: Block = deserialize_hex(
            include_str!("../../../../floresta-chain/testdata/mainnet_blocks.txt")
                .lines()
                .nth(1)
                .unwrap(),
        )
        .unwrap();
        let hash = block.block_hash();
        let request = InflightRequests::Blocks(hash);
        let timeout = node
            .context
            .block_request_timeout(node.average_peer_latency(ServiceFlags::NETWORK));
        node.inflight.insert(
            request.clone(),
            (0, Instant::now() - timeout - Duration::from_millis(1)),
        );
        // Make peer 1 the only eligible replacement.
        node.peers.get_mut(&0).unwrap().services = ServiceFlags::NONE;
        node.check_for_timeout().unwrap();
        assert_eq!(node.peers[&0].banscore, 0);
        assert_eq!(node.inflight[&request].0, 1);

        // Simulate the replacement having already completed.
        node.inflight.remove(&request);
        let mut hints =
            Hintsfile::from_reader(&mut &include_bytes!("../tests/test_data/bitcoin.hints")[..])
                .unwrap();
        for peer in [0, 1] {
            let latency = node.peers[&peer].message_times.value();
            node.handle_message(
                NodeNotification::FromPeer(
                    peer,
                    PeerMessages::Block(block.clone()),
                    Instant::now(),
                ),
                &mut hints,
            )
            .await
            .unwrap();
            assert_eq!(node.peers[&peer].banscore, 0);
            assert_eq!(node.peers[&peer].message_times.value(), latency);
        }
        assert!(node.blocks.is_empty());
    }

    #[tokio::test]
    async fn worker_pressure_tracks_queued_arrivals() {
        let mut node = node_with_blocks(MAX_PARALLEL_WORKERS + 2);
        let blocks: Vec<Block> =
            include_str!("../../../../floresta-chain/testdata/mainnet_blocks.txt")
                .lines()
                .skip(1)
                .take(MAX_PARALLEL_WORKERS + 2)
                .map(|line| deserialize_hex(line).unwrap())
                .collect();
        let mut hints =
            Hintsfile::from_reader(&mut &include_bytes!("../tests/test_data/bitcoin.hints")[..])
                .unwrap();
        node.last_block_request = (MAX_PARALLEL_WORKERS + 2) as u32;
        node.context.stump_updater = Some(StumpUpdater::spawn(
            Stump::new(),
            0,
            (MAX_PARALLEL_WORKERS + 2) as u32,
        ));

        // Simulate all occupied slots, then accept two more downloaded blocks.
        for block in &blocks[..MAX_PARALLEL_WORKERS] {
            let entry = InflightBlock {
                peer: 0,
                block: Arc::new(block.clone()),
                aux_data: None,
                processing_since: Some(Instant::now()),
            };
            node.blocks.insert(block.block_hash(), entry);
        }
        node.record_worker_occupancy();
        assert!(!node.context.worker_occupancy.all_busy_with_queue);
        for block in &blocks[MAX_PARALLEL_WORKERS..] {
            node.inflight.insert(
                InflightRequests::Blocks(block.block_hash()),
                (0, Instant::now()),
            );
            node.handle_message(
                NodeNotification::FromPeer(0, PeerMessages::Block(block.clone()), Instant::now()),
                &mut hints,
            )
            .await
            .unwrap();
            assert!(node.context.worker_occupancy.all_busy_with_queue);
        }
        assert_eq!(node.context.processed_blocks, 0);
        assert_eq!(node.blocks.len(), MAX_PARALLEL_WORKERS + 2);
        assert_eq!(node.busy_workers(), MAX_PARALLEL_WORKERS);
        assert!(node.context.worker_occupancy.all_busy_with_queue);
        assert_eq!(node.context.worker_occupancy.peak, MAX_PARALLEL_WORKERS);
    }

    #[tokio::test]
    async fn duplicate_block_responses_are_counted_once() {
        for first_peer in [0, 1] {
            let mut node = node();
            let block: Block = deserialize_hex(
                include_str!("../../../../floresta-chain/testdata/mainnet_blocks.txt")
                    .lines()
                    .nth(1)
                    .unwrap(),
            )
            .unwrap();
            let hash = block.block_hash();
            let size = block.total_size();
            node.context.stop_height = 1;
            node.last_block_request = 1;
            node.context.stump_updater = Some(StumpUpdater::spawn(Stump::new(), 0, 1));
            node.inflight
                .insert(InflightRequests::Blocks(hash), (1, Instant::now()));
            let mut hints = Hintsfile::from_reader(
                &mut &include_bytes!("../tests/test_data/bitcoin.hints")[..],
            )
            .unwrap();

            for peer in [first_peer, 1 - first_peer, first_peer] {
                node.handle_message(
                    NodeNotification::FromPeer(
                        peer,
                        PeerMessages::Block(block.clone()),
                        Instant::now(),
                    ),
                    &mut hints,
                )
                .await
                .unwrap();
            }
            assert_eq!(node.context.downloaded_block_bytes, size);
            assert_eq!(node.unprocessed_block_bytes(), size);
            while node.context.processed_blocks == 0 {
                let notification =
                    tokio::time::timeout(Duration::from_secs(5), node.node_rx.recv())
                        .await
                        .expect("worker notification timeout")
                        .expect("worker notification channel");
                node.handle_message(notification, &mut hints).await.unwrap();
            }
            assert_eq!(node.context.processed_blocks, 1);
            assert_eq!(node.context.downloaded_block_bytes, 0);
            assert_eq!(node.unprocessed_block_bytes(), 0);
            // Indexing and validation share one occupied slot for the block's lifetime.
            assert_eq!(node.busy_workers(), 0);
            assert_eq!(node.context.worker_occupancy.peak, 1);
            assert!(!node.context.worker_occupancy.all_busy_with_queue);
            assert_eq!(
                node.context.worker_occupancy.all_busy_with_queue_time,
                Duration::ZERO
            );
            assert_eq!(node.peers[&0].banscore, 0);
            assert_eq!(node.peers[&1].banscore, 0);
            let diagnostics = &node.context.diagnostics;
            assert_eq!(diagnostics.processed_blocks, 1);
            assert_eq!(diagnostics.processed_bytes, size as u64);
            assert_eq!(diagnostics.worker_results, 1);
            assert_eq!(diagnostics.validation_results, 1);
            assert_eq!(
                diagnostics
                    .peers
                    .values()
                    .map(|p| p.received_bytes)
                    .sum::<u64>(),
                3 * size as u64
            );
            assert_eq!(
                diagnostics
                    .peers
                    .values()
                    .map(|p| p.ignored_bytes)
                    .sum::<u64>(),
                2 * size as u64
            );
            assert_eq!(
                diagnostics
                    .peers
                    .values()
                    .map(|p| p.ignored_blocks)
                    .sum::<u64>(),
                2
            );
            assert_eq!(
                diagnostics
                    .peers
                    .values()
                    .map(|p| p.other_peer_replies)
                    .sum::<u64>(),
                u64::from(first_peer != 1)
            );
            let sample = node
                .context
                .download_window
                .update(
                    Instant::now() + Duration::from_secs(30),
                    Duration::ZERO,
                    true,
                    DownloadWindow::units_for_bytes(node.unprocessed_block_bytes()),
                )
                .unwrap();
            assert!(sample.bytes_per_second > 0.0);
            assert!(sample.bytes_per_second <= size as f64 / 30.0);
        }
    }

    #[tokio::test]
    async fn timeouts_from_multiple_peers_leave_window_decisions_to_throughput() {
        let mut node = node();
        let initial_window = node.context.download_window.limit();
        let timeout = node
            .context
            .block_request_timeout(node.average_peer_latency(ServiceFlags::NETWORK));
        let expired = Instant::now() - timeout - Duration::from_millis(1);
        for peer in [0, 1] {
            let hash = node.chain.get_block_hash(peer + 1).unwrap();
            node.inflight
                .insert(InflightRequests::Blocks(hash), (peer, expired));
        }
        assert!(matches!(
            node.check_for_timeout(),
            Err(WireError::NoPeersAvailable)
        ));
        assert_eq!(node.inflight.len(), 2);
        for peer in node.peers.values() {
            assert_eq!(peer.banscore, 0);
            assert!(peer.message_times.value().unwrap() > 1.0);
        }
        assert_eq!(node.context.download_window.limit(), initial_window);

        // Despite both peers timing out, throughput sampling starts a normal downward probe.
        let now = Instant::now() + Duration::from_secs(30);
        node.context.download_window.record_bytes(3_000, now);
        let waiting_units = DownloadWindow::units_for_bytes(node.unprocessed_block_bytes());
        let sample = node
            .context
            .download_window
            .update(now, Duration::ZERO, true, waiting_units)
            .unwrap();
        assert_eq!(sample.action, "probe-down");
        assert!(node.context.download_window.limit() < initial_window);
    }

    #[test]
    fn ordinary_sync_retains_its_fixed_window_and_timeout_penalty() {
        use crate::node::sync_ctx::SyncNode;

        assert_eq!(
            SyncNode::default().block_download_window(),
            SyncNode::BLOCKS_PER_GETDATA * SyncNode::MAX_CONCURRENT_GETDATA
        );
        const { assert!(SyncNode::PENALIZE_BLOCK_TIMEOUT) };
    }
}

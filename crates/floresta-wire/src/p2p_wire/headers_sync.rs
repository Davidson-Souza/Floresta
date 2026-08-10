// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-peer headers pre-synchronization state machine.
//!
//! Protects against disk-fill DoS during initial header download. Without this
//! guard a malicious peer can serve a long chain of individually-valid low-work
//! headers, filling our header storage before we discover the chain is useless.
//!
//! The algorithm runs in two phases:
//!
//! **PRESYNC** -- headers are validated (PoW + continuity) but never written to
//! disk. Sparse one-bit commitments are stored at every N-th header.
//! Once the chain accumulates sufficient work the state advances to REDOWNLOAD.
//!
//! **REDOWNLOAD** -- the same headers are re-requested from the peer, verified
//! against the stored commitments, and released from the front of the buffer once
//! it exceeds the per-network lookahead size. When sufficient chainwork is
//! re-accumulated all remaining buffered headers are flushed unconditionally.

/// Hard cap on stored commitment bits. Bitcoin Core uses `6 * elapsed_secs / commitment_period`
/// (headerssync.cpp:43); that dynamic bound needs the start block's MTP. 24k covers ~15M mainnet heights.
const MAX_COMMITMENTS: u64 = 24_000;

/// Byte length of the u64 commitment salt serialized as little-endian.
const COMMITMENT_SALT_LEN: usize = 8;
/// Total byte length of the SHA-256 input: salt (8 bytes) || block_hash (32 bytes).
const COMMITMENT_INPUT_LEN: usize = COMMITMENT_SALT_LEN + 32;

use std::collections::VecDeque;

use bitcoin::BlockHash;
use bitcoin::CompactTarget;
use bitcoin::Network;
use bitcoin::Target;
use bitcoin::TxMerkleNode;
use bitcoin::Work;
use bitcoin::block::Header;
use bitcoin::block::Version;
use bitcoin::consensus::params::Params;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256;
use floresta_chain::minimum_chain_work;
use tracing::warn;

/// Stateless check that `new_bits` is a permitted successor to `old_bits` at `height`.
///
/// Mirrors Bitcoin Core's `PermittedDifficultyTransition` (pow.cpp): at a retarget
/// boundary the target may move by at most 4x in either direction (capped at the
/// network's proof-of-work limit); between boundaries it must not change at all.
/// Needs no timestamps, so it can run on headers that are not yet in storage.
/// Without it an attacker with limited hashrate could compress the work into a
/// few very hard headers and have a greater chance of reaching the work
/// threshold by luck, since expected cost is the same but variance is not.
fn permitted_difficulty_transition(
    params: &Params,
    height: u32,
    old_bits: CompactTarget,
    new_bits: CompactTarget,
) -> bool {
    // Min-difficulty networks (testnet3/4, regtest) may reset to the limit at any height.
    if params.allow_min_difficulty_blocks {
        return true;
    }

    if u64::from(height) % params.difficulty_adjustment_interval() != 0 {
        // Not a retarget height: difficulty must stay exactly the same.
        return old_bits == new_bits;
    }

    let old_target = Target::from_compact(old_bits);
    let observed = Target::from_compact(new_bits);

    // Largest permitted target (easiest difficulty): old * 4, capped at the PoW limit.
    // Round-trip through compact encoding so we compare against what `bits` can
    // express, as Core does with SetCompact(GetCompact()).
    let max_target = Target::from_compact(
        old_target
            .max_transition_threshold(params)
            .to_compact_lossy(),
    );
    if observed > max_target {
        return false;
    }

    // Smallest permitted target (hardest difficulty): old / 4.
    let min_target = Target::from_compact(old_target.min_transition_threshold().to_compact_lossy());
    observed >= min_target
}

/// Per-network parameters for the headers pre-synchronization algorithm.
///
/// Values sourced from Bitcoin Core's
/// [`chainparams.cpp`](https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp).
#[derive(Debug, Clone)]
pub struct HeadersSyncParams {
    /// Commitment slot interval: one bit is recorded per this many headers.
    pub commitment_period: u64,
    /// Number of headers kept in the REDOWNLOAD buffer before releasing from the front.
    /// Sized so that ~24 commitment periods worth of headers are held before release.
    pub redownload_buffer_size: usize,
}

impl HeadersSyncParams {
    pub fn for_network(network: Network) -> Self {
        match network {
            // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L202-L203
            Network::Bitcoin => Self {
                commitment_period: 641,
                redownload_buffer_size: 15218,
            },
            // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L311-L312
            Network::Testnet => Self {
                commitment_period: 673,
                redownload_buffer_size: 14460,
            },
            // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L424-L425
            Network::Testnet4 => Self {
                commitment_period: 606,
                redownload_buffer_size: 16092,
            },
            // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L549-L550
            Network::Signet => Self {
                commitment_period: 620,
                redownload_buffer_size: 15724,
            },
            // https://github.com/bitcoin/bitcoin/blob/8d5515465542336d3d0fb83935d79783e91048a0/src/kernel/chainparams.cpp#L685-L686
            _ => Self {
                commitment_period: 275,
                redownload_buffer_size: 7017,
            },
        }
    }
}

/// The phase of the [`HeadersSyncState`] state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresyncPhase {
    /// Validating headers without writing them to disk; building commitments.
    Presync,
    /// Chain passed the work gate; re-downloading to verify commitments.
    Redownload,
    /// All headers verified and released. This instance can be discarded.
    Final,
    /// PRESYNC was aborted because the peer's chain exceeded the internal commitment cap.
    /// Not misbehavior; the peer is silently dropped without banning.
    Aborted,
}

/// A block header with `prev_blockhash` elided to save memory.
/// Dropping the 32-byte field halves per-entry cost; it is always recoverable from the preceding entry.
#[derive(Debug, Clone)]
pub struct CompressedHeader {
    version: i32,
    merkle_root: TxMerkleNode,
    time: u32,
    bits: CompactTarget,
    nonce: u32,
}

impl CompressedHeader {
    fn from_header(h: &Header) -> Self {
        Self {
            version: h.version.to_consensus(),
            merkle_root: h.merkle_root,
            time: h.time,
            bits: h.bits,
            nonce: h.nonce,
        }
    }

    /// Reconstructs the full [`Header`] by reattaching `prev_blockhash`.
    pub fn to_header(&self, prev_blockhash: BlockHash) -> Header {
        Header {
            version: Version::from_consensus(self.version),
            prev_blockhash,
            merkle_root: self.merkle_root,
            time: self.time,
            bits: self.bits,
            nonce: self.nonce,
        }
    }
}

/// Returned by [`HeadersSyncState::process_presync`] and [`HeadersSyncState::process_redownload`] after each batch.
#[derive(Debug, Default)]
pub struct ProcessingResult {
    /// Headers that passed commitment verification, ready for `accept_header`.
    pub headers_to_accept: Vec<Header>,
    /// Whether the caller should send the next `getheaders` to this peer.
    pub request_more: bool,
    /// `false` if the peer sent invalid data (bad PoW, broken continuity, or commitment
    /// mismatch). Caller should disconnect and ban.
    pub success: bool,
}

/// Per-peer state machine for the two-phase headers pre-synchronization protocol.
///
/// See the [module documentation](self) for a full description of the algorithm.
#[derive(Debug, Clone)]
pub struct HeadersSyncState {
    state: PresyncPhase,

    /// Height of the known block from which this sync branches.
    pub start_height: u32,

    /// Hash of the known block from which this sync branches.
    pub start_hash: BlockHash,

    /// `bits` of the known block from which this sync branches. Seeds the difficulty
    /// transition check for the first header of each phase.
    start_bits: CompactTarget,

    /// Consensus parameters used for the difficulty transition check.
    consensus_params: Params,

    /// Minimum cumulative work the peer's chain must demonstrate before PRESYNC advances.
    /// Mirrors Bitcoin Core's `nMinimumChainWork` per network.
    minimum_chain_work: Work,

    /// Per-network algorithm parameters.
    params: HeadersSyncParams,

    // ── PRESYNC ───────────────────────────────────────────────────────────────
    /// Cumulative chainwork accumulated during PRESYNC.
    presync_work: Work,

    /// Height of the last header received during PRESYNC.
    presync_height: u32,

    /// Last header received during PRESYNC, kept for continuity checks and locator building.
    presync_last_header: Option<Header>,

    /// Per-session random salt for the commitment hash function.
    /// Never transmitted; prevents an attacker from predicting commitment slots.
    commitment_salt: u64,

    /// Secret offset within each period: a commitment is stored at height `h`
    /// when `h as u64 % params.commitment_period == commit_offset`.
    commit_offset: u64,

    /// Upper bound on stored commitment bits. Hitting this cap is not misbehavior;
    /// it means the peer's chain is longer than our memory budget allows.
    max_commitments: u64,

    /// One-bit salted commitments recorded during PRESYNC.
    /// Front = lowest commitment height above `start_height`.
    /// Consumed front-to-back during REDOWNLOAD.
    commitments: VecDeque<bool>,

    // ── REDOWNLOAD ────────────────────────────────────────────────────────────
    /// Set when re-accumulated chainwork during REDOWNLOAD meets `minimum_chain_work`.
    /// Once set, commitment checking stops and the buffer is fully drained.
    process_all_remaining: bool,

    /// Cumulative chainwork re-accumulated during REDOWNLOAD.
    redownload_chain_work: Work,

    /// Headers received during REDOWNLOAD, not yet released to permanent storage.
    /// Released from the front once the buffer exceeds `params.redownload_buffer_size`,
    /// or all at once when `process_all_remaining` is set.
    redownload_buffer: VecDeque<CompressedHeader>,

    /// Height of the last header appended to `redownload_buffer`.
    redownload_last_height: u32,

    /// Hash of the last header appended to `redownload_buffer`.
    /// Used to validate continuity of the next incoming header.
    redownload_last_hash: BlockHash,

    /// `prev_blockhash` of the current front of `redownload_buffer`.
    /// Needed to reconstruct full headers when draining from the front.
    redownload_first_prev_hash: BlockHash,
}

impl HeadersSyncState {
    /// Create a new `HeadersSyncState` starting from the given chain position.
    ///
    /// `start_height` / `start_hash` / `start_bits` identify the last block we already
    /// know; the peer will send headers building on top of it.
    pub fn new(
        start_height: u32,
        start_hash: BlockHash,
        start_bits: CompactTarget,
        network: Network,
    ) -> Self {
        let params = HeadersSyncParams::for_network(network);
        let salt: u64 = rand::random();
        let commit_offset = salt % params.commitment_period;
        let minimum_chain_work = minimum_chain_work(network);

        Self {
            state: PresyncPhase::Presync,
            start_height,
            start_hash,
            start_bits,
            consensus_params: Params::from(network),
            minimum_chain_work,
            params,
            presync_work: Work::from_be_bytes([0u8; 32]),
            presync_height: start_height,
            presync_last_header: None,
            commitment_salt: salt,
            commit_offset,
            max_commitments: MAX_COMMITMENTS,
            commitments: VecDeque::new(),
            process_all_remaining: false,
            redownload_chain_work: Work::from_be_bytes([0u8; 32]),
            redownload_buffer: VecDeque::new(),
            redownload_last_height: start_height,
            redownload_last_hash: start_hash,
            redownload_first_prev_hash: start_hash,
        }
    }

    /// Returns the current phase.
    pub fn phase(&self) -> &PresyncPhase {
        &self.state
    }

    /// SHA-256(salt || block_hash) LSB. The salt makes commitment slots unpredictable to the peer.
    fn commitment_bit(&self, hash: &BlockHash) -> bool {
        let mut input = [0u8; COMMITMENT_INPUT_LEN];
        input[..COMMITMENT_SALT_LEN].copy_from_slice(&self.commitment_salt.to_le_bytes());
        input[COMMITMENT_SALT_LEN..].copy_from_slice(hash.as_byte_array());
        let digest = sha256::Hash::hash(&input);
        (digest.as_byte_array()[0] & 1) == 1
    }

    /// Returns `true` if `height` is a commitment slot for this session.
    fn is_commitment_height(&self, height: u32) -> bool {
        (height as u64 % self.params.commitment_period) == self.commit_offset
    }

    /// Process a batch of headers during PRESYNC.
    ///
    /// Headers are validated for PoW and chain continuity but never written to
    /// disk. Commitment bits are recorded at every `params.commitment_period` headers.
    /// Once the chain accumulates sufficient work and the tip is recent the state
    /// transitions to REDOWNLOAD.
    pub fn process_presync(&mut self, headers: &[Header]) -> ProcessingResult {
        if self.state != PresyncPhase::Presync {
            return ProcessingResult {
                success: false,
                ..Default::default()
            };
        }

        for header in headers {
            let expected_prev = self
                .presync_last_header
                .as_ref()
                .map(|h| h.block_hash())
                .unwrap_or(self.start_hash);

            if header.prev_blockhash != expected_prev {
                return ProcessingResult {
                    success: false,
                    ..Default::default()
                };
            }

            // Bound how fast the claimed difficulty may move before trusting the
            // header's work. The exact retarget value needs epoch timestamps we do
            // not have yet, but the permitted transition range does not.
            let prev_bits = self
                .presync_last_header
                .as_ref()
                .map(|h| h.bits)
                .unwrap_or(self.start_bits);
            let next_height = self.presync_height + 1;
            if !permitted_difficulty_transition(
                &self.consensus_params,
                next_height,
                prev_bits,
                header.bits,
            ) {
                return ProcessingResult {
                    success: false,
                    ..Default::default()
                };
            }

            // The hash must satisfy the target the header now legitimately claims.
            if header.validate_pow(header.target()).is_err() {
                return ProcessingResult {
                    success: false,
                    ..Default::default()
                };
            }

            self.presync_work = self.presync_work + header.work();
            self.presync_height = next_height;

            if self.is_commitment_height(self.presync_height) {
                if self.commitments.len() as u64 >= self.max_commitments {
                    // Chain exceeds our memory budget; not misbehavior, just give up.
                    warn!(
                        "Peer chain exceeded presync commitment cap at height {}; aborting presync",
                        self.presync_height
                    );
                    self.state = PresyncPhase::Aborted;
                    return ProcessingResult {
                        success: true,
                        request_more: false,
                        headers_to_accept: vec![],
                    };
                }
                let bit = self.commitment_bit(&header.block_hash());
                self.commitments.push_back(bit);
            }

            self.presync_last_header = Some(*header);
        }

        // Transition to REDOWNLOAD on work alone, matching Core's headerssync.cpp.
        if self.presync_work >= self.minimum_chain_work && self.presync_last_header.is_some() {
            self.state = PresyncPhase::Redownload;
            return ProcessingResult {
                request_more: false,
                success: true,
                headers_to_accept: vec![],
            };
        }

        ProcessingResult {
            request_more: true,
            success: true,
            headers_to_accept: vec![],
        }
    }

    /// Process a batch of headers during REDOWNLOAD.
    ///
    /// Each header is validated against the commitment bits recorded during PRESYNC.
    /// Headers are released from the front of the buffer once it exceeds
    /// `params.redownload_buffer_size`. Once sufficient chainwork is re-accumulated,
    /// commitment checking stops and all remaining buffered headers are flushed.
    pub fn process_redownload(&mut self, headers: &[Header]) -> ProcessingResult {
        if self.state != PresyncPhase::Redownload {
            return ProcessingResult {
                success: false,
                ..Default::default()
            };
        }

        for header in headers {
            if header.prev_blockhash != self.redownload_last_hash {
                return ProcessingResult {
                    success: false,
                    ..Default::default()
                };
            }

            // Same difficulty transition bound as PRESYNC. The buffer only ever
            // drains partially mid-sync (a full drain ends the phase), so its back
            // is the previous header whenever it is non-empty.
            let prev_bits = self
                .redownload_buffer
                .back()
                .map(|c| c.bits)
                .unwrap_or(self.start_bits);
            let next_height = self.redownload_last_height + 1;
            if !permitted_difficulty_transition(
                &self.consensus_params,
                next_height,
                prev_bits,
                header.bits,
            ) {
                return ProcessingResult {
                    success: false,
                    ..Default::default()
                };
            }

            if header.validate_pow(header.target()).is_err() {
                return ProcessingResult {
                    success: false,
                    ..Default::default()
                };
            }

            self.redownload_last_height = next_height;
            self.redownload_last_hash = header.block_hash();

            self.redownload_chain_work = self.redownload_chain_work + header.work();
            if self.redownload_chain_work >= self.minimum_chain_work {
                self.process_all_remaining = true;
            }

            // Commitment verification stops once sufficient chainwork is re-accumulated.
            if !self.process_all_remaining && self.is_commitment_height(self.redownload_last_height)
            {
                let expected = match self.commitments.pop_front() {
                    Some(bit) => bit,
                    None => {
                        return ProcessingResult {
                            success: false,
                            ..Default::default()
                        };
                    }
                };
                if self.commitment_bit(&header.block_hash()) != expected {
                    return ProcessingResult {
                        success: false,
                        ..Default::default()
                    };
                }
            }

            self.redownload_buffer
                .push_back(CompressedHeader::from_header(header));
        }

        // Release headers from the front while the buffer exceeds the lookahead size,
        // or drain everything once sufficient work is re-accumulated.
        let mut headers_to_accept = Vec::new();
        let mut prev_hash = self.redownload_first_prev_hash;

        while self.redownload_buffer.len() > self.params.redownload_buffer_size
            || (!self.redownload_buffer.is_empty() && self.process_all_remaining)
        {
            let compressed = self.redownload_buffer.pop_front().unwrap();
            let header = compressed.to_header(prev_hash);
            prev_hash = header.block_hash();
            headers_to_accept.push(header);
        }

        self.redownload_first_prev_hash = prev_hash;

        let done = self.process_all_remaining && self.redownload_buffer.is_empty();
        if done {
            self.state = PresyncPhase::Final;
        }

        ProcessingResult {
            headers_to_accept,
            request_more: !done,
            success: true,
        }
    }

    /// Flips the first stored commitment bit. Only available in tests.
    #[cfg(test)]
    fn flip_first_commitment(&mut self) {
        if let Some(bit) = self.commitments.front_mut() {
            *bit = !*bit;
        }
    }

    /// Overrides the commitment cap. Only available in tests.
    #[cfg(test)]
    fn with_max_commitments(mut self, cap: u64) -> Self {
        self.max_commitments = cap;
        self
    }

    /// Overrides the minimum chain work threshold. Only available in tests.
    #[cfg(test)]
    fn with_minimum_chain_work(mut self, work: Work) -> Self {
        self.minimum_chain_work = work;
        self
    }

    // Last-seen hash from in-memory state; never touches the chainstore.
    // Matches Core's NextHeadersRequestLocator() (headerssync.cpp:296-317).
    pub fn next_locator_hash(&self) -> Option<BlockHash> {
        match self.state {
            PresyncPhase::Presync => self.presync_last_header.as_ref().map(|h| h.block_hash()),
            PresyncPhase::Redownload => Some(self.redownload_last_hash),
            PresyncPhase::Final | PresyncPhase::Aborted => None,
        }
    }

    /// Overrides the REDOWNLOAD buffer size. Only available in tests.
    #[cfg(test)]
    fn with_redownload_buffer_size(mut self, size: usize) -> Self {
        self.params.redownload_buffer_size = size;
        self
    }

    /// Loads commitments from `headers` and enters REDOWNLOAD state directly.
    ///
    /// Simulates the PRESYNC commitment-recording pass so that REDOWNLOAD tests
    /// can run without going through a full PRESYNC with real chainwork.
    #[cfg(test)]
    fn force_redownload(mut self, headers: &[Header]) -> Self {
        for (i, header) in headers.iter().enumerate() {
            let height = self.start_height + i as u32 + 1;
            if self.is_commitment_height(height) {
                self.commitments
                    .push_back(self.commitment_bit(&header.block_hash()));
            }
        }
        self.state = PresyncPhase::Redownload;
        self
    }
}

#[cfg(test)]
mod tests {
    use bitcoin::CompactTarget;
    use bitcoin::Network;
    use bitcoin::TxMerkleNode;
    use bitcoin::Work;
    use bitcoin::block::Header;
    use bitcoin::block::Version;

    use super::*;

    /// Regtest commitment period; chain lengths are multiples of this to guarantee exact slot counts.
    const REGTEST_PERIOD: usize = 275;
    /// Easy compact target: ~50% of nonces satisfy this, so headers mine in 1-2 tries on average.
    const EASY_BITS: u32 = 0x207f_ffff;
    /// Arbitrary far-past timestamp used to build test chains.
    const ANCIENT_TIMESTAMP: u32 = 1_000_000;

    /// `EASY_BITS` as a [`CompactTarget`]. Also used as the start block's bits so that
    /// off-boundary transition checks see an unchanged difficulty.
    fn easy_bits() -> CompactTarget {
        CompactTarget::from_consensus(EASY_BITS)
    }

    fn make_header(prev: BlockHash, time: u32) -> Header {
        make_header_with_bits(prev, time, easy_bits())
    }

    fn make_header_with_bits(prev: BlockHash, time: u32, bits: CompactTarget) -> Header {
        (0u32..=u32::MAX)
            .map(|nonce| Header {
                version: Version::from_consensus(1),
                prev_blockhash: prev,
                merkle_root: TxMerkleNode::all_zeros(),
                time,
                bits,
                nonce,
            })
            .find(|h| h.validate_pow(h.target()).is_ok())
            .expect("no valid nonce found for EASY_BITS target")
    }

    fn make_chain(prev: BlockHash, time: u32, count: usize) -> Vec<Header> {
        let mut out = Vec::with_capacity(count);
        let mut prev = prev;
        for i in 0..count {
            let h = make_header(prev, time + i as u32);
            prev = h.block_hash();
            out.push(h);
        }
        out
    }

    #[test]
    fn difficulty_transition_off_boundary_requires_unchanged_bits() {
        let params = Params::from(Network::Bitcoin);
        // A real historic mainnet target, well below the PoW limit.
        let old = CompactTarget::from_consensus(0x1b04_04cb);
        let other = CompactTarget::from_consensus(0x1b04_04cc);

        assert!(permitted_difficulty_transition(&params, 1, old, old));
        assert!(!permitted_difficulty_transition(&params, 1, old, other));
        assert!(!permitted_difficulty_transition(&params, 2015, old, other));
    }

    #[test]
    fn difficulty_transition_at_boundary_allows_at_most_4x() {
        let params = Params::from(Network::Bitcoin);
        let old = CompactTarget::from_consensus(0x1b04_04cb);
        let old_target = Target::from_compact(old);

        // Exactly 4x easier and 4x harder are the permitted extremes.
        let easiest = old_target
            .max_transition_threshold(&params)
            .to_compact_lossy();
        let hardest = old_target.min_transition_threshold().to_compact_lossy();
        assert!(permitted_difficulty_transition(&params, 2016, old, old));
        assert!(permitted_difficulty_transition(&params, 2016, old, easiest));
        assert!(permitted_difficulty_transition(&params, 2016, old, hardest));

        // Bumping the compact exponent moves the target by 256x: far outside the clamp.
        let way_easier = CompactTarget::from_consensus(0x1c04_04cb);
        let way_harder = CompactTarget::from_consensus(0x1a04_04cb);
        assert!(!permitted_difficulty_transition(
            &params, 2016, old, way_easier
        ));
        assert!(!permitted_difficulty_transition(
            &params, 2016, old, way_harder
        ));
    }

    #[test]
    fn difficulty_transition_is_unchecked_on_min_difficulty_networks() {
        let params = Params::from(Network::Regtest);
        let old = CompactTarget::from_consensus(0x1b04_04cb);
        let way_easier = CompactTarget::from_consensus(0x1c04_04cb);
        assert!(permitted_difficulty_transition(&params, 1, old, way_easier));
        assert!(permitted_difficulty_transition(
            &params, 2016, old, way_easier
        ));
    }

    #[test]
    fn presync_rejects_illegal_difficulty_jump() {
        let genesis = BlockHash::all_zeros();
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Bitcoin);

        // Header 2 changes `bits` off a retarget boundary: consensus-invalid on mainnet.
        let h1 = make_header(genesis, ANCIENT_TIMESTAMP);
        let h2 = make_header_with_bits(
            h1.block_hash(),
            ANCIENT_TIMESTAMP + 1,
            CompactTarget::from_consensus(0x207f_fffe),
        );

        let result = state.process_presync(&[h1, h2]);
        assert!(!result.success);
    }

    #[test]
    fn redownload_rejects_illegal_difficulty_jump() {
        let genesis = BlockHash::all_zeros();
        let huge_work = Work::from_be_bytes([0xff; 32]);
        let good = make_chain(genesis, ANCIENT_TIMESTAMP, 3);

        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Bitcoin)
            .with_minimum_chain_work(huge_work)
            .force_redownload(&good);

        // Replay with header 2's `bits` changed off-boundary; hash and PoW are re-mined so
        // only the transition check can reject it.
        let bad_h2 = make_header_with_bits(
            good[0].block_hash(),
            ANCIENT_TIMESTAMP + 1,
            CompactTarget::from_consensus(0x207f_fffe),
        );
        let result = state.process_redownload(&[good[0], bad_h2]);
        assert!(!result.success);
    }

    #[test]
    fn presync_transitions_to_redownload_on_sufficient_work() {
        let genesis = BlockHash::all_zeros();
        // On Regtest minimum_chain_work = 0, so any non-empty batch satisfies the gate.
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest);

        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, 5);
        let result = state.process_presync(&chain);

        assert!(result.success);
        assert!(!result.request_more);
        assert_eq!(*state.phase(), PresyncPhase::Redownload);
    }

    #[test]
    fn presync_rejects_broken_continuity() {
        let genesis = BlockHash::all_zeros();
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest);

        let mut chain = make_chain(genesis, ANCIENT_TIMESTAMP, 5);
        chain[2].prev_blockhash = BlockHash::all_zeros();

        let result = state.process_presync(&chain);
        assert!(!result.success);
    }

    #[test]
    fn redownload_rejects_broken_continuity() {
        let genesis = BlockHash::all_zeros();
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest);

        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, 5);
        let _ = state.process_presync(&chain);
        assert_eq!(*state.phase(), PresyncPhase::Redownload);

        let mut bad = make_chain(BlockHash::all_zeros(), ANCIENT_TIMESTAMP, 3);
        bad[0].prev_blockhash = chain[3].block_hash();

        let result = state.process_redownload(&bad);
        assert!(!result.success);
    }

    #[test]
    fn redownload_succeeds_with_matching_chain() {
        let genesis = BlockHash::all_zeros();
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest);

        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, REGTEST_PERIOD);
        let _ = state.process_presync(&chain);
        assert_eq!(*state.phase(), PresyncPhase::Redownload);

        // REDOWNLOAD: replay the same chain. On Regtest (min_chain_work = 0),
        // process_all_remaining is set immediately so all headers are released at once.
        let result = state.process_redownload(&chain);
        assert!(result.success);
        assert!(!result.headers_to_accept.is_empty());
    }

    #[test]
    fn redownload_releases_all_headers_and_reaches_final() {
        let genesis = BlockHash::all_zeros();
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest);

        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, REGTEST_PERIOD);
        let _ = state.process_presync(&chain);
        assert_eq!(*state.phase(), PresyncPhase::Redownload);

        let result = state.process_redownload(&chain);
        assert!(result.success);
        assert_eq!(result.headers_to_accept.len(), chain.len());
        assert_eq!(*state.phase(), PresyncPhase::Final);
    }

    #[test]
    fn presync_stays_in_presync_below_minimum_work() {
        let genesis = BlockHash::all_zeros();
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Bitcoin);

        // Easy-difficulty headers have negligible work vs mainnet nMinimumChainWork.
        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, 5);
        let result = state.process_presync(&chain);

        assert!(result.success);
        assert!(result.request_more);
        assert_eq!(*state.phase(), PresyncPhase::Presync);
    }

    #[test]
    fn presync_stays_in_presync_below_signet_minimum_work() {
        let genesis = BlockHash::all_zeros();
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Signet);

        // Easy-difficulty headers have negligible work vs signet's nMinimumChainWork.
        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, 5);
        let result = state.process_presync(&chain);

        assert!(result.success);
        assert!(result.request_more);
        assert_eq!(*state.phase(), PresyncPhase::Presync);
    }

    #[test]
    fn presync_aborts_gracefully_at_max_commitments() {
        let genesis = BlockHash::all_zeros();
        // Cap at 1 so the second commitment slot triggers the abort.
        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest)
            .with_max_commitments(1);

        // 2 * REGTEST_PERIOD headers → exactly 2 commitment slots.
        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, 2 * REGTEST_PERIOD);
        let result = state.process_presync(&chain);

        assert!(result.success);
        assert!(!result.request_more);
        assert!(result.headers_to_accept.is_empty());
        assert_eq!(*state.phase(), PresyncPhase::Aborted);
    }

    #[test]
    fn redownload_releases_headers_incrementally_via_buffer_size() {
        let genesis = BlockHash::all_zeros();
        // Use a huge minimum work so process_all_remaining stays false, exercising the
        // buffer-size drain path rather than the flush-all path.
        let huge_work = Work::from_be_bytes([0xff; 32]);
        // 4 * REGTEST_PERIOD headers → 4 commitment slots.
        let chain = make_chain(genesis, ANCIENT_TIMESTAMP, 4 * REGTEST_PERIOD);

        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest)
            .with_minimum_chain_work(huge_work)
            .with_redownload_buffer_size(275)
            .force_redownload(&chain);

        // Feed 2 * REGTEST_PERIOD headers: buffer grows then drains back to REGTEST_PERIOD.
        let result = state.process_redownload(&chain[..2 * REGTEST_PERIOD]);
        assert!(result.success);
        assert_eq!(result.headers_to_accept.len(), REGTEST_PERIOD);
        assert!(result.headers_to_accept.len() < 2 * REGTEST_PERIOD);
        assert_ne!(*state.phase(), PresyncPhase::Final);
    }

    #[test]
    fn redownload_rejects_commitment_mismatch() {
        let genesis = BlockHash::all_zeros();
        // Use huge minimum work so process_all_remaining stays false and commitment
        // checking remains active throughout REDOWNLOAD.
        let huge_work = Work::from_be_bytes([0xff; 32]);
        // REGTEST_PERIOD headers → exactly one commitment slot.
        let bulk = make_chain(genesis, ANCIENT_TIMESTAMP, REGTEST_PERIOD);

        let mut state = HeadersSyncState::new(0, genesis, easy_bits(), Network::Regtest)
            .with_minimum_chain_work(huge_work)
            .force_redownload(&bulk);

        // Flip the stored bit so the replayed chain fails the commitment check.
        state.flip_first_commitment();

        let result = state.process_redownload(&bulk);
        assert!(!result.success);
    }
}

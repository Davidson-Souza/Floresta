// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transaction relay types for BIP-183 (Utreexo Peer Services).
//!
//! This module defines the data structures and inventory type constants used when
//! relaying transactions between Utreexo-aware peers.
//!
//! # Transaction relay overview
//!
//! Non-Utreexo transaction relay requires two round trips: an `inv` message followed by a
//! `getdata`/`tx` exchange. Utreexo transaction relay keeps the same round-trip count by
//! embedding merkle-forest position hints inside extra [`MSG_UTREEXO_PROOF_HASH`] inventory
//! vectors that are *appended* to the standard `MSG_TX` entry in the `inv` message.
//!
//! When the receiver needs the full transaction with its inclusion proof, it issues a
//! `getdata` using [`MSG_UTREEXO_TX`] (or [`MSG_WITNESS_UTREEXO_TX`] for segwit). The
//! responder then replies with a [`UtreexoTx`] message (P2Pv1 command string: `utreexotx`,
//! BIP-324 type: `34`).
//!
//! # Unconfirmed inputs
//!
//! An input may reference a UTXO that is not yet confirmed (i.e., it is itself in the
//! mempool). Such inputs cannot be proved against the accumulator and have no corresponding
//! leaf data. To distinguish them, the outpoint index stored in the serialized transaction is
//! encoded as follows:
//!
//! ```text
//! encoded_vout = vout << 1
//! if is_unconfirmed { encoded_vout |= 1 }
//! ```
//!
//! The [`UtreexoTx`] struct stores the transaction with these encoded vout values so that the
//! wire representation is a plain `MSG_TX` encoding and no extra per-input flags vector is
//! needed.

use bitcoin::Transaction;
use bitcoin::Txid;
use bitcoin::VarInt;
use bitcoin::consensus::Decodable;
use floresta_chain::CompactLeafData;
use floresta_chain::ScriptPubKeyKind;
use floresta_common::read_bounded_len;
use rustreexo::node_hash::BitcoinNodeHash;

/// Inventory type for Utreexo merkle-forest positions.
///
/// An inventory vector with this type carries up to four little-endian `u64` merkle-tree
/// positions packed into the 32-byte hash field (unused slots are padded with `u64::MAX`).
/// It **must** be appended immediately after an inventory vector of type `MSG_TX`,
/// `MSG_WITNESS_TX`, [`MSG_UTREEXO_TX`], or [`MSG_WITNESS_UTREEXO_TX`].
///
/// See BIP-183 § New Inventory Types.
pub const MSG_UTREEXO_PROOF_HASH: u32 = 6;

/// Inventory type used in `getdata` to request a [`UtreexoTx`] without witness data.
///
/// Defined as `MSG_UTREEXO_FLAG | 1` = `(1 << 24) | 1` = `16777217`.
///
/// See BIP-183 § MSG_UTREEXO_TX.
pub const MSG_UTREEXO_TX: u32 = 1 << 24 | 1;

/// Inventory type used in `getdata` to request a witness [`UtreexoTx`].
///
/// Defined as `(1 << 30) | (1 << 24) | 1` = `1090519041`.
///
/// See BIP-183 § MSG_WITNESS_UTREEXO_TX.
pub const MSG_WITNESS_UTREEXO_TX: u32 = 1 << 30 | 1 << 24 | 1;

/// Inventory type used in `getdata` to request a utreexo block summary.
///
/// Defined as `7`.
///
/// See BIP-183 § MSG_UTREEXO_SUMMARY.
pub const MSG_UTREEXO_SUMMARY: u32 = 7;

/// Flag bit that, when OR'd with `MSG_TX` or `MSG_WITNESS_TX`, signals that a Utreexo
/// transaction is desired.
///
/// Defined as `1 << 24`.
///
/// See BIP-183 § MSG_UTREEXO_FLAG.
pub const MSG_UTREEXO_FLAG: u32 = 1 << 24;

/// Maximum number of inputs a single transaction may have on the wire.
///
/// The smallest input is ~41 bytes (outpoint 36 + script_len 1 + sequence 4), so the
/// absolute limit in a standard-weight transaction is well below this value. We use a
/// conservative upper bound to guard against memory-exhaustion attacks during
/// deserialisation.
const MAX_TX_INPUTS: usize = 100_000;

/// How deep the Utreexo forest can be.
const MAX_TREE_DEPTH: usize = 64;

/// Maximum number of proof hashes for a single transaction.
///
/// Each confirmed input may need at most [`MAX_TREE_DEPTH`] hashes to prove its membership
/// in the accumulator. In the pathological case where all inputs are confirmed and no proofs
/// overlap, this is the upper bound.
const MAX_PROOF_HASHES: usize = MAX_TX_INPUTS * MAX_TREE_DEPTH;

/// P2Pv1 command string for the `MSG_UTREEXO_TX` message.
///
/// See BIP-183 § MSG_UTREEXO_TX.
pub const UTREEXO_TX_CMD_STRING: &str = "utreexotx";

/// The sentinel value used to pad unused slots in a [`MSG_UTREEXO_PROOF_HASH`] inventory vector.
///
/// Each 32-byte hash field packs up to four little-endian `u64` merkle-tree positions. Slots
/// that are not needed are filled with this value (`u64::MAX` = `0xFFFFFFFFFFFFFFFF`).
pub const UTREEXO_PROOF_HASH_PADDING: u64 = u64::MAX;

/// The number of `u64` positions that fit in a single [`MSG_UTREEXO_PROOF_HASH`] hash field.
const POSITIONS_PER_PROOF_HASH: usize = 4;

const BYTES_PER_U64: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
/// A transaction inventory announcement with its Utreexo merkle-forest positions.
///
/// When a Utreexo peer announces a transaction via `inv`, it appends one or more
/// [`MSG_UTREEXO_PROOF_HASH`] inventory vectors immediately after the transaction inventory
/// vector. Each such entry packs up to four little-endian `u64` merkle-tree positions into the
/// standard 32-byte hash field; unused slots are padded with [`UTREEXO_PROOF_HASH_PADDING`].
///
/// This struct collects everything that belongs to a single transaction announcement.
pub struct UtreexoTxInv {
    /// The transaction identifier being announced.
    pub txid: Txid,

    /// Flattened Utreexo merkle-forest positions for the transaction's
    /// confirmed inputs.
    pub positions: Vec<u64>,
}

/// Parse the four packed little-endian `u64` positions from a [`MSG_UTREEXO_PROOF_HASH`]
/// inventory hash field, discarding sentinel padding entries.
///
/// A 32-byte hash packs exactly four `u64` values in little-endian byte order. Any slot
/// equal to [`UTREEXO_PROOF_HASH_PADDING`] (`u64::MAX`) is padding and is excluded from
/// the returned iterator.
///
/// # Example
///
/// ```rust
/// use floresta_wire::tx_relay::parse_utreexo_proof_hash;
/// use floresta_wire::tx_relay::UTREEXO_PROOF_HASH_PADDING;
///
/// // Pack positions 0, 42, and 1000 into a 32-byte hash (fourth slot padded).
/// let mut hash = [0u8; 32];
/// hash[0..8].copy_from_slice(&0u64.to_le_bytes());
/// hash[8..16].copy_from_slice(&42u64.to_le_bytes());
/// hash[16..24].copy_from_slice(&1000u64.to_le_bytes());
/// hash[24..32].copy_from_slice(&UTREEXO_PROOF_HASH_PADDING.to_le_bytes());
///
/// let positions: Vec<u64> = parse_utreexo_proof_hash(&hash).collect();
/// assert_eq!(positions, vec![0, 42, 1000]);
/// ```
pub fn parse_utreexo_proof_hash(hash: &[u8; 32]) -> impl Iterator<Item = u64> + '_ {
    (0..POSITIONS_PER_PROOF_HASH).filter_map(move |i| {
        let offset = i * BYTES_PER_U64;
        let pos = u64::from_le_bytes(
            hash[offset..offset + BYTES_PER_U64]
                .try_into()
                .expect("slice is 8 bytes"),
        );
        if pos == UTREEXO_PROOF_HASH_PADDING {
            None
        } else {
            Some(pos)
        }
    })
}

/// Pack Utreexo merkle-forest positions into inventory hash fields.
///
/// The final field is padded with [`UTREEXO_PROOF_HASH_PADDING`]. An empty
/// position list produces no fields.
pub(crate) fn pack_utreexo_proof_hashes(positions: &[u64]) -> impl Iterator<Item = [u8; 32]> + '_ {
    positions.chunks(POSITIONS_PER_PROOF_HASH).map(|positions| {
        let mut hash = [u8::MAX; 32];

        for (index, position) in positions.iter().enumerate() {
            let offset = index * BYTES_PER_U64;
            hash[offset..offset + BYTES_PER_U64].copy_from_slice(&position.to_le_bytes());
        }

        hash
    })
}

#[derive(Debug, Clone)]
/// A Bitcoin transaction bundled with its Utreexo inclusion proof.
///
/// This is the payload of the `utreexotx` P2P message (BIP-183, type `34`).
///
/// # Outpoint index encoding
///
/// Each input's outpoint `vout` is encoded as:
///
/// ```text
/// encoded_vout = vout << 1
/// if is_unconfirmed { encoded_vout |= 1 }
/// ```
///
/// The decoder restores each input's original outpoint index. The unconfirmed
/// marker is only used while decoding the compact leaf data and is not retained.
pub struct UtreexoTx {
    /// The Bitcoin transaction with original outpoint indices restored.
    pub tx: Transaction,

    /// The Utreexo merkle proof targets for the confirmed inputs.
    pub targets: Vec<u64>,

    /// The Utreexo merkle proof hashes for the confirmed inputs.
    pub hashes: Vec<BitcoinNodeHash>,

    /// The compact leaf data for confirmed inputs, in the same order as confirmed inputs
    /// appear in `tx.input`. Unconfirmed inputs do not have a corresponding entry here.
    pub leaf_data: Vec<CompactLeafData>,
}

impl Decodable for UtreexoTx {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, bitcoin::consensus::encode::Error> {
        // Decode the proof
        let n_targets = read_bounded_len(reader, MAX_TX_INPUTS)?;
        let mut targets = Vec::with_capacity(n_targets);

        for _ in 0..n_targets {
            let target = VarInt::consensus_decode(reader)?;
            targets.push(target.0);
        }

        let n_hashes = read_bounded_len(reader, MAX_PROOF_HASHES)?;
        let mut hashes = Vec::with_capacity(n_hashes);

        for _ in 0..n_hashes {
            let hash = <[u8; 32]>::consensus_decode(reader)?;
            hashes.push(BitcoinNodeHash::Some(hash));
        }

        // Decode the transaction before its compact leaf data.
        let mut tx = Transaction::consensus_decode(reader)?;

        // Unconfirmed inputs are marked in the wire transaction and have no
        // corresponding compact leaf data.
        let n_leaf_data = tx
            .input
            .iter()
            .filter(|input| input.previous_output.vout & 1 == 0)
            .count();
        let mut leaf_data = Vec::with_capacity(n_leaf_data);
        for _ in 0..n_leaf_data {
            let leaf = CompactLeafData {
                header_code: u32::consensus_decode(reader)?,
                amount: u64::consensus_decode(reader)?,
                spk_ty: ScriptPubKeyKind::consensus_decode(reader)?,
            };
            leaf_data.push(leaf);
        }

        for input in &mut tx.input {
            input.previous_output.vout >>= 1;
        }

        Ok(Self {
            tx,
            hashes,
            targets,
            leaf_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use bitcoin::consensus::encode::deserialize;
    use bitcoin::consensus::encode::deserialize_hex;

    use super::MSG_UTREEXO_FLAG;
    use super::MSG_UTREEXO_PROOF_HASH;
    use super::MSG_UTREEXO_SUMMARY;
    use super::MSG_UTREEXO_TX;
    use super::MSG_WITNESS_UTREEXO_TX;
    use super::UtreexoTx;
    use super::pack_utreexo_proof_hashes;
    use super::parse_utreexo_proof_hash;

    #[test]
    fn utreexo_inventory_constants_match_bip183() {
        assert_eq!(MSG_UTREEXO_PROOF_HASH, 6);
        assert_eq!(MSG_UTREEXO_SUMMARY, 7);
        assert_eq!(MSG_UTREEXO_FLAG, 1 << 24);
        assert_eq!(MSG_UTREEXO_TX, MSG_UTREEXO_FLAG | 1);
        assert_eq!(MSG_WITNESS_UTREEXO_TX, MSG_UTREEXO_TX | (1 << 30));
    }

    #[test]
    fn proof_hash_positions_round_trip_without_dropping_positions() {
        let positions = [0, 1, 2, 3, 4, u64::MAX - 1];
        let hashes: Vec<_> = pack_utreexo_proof_hashes(&positions).collect();
        let decoded: Vec<_> = hashes.iter().flat_map(parse_utreexo_proof_hash).collect();

        assert_eq!(hashes.len(), 2);
        assert_eq!(decoded, positions);
    }

    #[test]
    fn empty_proof_positions_produce_no_inventory_fields() {
        assert!(pack_utreexo_proof_hashes(&[]).next().is_none());
    }

    #[test]
    fn truncated_utreexo_transaction_fails_to_decode() {
        assert!(deserialize::<UtreexoTx>(&[]).is_err());
    }

    #[test]
    fn decodes_utreexod_transaction_with_original_outpoint_index() {
        const UTXO_TX: &str = concat!(
            "01fe6c3fb2070002000000000101240316ddb00c69727722989ea4aec17ea37e",
            "77cc6f8c70ed08b4e398fa89f8d00200000000fdffffff02c057010000000000",
            "225120c9fcc4a474637d19b1d3ac424f78e14952d277f666c09f1ca52079151",
            "3523a2a6aa2990000000000225120de0522bc312dee3977129d9a2acae1284f",
            "b45c920e31dbb569922f4011376a6801403638eedbeb5297721ec76aecea1f2",
            "13a49206eee6ae79e3b33095f73696c191f4860f6f394f21d0b4ff295a49050",
            "803f5611c19048ea6473bfb36ebbdb283a7d0000000038300900c4fa9a000000",
            "000000225120de0522bc312dee3977129d9a2acae1284fb45c920e31dbb5699",
            "22f4011376a68"
        );

        let tx: UtreexoTx = deserialize_hex(UTXO_TX).expect("valid utreexod transaction");

        assert_eq!(tx.leaf_data.len(), 1);
        assert_eq!(tx.tx.input[0].previous_output.vout, 1);
    }
}

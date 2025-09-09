//! UData is the serialized data used for proof propagation in utreexo. It contains all
//! data needed for validating some piece of information, like a transaction and a block.

use bitcoin::consensus;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::Encodable;
use bitcoin::hashes::sha256;
use bitcoin::hashes::Hash;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::OutPoint;
use bitcoin::TxOut;
use bitcoin::VarInt;
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use rustreexo::accumulator::proof::Proof;
use sha2::Digest;
use sha2::Sha512_256;

use crate::prelude::vec;
use crate::prelude::Box;
use crate::prelude::Vec;
use crate::pruned_utreexo::consensus::UTREEXO_TAG_V1;

/// Leaf data is the data that is hashed when adding to utreexo state. It contains validation
/// data and some commitments to make it harder to attack an utreexo-only node.
#[derive(Debug, PartialEq)]
pub struct LeafData {
    /// A commitment to the block creating this utxo
    pub block_hash: BlockHash,
    /// The utxo's outpoint
    pub prevout: OutPoint,
    /// Header code is a compact commitment to the block height and whether or not this
    /// transaction is coinbase. It's defined as
    ///
    /// ```ignore
    /// header_code: u32 = if transaction.is_coinbase() {
    ///     (block_height << 1 ) | 1
    /// } else {
    ///     block_height << 1
    /// };
    /// ```
    pub header_code: u32,
    /// The actual utxo
    pub utxo: TxOut,
}

impl LeafData {
    pub fn _get_leaf_hashes(&self) -> sha256::Hash {
        let mut ser_utxo = Vec::new();
        self.utxo
            .consensus_encode(&mut ser_utxo)
            .expect("serializing TxOut never fails: Vec<u8>::Write always returns Ok");

        let leaf_hash = Sha512_256::new()
            .chain_update(UTREEXO_TAG_V1)
            .chain_update(UTREEXO_TAG_V1)
            .chain_update(self.block_hash)
            .chain_update(self.prevout.txid)
            .chain_update(self.prevout.vout.to_le_bytes())
            .chain_update(self.header_code.to_le_bytes())
            .chain_update(ser_utxo)
            .finalize();

        sha256::Hash::from_byte_array(leaf_hash.into())
    }
}

impl Decodable for LeafData {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, consensus::encode::Error> {
        Self::consensus_decode_from_finite_reader(reader)
    }
    fn consensus_decode_from_finite_reader<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, consensus::encode::Error> {
        let block_hash = BlockHash::consensus_decode(reader)?;
        let prevout = OutPoint::consensus_decode(reader)?;
        let header_code = u32::consensus_decode(reader)?;
        let utxo = TxOut::consensus_decode(reader)?;
        Ok(LeafData {
            block_hash,
            prevout,
            header_code,
            utxo,
        })
    }
}

/// Commitment of the leaf data, but in a compact way
///
/// The serialized format is:
/// `[<header_code><amount><spk_type>]`
///
/// The serialized header code format is:
///   bit 0 - containing transaction is a coinbase
///   bits 1-x - height of the block that contains the spent txout
///
/// It's calculated with:
///   header_code = <<= 1
///   if IsCoinBase {
///       header_code |= 1 // only set the bit 0 if it's a coinbase.
///   }
/// ScriptPubKeyKind is the output's scriptPubKey, but serialized in a more efficient way
/// to save bandwidth. If the type is recoverable from the scriptSig, don't download the
/// scriptPubKey.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct CompactLeafData {
    /// Header code tells the height of creating for this UTXO and whether it's a coinbase
    pub header_code: u32,
    /// The amount locked in this UTXO
    pub amount: u64,
    /// The type of the locking script for this UTXO
    pub spk_ty: ScriptPubKeyKind,
}

/// A recoverable scriptPubKey type, this avoids copying over data that are already
/// present or can be computed from the transaction itself.
/// An example is a p2pkh, the public key is serialized in the scriptSig, so we can just
/// grab it and hash to obtain the actual scriptPubKey. Since this data is committed in
/// the Utreexo leaf hash, it is still authenticated
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum ScriptPubKeyKind {
    /// An non-specified type, in this case the script is just copied over
    Other(Box<[u8]>),
    /// p2pkh
    PubKeyHash,
    /// p2wsh
    WitnessV0PubKeyHash,
    /// p2sh
    ScriptHash,
    /// p2wsh
    WitnessV0ScriptHash,
}

impl Decodable for ScriptPubKeyKind {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, consensus::encode::Error> {
        let ty = u8::consensus_decode(reader)?;
        match ty {
            0x00 => Ok(ScriptPubKeyKind::Other(Box::consensus_decode(reader)?)),
            0x01 => Ok(ScriptPubKeyKind::PubKeyHash),
            0x02 => Ok(ScriptPubKeyKind::WitnessV0PubKeyHash),
            0x03 => Ok(ScriptPubKeyKind::ScriptHash),
            0x04 => Ok(ScriptPubKeyKind::WitnessV0ScriptHash),
            _ => Err(consensus::encode::Error::ParseFailed("Invalid script type")),
        }
    }
}

impl Encodable for ScriptPubKeyKind {
    fn consensus_encode<W: bitcoin::io::Write + ?Sized>(
        &self,
        writer: &mut W,
    ) -> Result<usize, bitcoin::io::Error> {
        let mut len = 1;

        match self {
            ScriptPubKeyKind::Other(script) => {
                00_u8.consensus_encode(writer)?;
                len += script.consensus_encode(writer)?;
            }
            ScriptPubKeyKind::PubKeyHash => {
                0x01_u8.consensus_encode(writer)?;
            }
            ScriptPubKeyKind::WitnessV0PubKeyHash => {
                0x02_u8.consensus_encode(writer)?;
            }
            ScriptPubKeyKind::ScriptHash => {
                0x03_u8.consensus_encode(writer)?;
            }
            ScriptPubKeyKind::WitnessV0ScriptHash => {
                0x04_u8.consensus_encode(writer)?;
            }
        }
        Ok(len)
    }
}

/// BatchProof serialization defines how the utreexo accumulator proof will be
/// serialized both for i/o.
///
/// Note that this serialization format differs from the one from
/// github.com/mit-dci/utreexo/accumulator as this serialization method uses
/// varints and the one in that package does not.  They are not compatible and
/// should not be used together.  The serialization method here is more compact
/// and thus is better for wire and disk storage.
///
/// The serialized format is:
/// `[<target count><targets><proof count><proofs>]`
///
/// All together, the serialization looks like so:
/// Field          Type       Size
/// target count   varint     1-8 bytes
/// targets        []uint64   variable
/// hash count     varint     1-8 bytes
/// hashes         []32 byte  variable
#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct BatchProof {
    /// All targets that'll be deleted
    pub targets: Vec<VarInt>,
    /// The inner hashes of a proof
    pub hashes: Vec<BlockHash>,
}

impl From<&BatchProof> for Proof {
    fn from(batch_proof: &BatchProof) -> Self {
        let targets = batch_proof.targets.iter().map(|target| target.0).collect();
        let proof_hashes = batch_proof
            .hashes
            .iter()
            .map(|hash| BitcoinNodeHash::Some(*hash.as_byte_array()))
            .collect();

        Proof::new(targets, proof_hashes)
    }
}

/// UData contains data needed to prove the existence and validity of all inputs
/// for a Bitcoin block.  With this data, a full node may only keep the utreexo
/// roots and still be able to fully validate a block.
#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct UData {
    /// All the indexes of new utxos to remember.
    pub remember_idx: Vec<u64>,
    /// AccProof is the utreexo accumulator proof for all the inputs.
    pub proof: BatchProof,
    /// LeafData are the tx validation data for every input.
    pub leaves: Vec<CompactLeafData>,
}

impl Decodable for UData {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, consensus::encode::Error> {
        let n_positions = VarInt::consensus_decode(reader)?;
        let mut targets = vec![];
        for _ in 0..n_positions.0 {
            let pos = VarInt::consensus_decode(reader)?;
            targets.push(pos);
        }

        let n_hashes = VarInt::consensus_decode(reader)?;
        let mut hashes = vec![];
        for _ in 0..n_hashes.0 {
            let hash = BlockHash::consensus_decode(reader)?;
            hashes.push(hash);
        }

        let n_leaves = VarInt::consensus_decode(reader)?;
        let mut leaves = vec![];
        for _ in 0..n_leaves.0 {
            let header_code = u32::consensus_decode(reader)?;
            let amount = u64::consensus_decode(reader)?;
            let spk_ty = ScriptPubKeyKind::consensus_decode(reader)?;

            leaves.push(CompactLeafData {
                header_code,
                amount,
                spk_ty,
            });
        }

        Ok(UData {
            remember_idx: vec![],
            proof: BatchProof { targets, hashes },
            leaves,
        })
    }
}

/// A block plus some udata
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct UtreexoBlock {
    /// A actual block
    pub block: Block,
    /// The utreexo specific data
    pub udata: Option<UData>,
}

impl Decodable for UtreexoBlock {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, consensus::encode::Error> {
        let block = Block::consensus_decode(reader)?;

        if let Err(consensus::encode::Error::Io(_remember)) = VarInt::consensus_decode(reader) {
            return Ok(block.into());
        };

        let udata = UData::consensus_decode(reader)?;
        Ok(Self {
            block,
            udata: Some(udata),
        })
    }
}

impl From<UtreexoBlock> for Block {
    fn from(block: UtreexoBlock) -> Self {
        block.block
    }
}

impl From<Block> for UtreexoBlock {
    fn from(block: Block) -> Self {
        UtreexoBlock { block, udata: None }
    }
}

/// This module provides utility functions for working with Utreexo proofs.
///
/// These functions can be used, for example, when verifying if a mempool transaction is valid;
/// to consume a block (delete transactions included in it from the mempool);
/// or to validate a block.
pub mod proof_util {
    use bitcoin::blockdata::script;
    use bitcoin::blockdata::script::Instruction;
    use bitcoin::consensus::Encodable;
    use bitcoin::hashes::sha256;
    use bitcoin::hashes::Hash;
    use bitcoin::Amount;
    use bitcoin::Block;
    use bitcoin::BlockHash;
    use bitcoin::OutPoint;
    use bitcoin::PubkeyHash;
    use bitcoin::ScriptBuf;
    use bitcoin::ScriptHash;
    use bitcoin::Transaction;
    use bitcoin::TxIn;
    use bitcoin::TxOut;
    use bitcoin::Txid;
    use bitcoin::WPubkeyHash;
    use bitcoin::WScriptHash;
    use floresta_common::impl_error_from;
    use rustreexo::accumulator::node_hash::BitcoinNodeHash;
    use rustreexo::accumulator::proof::Proof;
    use sha2::Digest;
    use sha2::Sha512_256;

    use super::LeafData;
    use crate::prelude::*;
    use crate::pruned_utreexo::consensus::UTREEXO_TAG_V1;
    use crate::pruned_utreexo::utxo_data::UtxoData;
    use crate::BlockchainError;
    use crate::CompactLeafData;
    use crate::ScriptPubKeyKind;
    use crate::UData;

    #[derive(Debug)]
    /// Errors that may occur while reconstructing a leaf's scriptPubKey.
    pub enum LeafErrorKind {
        /// The witness or scriptsig was empty, so nothing could be inspected.
        EmptyStack,

        /// The scriptsig data could not be parsed into `Instruction`s.
        InvalidInstruction(script::Error),

        /// The last instruction in the scriptsig was not an `OP_PUSHBYTES`.
        NotPushBytes,
    }

    /// Error while reconstructing a leaf's scriptPubKey, returned by `process_proof`.
    ///
    /// This error is triggered if the input lacks the hashed data required by the
    /// [ScriptPubKeyKind] (i.e., the public key for P2PKH, the redeem script for P2SH, or the
    /// witness public key and witness script for P2WPKH/P2WSH).
    #[derive(Debug)]
    pub struct UtreexoLeafError {
        pub leaf: CompactLeafData,
        pub txid: Txid,
        pub vin: usize,
        pub kind: LeafErrorKind,
    }

    impl_error_from!(LeafErrorKind, script::Error, InvalidInstruction);
    impl_error_from!(BlockchainError, UtreexoLeafError, UtreexoLeaf);

    impl Display for UtreexoLeafError {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "failed to reconstruct leaf {:?} for TxIn {}:{}: {:?}",
                self.leaf, self.txid, self.vin, self.kind
            )
        }
    }

    /// This function returns the scriptPubKey type (i.e. address type) of a given script data.
    /// It can be:
    ///
    /// - `PubKeyHash`: A pay-to-public-key-hash script.
    /// - `ScriptHash`: A pay-to-script-hash script.
    /// - `WitnessV0PubKeyHash`: A pay-to-witness-public-key-hash script.
    /// - `WitnessV0ScriptHash`: A pay-to-witness-script-hash script.
    /// - `Other`: a non specified script type. It is just copied over.
    pub fn get_script_type(script: &ScriptBuf) -> ScriptPubKeyKind {
        if script.is_p2pkh() {
            return ScriptPubKeyKind::PubKeyHash;
        }

        if script.is_p2sh() {
            return ScriptPubKeyKind::ScriptHash;
        }

        if script.is_p2wpkh() {
            return ScriptPubKeyKind::WitnessV0PubKeyHash;
        }

        if script.is_p2wsh() {
            return ScriptPubKeyKind::WitnessV0ScriptHash;
        }

        ScriptPubKeyKind::Other(script.to_bytes().into_boxed_slice())
    }

    /// Reconstructs the leaf data from a [CompactLeafData], the UTXO's block hash, and its spending tx input.
    pub fn reconstruct_leaf_data(
        leaf: &CompactLeafData,
        input: &TxIn,
        block_hash: BlockHash,
    ) -> Result<LeafData, LeafErrorKind> {
        let spk = reconstruct_script_pubkey(leaf, input)?;

        Ok(LeafData {
            block_hash,
            header_code: leaf.header_code,
            prevout: input.previous_output,
            utxo: TxOut {
                script_pubkey: spk,
                value: Amount::from_sat(leaf.amount),
            },
        })
    }

    /// Checks if a script is unspendable either by its length or if it contains the `OP_RETURN` opcode.
    /// It follows the implementation on Bitcoin Core.
    fn is_unspendable(script: &ScriptBuf) -> bool {
        if script.len() > 10_000 {
            return true;
        }

        if !script.is_empty() && script.as_bytes()[0] == 0x6a {
            return true;
        }

        false
    }

    /// Computes the hash of a leaf node in the utreexo accumulator.
    #[inline]
    fn get_leaf_hashes(
        txid: Txid,
        is_coinbase: bool,
        vout: u32,
        utxo: &TxOut,
        height: u32,
        block_hash: BlockHash,
    ) -> sha256::Hash {
        // An utreexo leaf hash is computed by hashing the UTXO bytes with some metadata
        let mut ser_utxo = Vec::new();
        utxo.consensus_encode(&mut ser_utxo)
            .expect("serializing TxOut never fails: Vec<u8>::Write always returns Ok");

        // Header code encodes the block height (at the 31 MSB) and coinbase flag (LSB = 1 for coinbase)
        let header_code = if is_coinbase {
            height << 1 | 1
        } else {
            height << 1
        };

        let leaf_hash = Sha512_256::new()
            .chain_update(UTREEXO_TAG_V1)
            .chain_update(UTREEXO_TAG_V1)
            .chain_update(block_hash)
            .chain_update(txid)
            .chain_update(vout.to_le_bytes())
            .chain_update(header_code.to_le_bytes())
            .chain_update(ser_utxo)
            .finalize();

        sha256::Hash::from_byte_array(leaf_hash.into())
    }

    /// From a block, gets the roots that will be included on the acc, certifying
    /// that any utxo will not be spent in the same block.
    pub fn get_block_adds(
        block: &Block,
        height: u32,
        block_hash: BlockHash,
    ) -> Vec<BitcoinNodeHash> {
        // Get inputs from the block, we'll need this HashSet to check if an output is spent
        // in the same block. If it is, we don't need to add it to the accumulator.
        let mut spent = HashSet::new();
        for tx in &block.txdata {
            for input in &tx.input {
                spent.insert((input.previous_output.txid, input.previous_output.vout));
            }
        }

        // Get all leaf hashes that will be added to the accumulator
        let mut adds = Vec::new();
        for tx in &block.txdata {
            let txid = tx.compute_txid();
            let is_cb = tx.is_coinbase();

            for (vout, output) in tx.output.iter().enumerate() {
                let utxo_id = (txid, vout as u32);

                if is_unspendable(&output.script_pubkey) || spent.contains(&utxo_id) {
                    // Do not add unspendable nor already spent utxos
                    continue;
                }
                adds.push(
                    get_leaf_hashes(txid, is_cb, vout as u32, output, height, block_hash).into(),
                );
            }
        }

        adds
    }

    /// A hash map that provides the UTXO data given the outpoint. We will get this data from either our own cache or the utreexo proofs, and use it to validate blocks and transactions.
    type UtxoMap = HashMap<OutPoint, UtxoData>;

    /// This function processes a proof of inclusion for a given block.
    /// It takes a `UData`, a slice of transactions, the block height, and a function to get the block hash.
    /// It returns a Result containing a Proof, a vector of deleted hashes, and a `UtxoMap`, which is defined as `HashMap<OutPoint, UtxoData>`.
    pub fn process_proof<F, E>(
        udata: &UData,
        txdata: &[Transaction],
        height: u32,
        get_block_hash: F,
    ) -> Result<(Proof, Vec<sha256::Hash>, UtxoMap), E>
    where
        F: Fn(u32) -> Result<BlockHash, E>,
        E: From<UtreexoLeafError>,
    {
        // Initialize return values
        let proof = Proof::from(&udata.proof);
        let mut del_hashes = Vec::new();
        let mut utxos = HashMap::new();

        let mut leaves_iter = udata.leaves.iter().cloned();

        // Skip coinbase transaction
        for tx in txdata.iter().skip(1) {
            let txid = tx.compute_txid();

            // Collect new UTXOs, which may be spent by later transactions in the block
            for (vout, out) in tx.output.iter().enumerate() {
                utxos.insert(
                    OutPoint::new(txid, vout as u32),
                    UtxoData {
                        txout: out.clone(),
                        is_coinbase: tx.is_coinbase(),
                        creation_height: height,
                        creation_time: 0, // TODO add MTP(`height` - 1)
                    },
                );
            }

            for (vin, input) in tx.input.iter().enumerate() {
                // Only reconstruct UTXOs missing from the map, from prior blocks. Transactions
                // spending uncreated UTXOs yield an invalid deletion hash, failing utreexo verification.
                if utxos.contains_key(&input.previous_output) {
                    continue;
                }
                let leaf = match leaves_iter.next() {
                    Some(leaf) => leaf,
                    None => continue,
                };

                let creation_height = leaf.header_code >> 1;
                // The coinbase flag is the LSB
                let is_coinbase = (leaf.header_code & 1) != 0;

                let hash = get_block_hash(creation_height)?;
                let leaf =
                    reconstruct_leaf_data(&leaf, input, hash).map_err(|e| UtreexoLeafError {
                        leaf,
                        txid,
                        vin,
                        kind: e,
                    })?;

                // Push the UTXO to remove from the set and its leaf hash (deletion hash)
                del_hashes.push(leaf._get_leaf_hashes());
                utxos.insert(
                    leaf.prevout,
                    UtxoData {
                        txout: leaf.utxo,
                        is_coinbase,
                        creation_height,
                        creation_time: 0, // TODO add MTP(`creation_height` - 1)
                    },
                );
            }
        }

        Ok((proof, del_hashes, utxos))
    }

    /// Reconstructs the output script, also called scriptPubKey, from a [CompactLeafData] and
    /// the spending tx input. Returns an error if we can't reconstruct the script (the input
    /// doesn't contain the required data).
    ///
    /// The reconstructed output script is the hash of either a public key or a script (i.e., P2PKH,
    /// P2SH, P2WPKH, and P2WSH).
    ///
    /// The logic behind is:
    ///
    /// For some script types, the output script is just the hash of something that needs to be
    /// revealed at some later stage (e.g., pkh is the hash of a public key that will be revealed
    /// afterwards in the scriptSig, at spend time). Therefore, this information is redundant,
    /// as we have it inside the spending transaction. For types where reconstruction is possible,
    /// we just need to communicate the type with a single byte marker, and the rest can be built
    /// from that using the spending transaction.
    pub fn reconstruct_script_pubkey(
        leaf: &CompactLeafData,
        input: &TxIn,
    ) -> Result<ScriptBuf, LeafErrorKind> {
        match &leaf.spk_ty {
            ScriptPubKeyKind::Other(spk) => Ok(ScriptBuf::from(spk.clone().into_vec())),
            ScriptPubKeyKind::PubKeyHash => {
                let pkhash = get_pk_hash(input)?;
                Ok(ScriptBuf::new_p2pkh(&pkhash))
            }
            ScriptPubKeyKind::WitnessV0PubKeyHash => {
                let pk_hash = get_witness_pk_hash(input)?;
                Ok(ScriptBuf::new_p2wpkh(&pk_hash))
            }
            ScriptPubKeyKind::ScriptHash => {
                let script_hash = get_script_hash(input)?;
                Ok(ScriptBuf::new_p2sh(&script_hash))
            }
            ScriptPubKeyKind::WitnessV0ScriptHash => {
                let witness_program_hash = get_witness_script_hash(input)?;
                Ok(ScriptBuf::new_p2wsh(&witness_program_hash))
            }
        }
    }

    /// Computes the public key hash from the pushed key in the input's scriptSig.
    fn get_pk_hash(input: &TxIn) -> Result<PubkeyHash, LeafErrorKind> {
        match input.script_sig.instructions().last() {
            None => Err(LeafErrorKind::EmptyStack),
            // Only valid if it's a push bytes instruction
            Some(Ok(Instruction::PushBytes(b))) => Ok(PubkeyHash::hash(b.as_bytes())),
            Some(Ok(_)) => Err(LeafErrorKind::NotPushBytes),
            Some(Err(e)) => Err(e.into()),
        }
    }

    /// Computes the script hash from the input's scriptSig.
    fn get_script_hash(input: &TxIn) -> Result<ScriptHash, LeafErrorKind> {
        match input.script_sig.instructions().last() {
            None => Err(LeafErrorKind::EmptyStack),
            // Only valid if it's a push bytes instruction
            Some(Ok(Instruction::PushBytes(b))) => Ok(ScriptHash::hash(b.as_bytes())),
            Some(Ok(_)) => Err(LeafErrorKind::NotPushBytes),
            Some(Err(e)) => Err(e.into()),
        }
    }

    /// Computes the witness public key hash from the input's witness data.
    fn get_witness_pk_hash(input: &TxIn) -> Result<WPubkeyHash, LeafErrorKind> {
        match input.witness.last() {
            Some(pk) => Ok(WPubkeyHash::hash(pk)),
            None => Err(LeafErrorKind::EmptyStack),
        }
    }

    /// Computes the witness script hash from the input's witness data.
    fn get_witness_script_hash(input: &TxIn) -> Result<WScriptHash, LeafErrorKind> {
        match input.witness.last() {
            Some(script) => Ok(WScriptHash::hash(script)),
            None => Err(LeafErrorKind::EmptyStack),
        }
    }
}

#[cfg(all(test, feature = "kv-chainstore"))]
mod test {
    extern crate std;

    use std::format;
    use std::str::FromStr;

    use bitcoin::blockdata::script;
    use bitcoin::consensus::encode::deserialize_hex;
    use bitcoin::hashes::sha256;
    use bitcoin::opcodes::all::OP_NOP;
    use bitcoin::opcodes::all::OP_PUSHBYTES_1;
    use bitcoin::Amount;
    use bitcoin::BlockHash;
    use bitcoin::Network;
    use bitcoin::ScriptBuf;
    use bitcoin::Transaction;
    use bitcoin::TxIn;
    use floresta_common::acchashes;
    use floresta_common::bhash;
    use rustreexo::accumulator::node_hash::BitcoinNodeHash;
    use rustreexo::accumulator::stump::Stump;

    use super::proof_util::reconstruct_leaf_data;
    use super::CompactLeafData;
    use super::LeafData;
    use super::ScriptPubKeyKind;
    use crate::proof_util::process_proof;
    use crate::proof_util::reconstruct_script_pubkey;
    use crate::proof_util::LeafErrorKind;
    use crate::AssumeValidArg;
    use crate::BlockchainError;
    use crate::ChainState;
    use crate::KvChainStore;
    use crate::UtreexoBlock;

    fn setup_test_chain<'a>(
        network: Network,
        assume_valid_arg: AssumeValidArg,
    ) -> ChainState<KvChainStore<'a>> {
        let test_id = rand::random::<u64>();
        let chainstore = KvChainStore::new(format!("./tmp-db/{test_id}/")).unwrap();
        ChainState::new(chainstore, network, assume_valid_arg)
    }

    fn to_acc_hashes(vec: Vec<sha256::Hash>) -> Vec<BitcoinNodeHash> {
        vec.into_iter().map(Into::into).collect()
    }

    #[test]
    fn test_process_proof_utxo_order() {
        // STEP 0: Set up the block and utreexo data
        let utreexo_block: UtreexoBlock = deserialize_hex("0100000011d60326c600e89bcea01a9884670bff6ba841728e70cd959d04000000000000a358f735ae382a260b709e818d55d5074f4bf0a930293a58e27baf19f18e9d203d19254f3fd40c1a62aa10e50b01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff34043e19254f010a2c4d4d3d3d8eb7a0c4ff07bdb0eb81f5f85f5c6865a9abc1552cfd4119191682bb5d9007910100000000000000ffffffff01a078072a0100000023210278aac0dccc3046120e0076a4cbfef47d4ff1d407f9597ffc21062ab55be3b2f7ac0000000001000000015f2cf7f37d5a83fc9ad514e42d415ef0342574712ecc365f55ca7c0ba79e12ba000000008b483045022100ef215fbea3427afaefccfe6159e63d303e2d658b46adf3de0e5e2c11f5590f2e022074bc3f5ae713290db79dfcef41ab669d92558a2b2f7f64c81e83f14ca887b841014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff02aee8371e010000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac00c2eb0b000000001976a91400591cd5fe5cfcefb38fa3d5ca7b1667ec52ac3388ac000000000100000001d71be7b3b3ea37c8ee8bbdc588366a6555e4cbeeca5173cd0d634a109716f630000000008a47304402200cba67f2c1e806ae2e5ee5a893cc55f549cf0af530cd86e532873b94552b5144022059249e1f3ca460f4d0661f3b3f00c59699e1fd4b50ec5396caaa4133aa27004e014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff0267cafb1d010000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac00127a00000000001976a914987c9643069837ea82cdfc6287fc74e7766fde2f88ac0000000001000000019574d909d8de1697acf4674d7031d28adb28b834c1b295ba80911a593f9003ad000000008b483045022023e6d08ec35dd9b7f77fb5eb6806647fb7bf2c46b056ccfa57314ccbe64eff0602210093d134951d33d139ebd20df4e6ce1a9b756cf49913f858415e965644b9e4afd6014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff020adc3415010000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac0024f400000000001976a914c5d64c5c00d62d80a69ae7d63dc081edc71dbc8e88ac0000000001000000011b1389c7ec6011d1a7f80d097607273bc0eb4fca4090936cfd13c4187469e6d3000000008b48304502204f4119c864e8a0e646b83740979db2f330e2b64243f2da7d85b931762ee0a673022100d7f909522d6c55dc899773d5c810dfea3837012b008876fd9a4094441f8a7b54014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff02f26065ff000000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac00e1f505000000001976a914f54b60efd3e70eb2f4866e7bd8db698bb13aa59788ac000000000100000001d3f9fc5aad3884b447e5921fe747e5aa6c1ef5bc354dfa96c5154e9d8aad40fb000000008a473044022002f329d215d6ea3311003ae5a996122f25e404864df4792f7094b34e823fdc37022000d573a1d114ebf43428e79280c853ceb46aa390fb075459d59454a755d3174d014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff02d00cebeb000000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac40420f00000000001976a91412257a70afb08e512cb9f91479a1edb9de6b81fe88ac000000000100000001507ee46589e2a81df64e2d5ee008c27e6a4e928fa95b7648da9328c1e9f91a9c000000008b483045022100e50513d07dca027abe419e9ef4eed989e40b4ea0a70e06db0aa3545752379419022011aa017ed5de218379ac6dd087563c086b7d4468a39dbb2b3128283e7117751d014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff02c5b57fe6000000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac00879303000000001976a914212441eebd4f77e2e3ef8c2d1bf0ff1687b3c87a88ac000000000100000001052a4d7756177801ace95de9b4d90d1f4bc7dfdb757baad72696453aa26085f8000000008a47304402207d692726dde4ef6bc88dd91e42d2d0df3098bff898fe739ab0599225d63c5a91022029224f844ca00e43f091c951a913012b7fbbad51619d5724149a7cf0dbe62b97014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff0240dc64b0000000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688acc000f837000000001976a914968845256e95c1982a9e60f060ea39f35792f0b788ac000000000100000001fb4cbcbd2436c632f73db5272608acb8dee23733d05f10b3f376f0c0d7b75cbd000000008b48304502210092b1b11f12176a708291b608c259eaff9ce3e98fb9db66175da25374860668f7022048f55939440743f9925b3d34232e613715718d10119a5c23c71f46cdacb49f63014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff0293816e27000000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac88685a00000000001976a9140d238c7c0ce909d511c193efcccb804584f6dbf888ac0000000001000000013afec4b3b670276e281bf3b6626ced8bc8356faee44348cbb4c65096af7dabfd000000008b483045022100e1cfbe6cdb54d081614fe2675d8554fdbd9271d616d875df30b046dd55d991be022013b9b63e78b260300a32219a2631f6b86d68492415c0f1b228010b195c6d3e53014104e1412429a76076dc25b2a7fc98c93da7b6e3e8045d48fecc3cb67c9163f6d616392458bdbf1e25c50171883619f91992dfe0abbecaac307e23be8e26ce9d19f8ffffffff0380a20442000000001976a914cea0534850d52cc6f19dadaf93ce30db9302187c88ac20a10700000000001976a9142911fad6636d73faea7ada74a30032c32620a94f88ac20a10700000000001976a914618118641b48c1679a3cc7ad00a9d246aa695fd888ac0000000001000000018fb56eb2ccdd91d7eb42e3e47515c828470ab88e721463fcafe9e7122e143f65000000008a47304402203fcdb0b731282e5a142305a304d23737a84ae8c7ce4bdd54d477846b572939d502205fbf74660a1a6f3eaaa48e0c5d24d9178f94c12498a269056d970f190a00197f0141040fed4775d5f5b47f2d73b35974b26ae19ccbde2c5a2ce741e5172c0baaa32bd30c23c664a5b91d84cf6c59e62a1c217dc318e24ce42977cbd8eaa89046d57639ffffffff0f005a6202000000001976a914dfcc1e75593def6563d3e35fa6615be5fd4f560e88ac80f0fa02000000001976a914ba23c86fe7c519d30f743e3491a67ef5d27c447388ac00e1f505000000001976a914dab269b027f6432ae7a6c34930485a4c8c9f5ace88ac0065cd1d000000001976a91408645206ee8a59c8d876e8bbb1a9512f5b1b14ea88ac00e1f505000000001976a914f16e0dbc76ccf3018f381b7726ce83cf874e444e88ac00e1f505000000001976a9146d293ad798d4cba8532b4f604e5c585674593d3b88ac00e1f505000000001976a91427be1119f8de54a8a7d3e65f131c6ca42eed7ba888ac80969800000000001976a914075c1dd720c63e10588b7a158aa7f43d4a6dd56288ac80969800000000001976a914c7d4c137526e35257fb94af3bd862fd8f1b65b5f88ac00e1f505000000001976a91495913e7275bfb35b3a0a9b749a013d56e54791f788ac80969800000000001976a914539845befa9b3017e0e42bfc6d1806b1d04dea2488ac00e1f505000000001976a9140a25839f64951ab152d0983d88d32cdc59120f7b88ace0151718000000001976a914ae4568bb485b0859c0d6db04d2094f3e5c43d2ce88ac80969800000000001976a9148c5a2a5c19f3f492a7df212c41f5e6df3504f4e988ac00879303000000001976a91425570998b4dfd2eb7c892c6d61772b0d209b050588ac00000000000afe2d024900fe46024900fe4a024900fe48024900fe4c024900fe4e024900fe50024900febd014900fee8024900fee602490021db2d9d4e21d41ec389811a282bddd70d2822bbc31d5c36bd25f2f7268bd2660ef15151f2df93ade1fa38c3807f6cb7446c5eb78131f188a5b5f4049b6b337090bf472b504976cbe726b549ca6fdd5aa198e0340eef4cc54609dd2b1b0e287d85379f13885795902a87a71e2c118dcf92e7b8d63d6ee2db727d1d535082817b5bf7b687299b0f8bedba623915bc8c11f37bbb852ee132ed28f1662dacd237babe4c0a5a306f1e0415e2db8e14d260c02f4b55f79552aff452bef1d66024c1ec07c365744675c11701c8b79cb942b42f5b120597dbe0ab2da7d60d502d799de9a91d11fb7546e30eb3e402bcca609436f3bb324808e0c143064799bd33691d892f60722b6bbacc53f9ceedf13dc0d88d93f7b407745537b0121fbb02e77b875995375b3b3cffc1c8b7101b45bfe963f91a5916bc46f755cfe578b21c2fa65d8caccbb69854d0246e0166f677ca4faeb7320523dc78b1d3c52d5a27700ba1f6061a6ca5297daa71ee816589e733bd455da6677aed14f0dde90c083dac1da7284e3fb72d97eb35051d774aa71fe3d434e28c227dbd53c1ae166ae76ac6fbaa744641d4a1050efbbc3c752f0a22f4187f600b8fe6be363597512ed6b5425ce017057b155edc7b9d3d04c52ab698cec15e32eeeb87e9b7bbc49ef6e1f25fd2400d2dda47c7239006477bcb15d5c5adb17e36b8df6ee0e26c8c778cdd21e8b3bd4970e3cc89d33d3ee3e6f16c00249e2a562f0f99a33fa12ae0130a67c137619d58c04a0e50e464bced58301ab8c0992b26653a15c0b2866e4af60d1172013d2d84c017c07cf7352220e579bb7989d938f34ac88cd73b45bf5e653346fd163380a5db1aa5537f1d2fafa09311dbf3c1478e16e21caed7eaebfab03c55cd7968f099d64a36411a4d7e2d23d7fa99ad8a33d701480ecfd426772d1961a0038af07bc3fca40d5ee4c9a212f5e1c437466d1f00875a6d42d04edfdafefd71046b96e1df309571bda1cbe97e8dc8b5efa0581a48e5d8fbe26066c01571b7dd6de32bc06d827bd78a03a80cff2b53374cc6fa61efff77aeda1e602a256b72f328187abb359bf10b60a36d599e18bbd71c98ee5dd910ba5efdad88ee45de41bd7fdb1775fcadcd60f9bc4e144f040e667cea94049d7f1df31cb54594da5879d027783bde41d4b658a2c540e41a2c04ff5b4169a2ee5492fe6a17e27700fb85bb717f4c5c955acb12f52bce5b9b7c90656cf9bbda51b0e80c82c9b4ff7967b1c0c8c5908d6c1c71e23b75268a6ada55750ff0a6e0d2c6ec3e29051aa5e632ad1bdc2c457fde45de454ca8e544b3c57da06acad7e3fff8aa0e23227fb9e40f192591f7d015d770e620187e065b47e8c5909d26a63a2da2047b26755c3970719c06285f7baa81e4071019ec78ca45522155090f11d42bff420034095608bc466cebcd51ba00514bcee815e587430248a07f9da39b0f3b2f2fafa5f667c59a9a03d1a7e2c7463433130a04040500aeaa232a01000000010404050067dc751e0100000001040405000a002916010000000104040500f2415b05010000000104040500104ffaeb000000000104040500c53c13ea00000000010404050000dd5ce80000000001fc0305001beac82700000000010804050010a81442000000000108040500b0affb640000000001").unwrap();
        let height = 164_357;
        let udata = utreexo_block.udata.as_ref().unwrap();
        let txdata = &utreexo_block.block.txdata;

        // This block spends UTXOs from three previous blocks
        let get_block_hash = |height| match height {
            164_354 => Ok(bhash!(
                "0000000000000274e38b79d9c971de45f66d80bd2a586efa3e947ab448c65f0a"
            )),
            164_350 => Ok(bhash!(
                "0000000000000305e4bb508c0b92a288ccac79b4c06bb92220a12650589e79df"
            )),
            164_356 => Ok(bhash!(
                "000000000000049d95cd708e7241a86bff0b6784981aa0ce9be800c62603d611"
            )),
            _ => Err(BlockchainError::BlockNotPresent),
        };

        let state = setup_test_chain(Network::Bitcoin, AssumeValidArg::Disabled);
        let acc = Stump {
            leaves: 4784881,
            roots: acchashes![
                "ad9b9fc757b185e729219b88c01011ddc57f7c009c16d52b45f18e03e4e4c8b8",
                "dc2a0fd968da4098822fb9941e7cd448336a6f8c4f0c3d5f170f6ea7323e125d",
                "6b6020ee6bb92cdd976b099de8e5ad86dd82a002b046899414006409149ddcd3",
                "158861d1078a9f394313afacaf70dd653d6439fcab287964cb1ff15ee0a4e232",
                "89000eaf560a5ba3a15af0dd3cf06aab929a9478060e0d4eda6e215f81ea82c7",
                "fae7b8f7dc1c8d2476002baa278b0381eebf88359c2da052228e3ea9110a67b4",
                "346b5f1bede34d73d9cdf3a78d29a35785a903abc397a1ce614a41158bc0bf7d",
                "b6b56f1ccc990e380ce8c528bb20d92919653d2c12d6d794aa7bac84ca9458e0",
                "7cad5f60ff32a8939d803d289c4d15cc44e98a2db2aa47e79bcf6712ea745ba6",
            ]
            .to_vec(),
        };

        // STEP 1: Verify the accumulator and the block
        let (proof, del_hashes, inputs) =
            process_proof(udata, txdata, height, get_block_hash).unwrap();

        if !acc.verify(&proof, &to_acc_hashes(del_hashes)).unwrap() {
            panic!("Proof must be valid")
        }
        state
            .validate_block_no_acc(&utreexo_block.block, height, inputs)
            .expect("Block validation must pass for the given UTXOs map");

        // STEP 2: Add a tx that tries to spend an UTXO created later in the block; utreexo fails
        let spending_tx: Transaction = deserialize_hex("0100000001ed5598cd2a9d8c4782e553dfc149a4a14402d1e4cfe61ae104f8cdf116486341000000008b483045022100aa2d2e5647b2ee15e42effd358946ce7d9515e5a56c21c9f6661b8d3f6f3fc51022028e433bcd457cb40bfdc253d6448f3bc2fc5bb7cc417cbf2458a9803489b9be5014104a39b9e4fbd213ef24bb9be69de4a118dd0644082e47c01fd9159d38637b83fbcdc115a5d6e970586a012d1cfe3e3a8b1a3d04e763bdc5a071c0e827c0bd834a5ffffffff0213ebd526000000001976a9140568015a9facccfd09d70d409b6fc1a5546cecc688ac80969800000000001976a91449bab0d204e6c1fdf99b0b8e0ebf70ab6421e2f488ac00000000").unwrap();
        assert_eq!(
            format!("{}", spending_tx.compute_txid()),
            "d55a5cb093e7d354ebb4ae3ecc12f281f17ca1194ff1d3ab16dd97cfd7eacf46",
        );

        let mut invalid_txdata = txdata.clone();
        invalid_txdata.insert(1, spending_tx);

        let (proof, del_hashes, _) =
            process_proof(udata, &invalid_txdata, height, get_block_hash).unwrap();

        if acc.verify(&proof, &to_acc_hashes(del_hashes)).unwrap() {
            panic!("Proof must be invalid")
        }
    }

    macro_rules! assert_recover_spk {
        (
            $tx_hex:literal,
            $height:literal,
            $index:literal,
            $amount:literal,
            $block_hash:literal,
            $spk_type:ident,
            $expected_spk:literal
        ) => {
            let s: Transaction = deserialize_hex($tx_hex).unwrap();
            let leaf = CompactLeafData {
                amount: Amount::from_btc($amount).unwrap().to_sat(),
                header_code: $height,
                spk_ty: ScriptPubKeyKind::$spk_type,
            };
            let spk =
                super::proof_util::reconstruct_leaf_data(&leaf, &s.input[0], bhash!($block_hash))
                    .unwrap();
            assert_eq!(
                spk.utxo.script_pubkey,
                ScriptBuf::from_hex($expected_spk).unwrap()
            )
        };
    }

    macro_rules! assert_recover_spk_err {
        ($spk_kind:ident, $tx_in:expr, $err_kind:ident $( ( $inner:pat ) )?) => {
            let compact = CompactLeafData {
                header_code: 0,
                amount: 1,
                // The only relevant field to return the different error kinds
                spk_ty: ScriptPubKeyKind::$spk_kind,
            };
            let err = reconstruct_script_pubkey(&compact, &$tx_in).unwrap_err();

            assert!(
                matches!(err, LeafErrorKind::$err_kind $( ( $inner ) )?),
                "Expected LeafErrorKind::{}, got {err:?}",
                stringify!($err_kind),
            );
        };
    }

    #[test]
    fn test_spk_recovery() {
        // p2pkh
        assert_recover_spk!(
            "010000000114baa734ec1a75e84726af2da3abcd41fe9d96f3f8b7e99bcefdfc040cffc2ba030000006a47304402202f89e2deb17f0c2c5732d6f7791a2731703cb128dc86ae0bf288e55a3d7ce9d6022051c2242ca0885a4a2054391385eda03132616fb0c2daa61d6823eff7a21b5d0c01210395c223fbf96e49e5b9e06a236ca7ef95b10bf18c074bd91a5942fc40360d0b68fdffffff04b400000000000000536a4c5058325bc5b3f7d4e7acf388d63ab92d14d7f8f8bcdff384bddd4668f283df0bfe1c2f7728ec1e550ca841794fa28e16e154c5b92b5a1d1d98db4e89f15726bf75e352fe000bddf10068000bd72600012bc20000000000000017a914352481ec2fecfde0c5cdc635a383c4ac27b9f71e87c20000000000000017a9144890aae025c84cb72a9730b49ca12595d6f6088d8772aa0900000000001976a914bf2646b8ba8b4a143220528bde9c306dac44a01c88ac00000000",
            0,
            777548,
            0.03956721,
            "000000000000000000066caa76847c109010eb58402d7a5bf05cc201a011071d",
            PubKeyHash,
            "76a914bf2646b8ba8b4a143220528bde9c306dac44a01c88ac"
        );
        // p2sh
        assert_recover_spk!(
            "0200000001ff1ba24eb11f1290b293b2c5520e4863ffedcc4a4ed9e4933334639ecbcc946500000000fc00473044022001460e6d06dc44e163ef1f692d275a1e357d086d0361fbe5012dbf18cbf2617202207f9e8fb54e776d7e98a6425da2be15e2ffca2e623b7617234226eafe77c70eaa01473044022076d756a250ad4044e2b4a0049112d87367b2f0ce80253e400f3ba09d620cbbdd022020f67b65f7cb5e109b8ccbc852e30b4e84b0b682136a5e72f679bd581b271ea8014c695221021c04b91bffe90c3e4defd021a4b6da4983b97e13c772bf15009f1661480658832102be11f7f0d9696ef731c13ed8b6e955df43cd4238d694a1698b02fcb3c2d275322102e0ad7274a4e93b3b30793ff7a04a31d2792ed22a563fe5ea0095af844c10c9c453aefdffffff02fd5403000000000017a914351bb17d072fff46336baec11a6a8d13ab6b590e87305837000000000017a9147b8d77369df3d2172b0d56792308d7f2635ca79087f1dd0b00",
            0,
            777548,
            0.03956721,
            "00000000000000000005a784e2b5006b34ff63644408df00bfc1a0b1b9507021",
            ScriptHash,
            "a914ed9371b30de550c0617cd0c4b2c0c0dc5e88c65487"
        );
        //p2wpkh
        assert_recover_spk!(
            "01000000000101a742910d02da84259631288eab229ca2bdd39ed7edc8811ca125dc0bcf2b654c0100000000ffffffff02ba150a000000000016001406a9852b7c9f4ff9993b5d2192ac42a5df54828e34c812000000000016001486bdf86c7cbce4841f95b4d8ef101ce8a306e6ad0247304402202936300c12249c8696bb90addcc9482995429d7be0418260178ddc0c630c10ed02206128cac337841b171d15d9aadc2af77d280da7cd85c049149c8134ddb5adc8a10121038adb3497e025c0ff14521a789af4f10d526ec4c95348e708ebdc4d5ac58228e500000000",
            1,
            777716,
            0.01893642,
            "00000000000000000002264d1e3f90280ededd1587c7af932480dac3e2480048",
            WitnessV0PubKeyHash,
            "001406a9852b7c9f4ff9993b5d2192ac42a5df54828e"
        );
        //p2wsh
        assert_recover_spk!(
            "01000000000101cacacdfdc79620cac8bc463cdac9864f557fdb73b6ef0dea8e0d74297d2e4c1a0100000000ffffffff0280841e000000000017a914cef5ab6252860ada719556abebe952c79c466f86878af74e0c00000000220020701a8d401c84fb13e6baf169d59684e17abd9fa216c8cc5b9fc63d622ff8c58d04004730440220289b2e0b6aec5a8f43d283edef0757206de77e3f3acdb322ade452a0468764db02201c332ec46a2ed3614fe392c4011063f39e77def57d89991ccbb99b6c7de2491901473044022044eaf71bdb4b3f0b0ba2f1eec82cad412729a1a4d5fc3b2fa251fecb73c56c0502201579c9e13b4d7595f9c6036a612828eac4796902c248131a7f25a117a0c68ca8016952210375e00eb72e29da82b89367947f29ef34afb75e8654f6ea368e0acdfd92976b7c2103a1b26313f430c4b15bb1fdce663207659d8cac749a0e53d70eff01874496feff2103c96d495bfdd5ba4145e3e046fee45e84a8a48ad05bd8dbb395c011a32cf9f88053ae00000000",
            0,
            487740,
            2.08551770,
            "0000000000000000004fce5d650f72e8f288e8c81b36377c3c7de3d2bc5b3118",
            WitnessV0ScriptHash,
            "0020701a8d401c84fb13e6baf169d59684e17abd9fa216c8cc5b9fc63d622ff8c58d"
        );
        assert_recover_spk!(
            "020000000001018f97e04dd76eec325c149ad417175f01f71b45523d8df79d2745cfee110eabf20000000000ffffffff015cddc71d000000002200204ae81572f06e1b88fd5ced7a1a000945432e83e1551e6f721ee9c00b8cc3326001015100000000",
            0,
            27366,
            4.99637721,
            "000000069585e4b2517a8862d527558ff18df7d4b8c2795b249c116aba9c6c98",
            WitnessV0ScriptHash,
            "00204ae81572f06e1b88fd5ced7a1a000945432e83e1551e6f721ee9c00b8cc33260"
        );
    }

    #[test]
    fn test_invalid_spk_recovery() {
        fn dummy_txin(script_sig: ScriptBuf) -> TxIn {
            TxIn {
                script_sig,
                ..Default::default()
            }
        }

        let empty_txin = dummy_txin(ScriptBuf::new());
        assert_recover_spk_err!(PubKeyHash, empty_txin, EmptyStack);
        assert_recover_spk_err!(ScriptHash, empty_txin, EmptyStack);
        assert_recover_spk_err!(WitnessV0PubKeyHash, empty_txin, EmptyStack);
        assert_recover_spk_err!(WitnessV0ScriptHash, empty_txin, EmptyStack);

        let mut script_sig = ScriptBuf::new();
        for _ in 0..255 {
            // Trying with different lengths of the script_sig
            script_sig.push_opcode(OP_NOP);

            let non_push_txin = dummy_txin(script_sig.clone());
            assert_recover_spk_err!(PubKeyHash, non_push_txin, NotPushBytes);
            assert_recover_spk_err!(ScriptHash, non_push_txin, NotPushBytes);
            assert_recover_spk_err!(WitnessV0PubKeyHash, non_push_txin, EmptyStack);
            assert_recover_spk_err!(WitnessV0ScriptHash, non_push_txin, EmptyStack);
        }
        // Using an `OP_PUSHBYTES` without data, i.e., an invalid instruction
        script_sig.push_opcode(OP_PUSHBYTES_1);

        let invalid_txin = dummy_txin(script_sig);
        assert_recover_spk_err!(
            PubKeyHash,
            invalid_txin,
            InvalidInstruction(script::Error::EarlyEndOfScript)
        );
        assert_recover_spk_err!(
            ScriptHash,
            invalid_txin,
            InvalidInstruction(script::Error::EarlyEndOfScript)
        );
        assert_recover_spk_err!(WitnessV0PubKeyHash, invalid_txin, EmptyStack);
        assert_recover_spk_err!(WitnessV0ScriptHash, invalid_txin, EmptyStack);
    }

    #[test]
    fn test_reconstruct_leaf_data() {
        let leaf: LeafData = deserialize_hex("f99e24b9e96a3c6220449b2bf520d6a9562237e2f4fc6f6b2ba57a71de000000e6f50efb6747f836ca3510df3da120fdb2ae4cf62893cc014e08c25dab70248b01000000cc000400b429653b4f0600001600142b91c8f80b071c5f60e1a512d49a6a544e51165b").unwrap();
        let spending_tx: Transaction = deserialize_hex("02000000000101e6f50efb6747f836ca3510df3da120fdb2ae4cf62893cc014e08c25dab70248b0100000000feffffff02dbe6553b4f0600001600148d57f8da7fc15371dc14d35e97850ab564a17b1240420f0000000000220020ed59bf193c5197a5b1dbbbc723ddeca82cdfbb188218b3ede50150e1890fc55202473044022024979ec4bda473b71288b2c15390418d7d300551aa5e463cc6b64acd5c3070b50220444c94242aff2ba1bd966308d60f537524b0755931d545d98e1fc45239ff6b08012103de7c420624c009d6a5761871e78b39ff864887f885ed313e27f778b3772e74916a000200").unwrap();

        let compact = CompactLeafData {
            amount: Amount::from_btc(69373.68668596).unwrap().to_sat(),
            header_code: 262348,
            spk_ty: ScriptPubKeyKind::WitnessV0PubKeyHash,
        };
        let reconstructed = reconstruct_leaf_data(
            &compact,
            &spending_tx.input[0],
            bhash!("000000de717aa52b6b6ffcf4e2372256a9d620f52b9b4420623c6ae9b9249ef9"),
        )
        .unwrap();
        assert_eq!(leaf, reconstructed);
    }
}

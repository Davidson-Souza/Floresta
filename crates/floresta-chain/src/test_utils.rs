// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers for the `floresta-chain` unit tests.
//!
//! This module is only compiled during tests and centralizes the
//! block/transaction-building helpers used by different test modules.

use std::fs::File;

use bitcoin::Block;
use bitcoin::OutPoint;
use bitcoin::ScriptBuf;
use bitcoin::Transaction;
use bitcoin::TxIn;
use bitcoin::TxOut;
use bitcoin::absolute::LockTime;
use bitcoin::consensus::deserialize;
use bitcoin::hashes::Hash;
use bitcoin::opcodes::all::OP_NOP;
use bitcoin::transaction::Version;
use rand::SeedableRng;
use rand::prelude::IndexedMutRandom;
use rand::rngs::StdRng;

#[macro_export]
/// Macro for creating a TxOut
macro_rules! txout {
    ($sats:expr, $script:expr) => {
        bitcoin::TxOut {
            value: bitcoin::Amount::from_sat($sats),
            script_pubkey: $script,
        }
    };
}

#[macro_export]
/// Macro for constructing a legacy [`TxIn`] with optional scriptSig and sequence number.
/// Needs the outpoint and, if not provided, defaults to empty scriptSig and `Sequence::MAX`.
macro_rules! txin {
    ($outpoint:expr) => {
        bitcoin::TxIn {
            previous_output: $outpoint,
            script_sig: bitcoin::ScriptBuf::new(),
            sequence: bitcoin::Sequence::MAX,
            witness: bitcoin::Witness::new(),
        }
    };
    ($outpoint:expr, $script:expr) => {
        bitcoin::TxIn {
            previous_output: $outpoint,
            script_sig: $script,
            sequence: bitcoin::Sequence::MAX,
            witness: bitcoin::Witness::new(),
        }
    };
    ($outpoint:expr, $script:expr, $sequence:expr) => {
        bitcoin::TxIn {
            previous_output: $outpoint,
            script_sig: $script,
            sequence: $sequence,
            witness: bitcoin::Witness::new(),
        }
    };
}

/// Test helper to update the witness commitment in a block, assuming txdata was modified.
/// This ensures `block` is not considered mutated, so we can exercise other error cases.
pub(crate) fn update_witness_commitment(block: &mut Block) -> Option<()> {
    // BIP141 witness-commitment prefix, where the full commitment data is:
    //  1-byte - OP_RETURN (0x6a)
    //  1-byte - Push the following 36 bytes (0x24)
    //  4-byte - Commitment header (0xaa21a9ed)
    // 32-byte - Commitment hash: Double-SHA256(witness root hash|witness reserved value)
    const MAGIC: [u8; 6] = [0x6a, 0x24, 0xaa, 0x21, 0xa9, 0xed];

    let coinbase = &block.txdata[0];

    // Commitment is in the last coinbase output that starts with magic bytes.
    let pos = coinbase.output.iter().rposition(|out| {
        let spk = out.script_pubkey.as_bytes();
        spk.len() >= 38 && spk[0..6] == MAGIC
    })?;

    // Witness reserved value is in coinbase input witness.
    let witness_rv: &[u8; 32] = {
        let mut it = coinbase.input[0].witness.iter();
        match (it.next(), it.next()) {
            (Some(rv), None) => rv.try_into().ok()?,
            _ => return None,
        }
    };

    let root = block.witness_root()?;
    let c = *Block::compute_witness_commitment(&root, witness_rv).as_byte_array();

    block.txdata[0].output[pos].script_pubkey.as_mut_bytes()[6..38].copy_from_slice(&c);
    Some(())
}

/// Decode and deserialize a zstd-compressed block in the given file path.
pub(crate) fn decode_block(file_path: &str) -> Block {
    let block_file = File::open(file_path).unwrap();
    let block_bytes = zstd::decode_all(block_file).unwrap();
    deserialize(&block_bytes).unwrap()
}

pub(crate) fn decode_block_866342() -> Block {
    decode_block("./testdata/block_866342/raw.zst")
}

/// Modifies historical block at height 866,342 by adding one extra transaction so that the
/// updated block weight is 4,000,001 WUs. The block merkle roots are updated accordingly.
pub(crate) fn build_oversized_866_342() -> Block {
    let mut block = decode_block_866342();

    // This block is close but below to the max weight
    assert_eq!(block.weight().to_wu(), 3_993_209);

    // Modify the block by adding one transaction that makes it exceed the weight limit
    let mut script_out = ScriptBuf::default();
    for _ in 0..1_636 {
        script_out.push_opcode(OP_NOP);
    }
    let out = txout!(1, script_out);
    let tx = build_tx(vec![txin!(dummy_outpoint())], vec![out]);

    block.txdata.insert(1, tx);

    // Update the witness commitment, and then the merkle root, which depends on the former
    update_witness_commitment(&mut block).expect("should be able to update");
    block.header.merkle_root = block.compute_merkle_root().unwrap();

    block
}

/// Helper for building a zero-locktime transaction given the input and output list.
pub(crate) fn build_tx(input: Vec<TxIn>, output: Vec<TxOut>) -> Transaction {
    Transaction {
        version: Version::TWO,
        lock_time: LockTime::ZERO,
        input,
        output,
    }
}

/// Helper to avoid boilerplate in test cases. Note this is not a null outpoint, restricted to
/// coinbase transactions only, as that requires the vout to be `u32::MAX`.
pub(crate) fn dummy_outpoint() -> OutPoint {
    OutPoint {
        txid: bitcoin::Txid::all_zeros(),
        vout: 0,
    }
}

pub(crate) fn coinbase(is_valid: bool) -> Transaction {
    // This coinbase transaction was retrieved from https://learnmeabitcoin.com/explorer/block/0000000000000a0f82f8be9ec24ebfca3d5373fde8dc4d9b9a949d538e9ff679

    // Create input
    let script_sig = if is_valid {
        ScriptBuf::from_hex("03f0a2a4d9f0a2").unwrap()
    } else {
        // This must invalidate the coinbase transaction since it's a big script.
        ScriptBuf::from_hex(&format!("{:0>420}", "")).unwrap()
    };
    let input = txin!(OutPoint::null(), script_sig);

    // Create outputs
    let output_script = ScriptBuf::from_hex("41047eda6bd04fb27cab6e7c28c99b94977f073e912f25d1ff7165d9c95cd9bbe6da7e7ad7f2acb09e0ced91705f7616af53bee51a238b7dc527f2be0aa60469d140ac").unwrap();
    let output = txout!(5_000_350_000, output_script);

    Transaction {
        version: Version::ONE,
        lock_time: LockTime::from_height(150_007).unwrap(),
        input: vec![input],
        output: vec![output],
    }
}

/// Modifies a block to have a different output script (txdata is tampered with).
pub(crate) fn mutate_block(block: &mut Block) {
    let mut rng = StdRng::seed_from_u64(0x_bebe_cafe);

    let tx = block.txdata.choose_mut(&mut rng).unwrap();
    let out = tx.output.choose_mut(&mut rng).unwrap();
    let spk = out.script_pubkey.as_mut_bytes();
    // Random byte from a random scriptPubKey
    let byte = spk.choose_mut(&mut rng).unwrap();

    *byte += 1;
}

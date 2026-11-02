// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bitcoin Core-compatible signature-operation cost accounting for [`Transaction`].
//!
//! [`SigopExt`] delegates context-free legacy counting to rust-bitcoin and implements the
//! prevout-dependent P2SH and witness rules its API cannot express exactly.

use bitcoin::OutPoint;
use bitcoin::Transaction;

use super::chainparams::VERIFY_P2SH;
use super::chainparams::VERIFY_WITNESS;
use super::utxo_data::UtxoData;
use crate::prelude::HashMap;

pub(crate) trait SigopExt {
    /// Returns the legacy sigop cost using Bitcoin Core's
    /// [`GetLegacySigOpCount`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/consensus/tx_verify.cpp#L112-L124).
    ///
    /// [`Transaction::total_sigop_cost`] is Core-equivalent with `|_| None`: without
    /// prevouts it skips P2SH and witness counting, while legacy `CHECKMULTISIG`
    /// always counts as 20 sigops and therefore needs no accurate `OP_N` handling.
    fn legacy_sigop_cost(&self) -> usize;

    /// Returns the total sigop cost using Bitcoin Core's
    /// [`GetTransactionSigOpCost`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/consensus/tx_verify.cpp#L143-L162)
    /// rules: legacy and P2SH sigops cost four units, witness-v0 sigops cost one.
    ///
    /// This cannot use [`Transaction::total_sigop_cost`] with prevouts because
    /// rust-bitcoin 0.32.8:
    /// - always counts P2SH and witness instead of independently honoring
    ///   `VERIFY_P2SH` and `VERIFY_WITNESS`,
    /// - counts a P2SH redeem script without requiring the whole scriptSig to be push-only,
    /// - may apply an `OP_N` to a non-adjacent `CHECKMULTISIG`, undercounting P2SH/P2WSH scripts.
    ///
    /// TODO: Delegate this method upstream once `total_sigop_cost` accepts verification
    /// flags and implements all the contextual rules above (or even get rid of the trait).
    fn sigop_cost(&self, utxos: &HashMap<OutPoint, UtxoData>, flags: u32) -> usize;
}

impl SigopExt for Transaction {
    fn legacy_sigop_cost(&self) -> usize {
        self.total_sigop_cost(|_| None)
    }

    fn sigop_cost(&self, utxos: &HashMap<OutPoint, UtxoData>, flags: u32) -> usize {
        let mut cost = self.legacy_sigop_cost();
        if self.is_coinbase() || flags & (VERIFY_P2SH | VERIFY_WITNESS) == 0 {
            return cost;
        }

        for input in &self.input {
            let Some(prevout) = utxos.get(&input.previous_output) else {
                continue;
            };
            let prevout_script = prevout.txout.script_pubkey.as_script();

            if flags & VERIFY_P2SH != 0 {
                cost = cost.saturating_add(contextual::p2sh_sigop_cost(input, prevout_script));
            }
            if flags & VERIFY_WITNESS != 0 {
                cost = cost.saturating_add(contextual::witness_sigop_cost(input, prevout_script));
            }
        }
        cost
    }
}

/// Prevout-dependent P2SH and witness sigop accounting.
mod contextual {
    use bitcoin::Script;
    use bitcoin::TxIn;
    use bitcoin::constants::WITNESS_SCALE_FACTOR;
    use bitcoin::opcodes::Class;
    use bitcoin::opcodes::ClassifyContext;
    use bitcoin::opcodes::Ordinary;
    use bitcoin::script;

    /// The per-input part of Bitcoin Core's
    /// [`GetP2SHSigOpCount`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/consensus/tx_verify.cpp#L126-L141),
    /// scaled to sigop cost units.
    pub(super) fn p2sh_sigop_cost(input: &TxIn, prevout: &Script) -> usize {
        if !prevout.is_p2sh() {
            return 0;
        }

        let Some(redeem_script) = pushed_redeem_script(&input.script_sig) else {
            return 0;
        };

        let sigop_count = count_sigops_accurate(redeem_script);
        sigop_count.saturating_mul(WITNESS_SCALE_FACTOR)
    }

    /// Bitcoin Core's per-input
    /// [`CountWitnessSigOps`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/script/interpreter.cpp#L2139-L2166).
    pub(super) fn witness_sigop_cost(input: &TxIn, prevout: &Script) -> usize {
        let program = match prevout {
            script if script.is_witness_program() => script, // Native witness
            script if script.is_p2sh() => {
                // P2SH-wrapped witness
                let Some(redeem) = pushed_redeem_script(&input.script_sig) else {
                    return 0;
                };
                redeem
            }
            _ => return 0,
        };

        match program {
            script if script.is_p2wpkh() => 1, // The implied CHECKSIG
            script if script.is_p2wsh() => input.witness.last().map_or(0, |script| {
                count_sigops_accurate(Script::from_bytes(script))
            }),
            // Taproot uses BIP342's separate per-input sigops budget:
            // https://github.com/bitcoin/bips/blob/master/bip-0342.mediawiki#resource-limits
            _ => 0,
        }
    }

    /// Core's accurate [`CScript::GetSigOpCount`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/script/script.cpp#L159-L181).
    ///
    /// This backports the accurate path of rust-bitcoin master's
    /// [`count_sigops_internal`](https://github.com/rust-bitcoin/rust-bitcoin/blob/master/bitcoin/src/blockdata/script/borrowed.rs#L343-L383).
    ///
    /// TODO: Replace this with [`Script::count_sigops`] after upgrading to a release containing
    /// [rust-bitcoin#6409](https://github.com/rust-bitcoin/rust-bitcoin/pull/6409).
    fn count_sigops_accurate(script: &Script) -> usize {
        let (mut count, mut last_pushnum) = (0, None);
        for instruction in script.instructions().map_while(Result::ok) {
            let Some(opcode) = instruction.opcode() else {
                last_pushnum = None;
                continue;
            };
            last_pushnum = match opcode.classify(ClassifyContext::Legacy) {
                Class::Ordinary(Ordinary::OP_CHECKSIG | Ordinary::OP_CHECKSIGVERIFY) => {
                    count += 1;
                    None
                }
                Class::Ordinary(Ordinary::OP_CHECKMULTISIG | Ordinary::OP_CHECKMULTISIGVERIFY) => {
                    count += last_pushnum.unwrap_or(20); // MAX_PUBKEYS_PER_MULTISIG
                    None
                }
                Class::PushNum(n @ 1..=16) => Some(n as usize),
                _ => None,
            };
        }
        count
    }

    /// Matches Core's redeem-script extraction in
    /// [`CScript::GetSigOpCount(scriptSig)`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/script/script.cpp#L183-L205)
    /// and [`CountWitnessSigOps`](https://github.com/bitcoin/bitcoin/blob/v31.0/src/script/interpreter.cpp#L2139-L2166).
    fn pushed_redeem_script(script_sig: &Script) -> Option<&Script> {
        if !script_sig.is_push_only() {
            return None;
        }
        match script_sig.instructions().last() {
            Some(Ok(script::Instruction::PushBytes(bytes))) => {
                Some(Script::from_bytes(bytes.as_bytes()))
            }
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use bitcoin::ScriptBuf;
        use bitcoin::TxIn;
        use bitcoin::WPubkeyHash;
        use bitcoin::Witness;
        use bitcoin::hashes::Hash;
        use bitcoin::opcodes::Opcode;
        use bitcoin::opcodes::all::OP_CHECKMULTISIG;
        use bitcoin::opcodes::all::OP_CHECKSIG;
        use bitcoin::opcodes::all::OP_PUSHBYTES_0;
        use bitcoin::opcodes::all::OP_PUSHDATA1;
        use bitcoin::opcodes::all::OP_PUSHNUM_1;
        use bitcoin::opcodes::all::OP_PUSHNUM_2;
        use bitcoin::script::Builder;
        use bitcoin::script::PushBytesBuf;

        use super::count_sigops_accurate;
        use super::pushed_redeem_script;
        use super::witness_sigop_cost;

        fn ops<const N: usize>(opcodes: [Opcode; N]) -> ScriptBuf {
            ScriptBuf::from_bytes(opcodes.map(Opcode::to_u8).to_vec())
        }

        #[test]
        fn accurate_sigop_count_edges() {
            let script = ops([OP_PUSHNUM_1, OP_CHECKSIG, OP_CHECKMULTISIG]);

            // Core clears OP_1 at CHECKSIG and counts 1 + 20; rust-bitcoin 0.32.8 returns 1 + 1.
            assert_eq!(
                script.count_sigops(),
                2,
                "rust-bitcoin fixed this, retire count_sigops_accurate!"
            );
            assert_eq!(count_sigops_accurate(&script), 21);

            // OP_2 counts two only when adjacent; an intervening data push restores the default 20.
            let adjacent = ops([OP_PUSHNUM_2, OP_CHECKMULTISIG]);
            let separated = ops([OP_PUSHNUM_2, OP_PUSHBYTES_0, OP_CHECKMULTISIG]);
            assert_eq!(count_sigops_accurate(&adjacent), 2);
            assert_eq!(count_sigops_accurate(&separated), 20);

            // A truncated push stops parsing but preserves the preceding CHECKSIG.
            let malformed =
                ScriptBuf::from_bytes(vec![OP_CHECKSIG.to_u8(), OP_PUSHDATA1.to_u8(), 1]);
            assert_eq!(count_sigops_accurate(&malformed), 1);
        }

        #[test]
        fn pushed_redeem_script_matches_core_push_rules() {
            let redeem = ScriptBuf::from_bytes(vec![OP_CHECKSIG.to_u8()]);
            let script_sig =
                |len| ScriptBuf::from_bytes(vec![OP_PUSHDATA1.to_u8(), len, OP_CHECKSIG.to_u8()]);

            // PUSHDATA1 is non-minimal but valid for one byte; claiming two makes it truncated.
            let pushed = script_sig(1);
            assert_eq!(pushed_redeem_script(&pushed), Some(redeem.as_script()));
            assert_eq!(pushed_redeem_script(&script_sig(2)), None);

            // Core permits OP_1 before the final byte push, Script::redeem_script does not.
            let with_pushnum =
                ScriptBuf::from_bytes([&[OP_PUSHNUM_1.to_u8()], pushed.as_bytes()].concat());

            assert_eq!(
                pushed_redeem_script(&with_pushnum),
                Some(redeem.as_script())
            );
        }

        #[test]
        fn witness_sigop_cost_covers_v0_and_future_programs() {
            let witness_script = ops([OP_PUSHNUM_2, OP_CHECKMULTISIG]);
            let p2wsh = witness_script.to_p2wsh();
            let native_input = TxIn {
                witness: Witness::from_slice(&[witness_script.as_bytes()]),
                ..TxIn::default()
            };

            // P2WSH accurately counts OP_2 CHECKMULTISIG in the witness script as two.
            assert_eq!(witness_sigop_cost(&native_input, &p2wsh), 2);
            // Without a witness script, there is nothing to count.
            assert_eq!(witness_sigop_cost(&TxIn::default(), &p2wsh), 0);

            // P2SH-wrapped P2WPKH still contributes its one implied CHECKSIG.
            let p2wpkh = ScriptBuf::new_p2wpkh(&WPubkeyHash::all_zeros());
            let program = PushBytesBuf::try_from(p2wpkh.to_bytes()).unwrap();
            let wrapped_input = TxIn {
                script_sig: Builder::new().push_slice(program).into_script(),
                ..TxIn::default()
            };
            let wrapped_p2wpkh = p2wpkh.to_p2sh();
            assert_eq!(witness_sigop_cost(&wrapped_input, &wrapped_p2wpkh), 1);

            // Taproot is excluded from the block-wide cost as BIP342 gives it a per-input budget.
            let taproot = Builder::new().push_int(1).push_slice([0; 32]).into_script();
            assert_eq!(witness_sigop_cost(&native_input, &taproot), 0);
        }
    }
}

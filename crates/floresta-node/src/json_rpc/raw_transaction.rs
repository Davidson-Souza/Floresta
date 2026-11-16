// SPDX-License-Identifier: MIT OR Apache-2.0

use bitcoin::Address;
use bitcoin::Transaction;
use bitcoin::TxIn;
use bitcoin::TxOut;
use bitcoin::Txid;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::hashes::hex::FromHex;
use bitcoin::hex;
use bitcoin::hex::DisplayHex;
use floresta_domain::node_interface::MempoolMethods;
use floresta_domain::node_interface::NodeError;
use floresta_rpc::rpc_interfaces::RawTransactionRpc;
use floresta_rpc::rpc_types::GetRawTransactionRes;
use floresta_rpc::rpc_types::GetRawTransactionVerbose;
use floresta_rpc::rpc_types::RawTransactionInput;
use floresta_rpc::rpc_types::RawTransactionOutput;
use floresta_rpc::rpc_types::ScriptPubKey;
use floresta_rpc::rpc_types::ScriptSig;
use floresta_watch_only::CachedTransaction;

use super::server::RpcChain;
use super::server::RpcImpl;
use crate::json_rpc::res::jsonrpc_interface::JsonRpcError;
use crate::json_rpc::server::to_core_asm_string;

impl<Blockchain: RpcChain> RawTransactionRpc for RpcImpl<Blockchain> {
    type Error = JsonRpcError;

    async fn get_raw_transaction(
        &self,
        tx_id: Txid,
        verbosity: Option<u8>,
    ) -> Result<GetRawTransactionRes, JsonRpcError> {
        let verbosity = verbosity.unwrap_or(0);

        if verbosity > 1 {
            return Err(JsonRpcError::InvalidVerbosityLevel);
        }

        let tx = self
            .wallet
            .get_transaction(&tx_id)?
            .ok_or(JsonRpcError::TxNotFound)?;

        match verbosity {
            0 => Ok(GetRawTransactionRes::Zero(serialize_hex(&tx.tx))),
            1 => Ok(GetRawTransactionRes::One(Box::new(
                self.make_raw_transaction(tx)?,
            ))),
            _ => Err(JsonRpcError::InvalidVerbosityLevel),
        }
    }

    async fn send_raw_transaction(&self, tx: String) -> Result<Txid, JsonRpcError> {
        let tx_hex = Vec::from_hex(&tx).map_err(|_| JsonRpcError::InvalidHex)?;
        let tx: Transaction =
            deserialize(&tx_hex).map_err(|e| JsonRpcError::Decode(e.to_string()))?;

        self.node
            .broadcast_transaction(tx)
            .await
            .map_err(|e| match e {
                NodeError::Mempool(e) => JsonRpcError::MempoolAccept(e),
                e => JsonRpcError::Node(e.to_string()),
            })
    }
}

impl<Blockchain: RpcChain> RpcImpl<Blockchain> {
    fn make_vin(&self, input: TxIn, is_coinbase: bool) -> RawTransactionInput {
        let sequence = input.sequence.0;
        let txin_witness = (!input.witness.is_empty()).then_some(
            input
                .witness
                .iter()
                .map(|w| w.to_hex_string(hex::Case::Lower))
                .collect(),
        );

        if is_coinbase {
            return RawTransactionInput {
                coinbase: Some(input.script_sig.to_hex_string()),
                sequence,
                txin_witness,
                script_sig: None,
                txid: None,
                vout: None,
            };
        }

        let txid = Some(input.previous_output.txid.to_string());
        let vout = Some(input.previous_output.vout);
        let script_sig = ScriptSig {
            asm: to_core_asm_string(&input.script_sig, true),
            hex: input.script_sig.to_hex_string(),
        };

        RawTransactionInput {
            coinbase: None,
            txid,
            vout,
            script_sig: Some(script_sig),
            txin_witness,
            sequence,
        }
    }

    fn make_vout(&self, output: TxOut, index: u64) -> RawTransactionOutput {
        let value = output.value;
        RawTransactionOutput {
            value: value.to_btc(),
            index,
            script_pubkey: ScriptPubKey {
                asm: to_core_asm_string(&output.script_pubkey, false),
                hex: output.script_pubkey.to_hex_string(),
                // `Address::from_script` can fail for nonstandard scripts. Bitcoin Core
                // omits the `address` field entirely when `ExtractDestination` fails:
                // https://github.com/bitcoin/bitcoin/blob/f50d53c84736f8ada8419346c4d1734d5a6686d4/src/core_io.cpp#L424
                address: Address::from_script(&output.script_pubkey, self.network)
                    .map(|a| a.to_string())
                    .ok(),
                type_: Self::get_script_type_label(&output.script_pubkey).to_string(),
                descriptor: Some(Self::get_script_type_descriptor(
                    &output.script_pubkey,
                    &Address::from_script(&output.script_pubkey, self.network).ok(),
                )),
                required_signatures: None, // This field is deprecated in Core v22
                addresses: None,           // This field is deprecated in Core v22
            },
        }
    }

    fn make_raw_transaction(
        &self,
        tx: CachedTransaction,
    ) -> Result<GetRawTransactionVerbose, JsonRpcError> {
        let raw_tx = tx.tx;
        let in_active_chain = tx.height != 0;
        let hex = serialize_hex(&raw_tx);
        let txid = raw_tx.compute_txid().to_string();

        let mut block_hash = None;
        let mut block_time = None;
        let mut transaction_time = None;
        let mut confirmations = Some(0);
        if in_active_chain {
            confirmations = self.chain.get_height().ok().and_then(|tip| {
                if tip >= tx.height {
                    Some((tip - tx.height + 1).into())
                } else {
                    None
                }
            });

            if let Ok(hash) = self.chain.get_block_hash(tx.height) {
                if let Ok(header) = self.chain.get_block_header(&hash) {
                    block_hash = Some(header.block_hash().to_string());
                    block_time = Some(header.time.into());
                    transaction_time = Some(header.time.into());
                }
            }
        }

        Ok(GetRawTransactionVerbose {
            in_active_chain: Some(in_active_chain),
            hex,
            txid,
            hash: raw_tx.compute_wtxid().to_string(),
            size: raw_tx.total_size().try_into()?,
            vsize: raw_tx.vsize().try_into()?,
            weight: raw_tx.weight().to_wu(),
            version: raw_tx.version.0,
            lock_time: raw_tx.lock_time.to_consensus_u32(),
            inputs: raw_tx
                .input
                .iter()
                .map(|input| self.make_vin(input.clone(), raw_tx.is_coinbase()))
                .collect(),
            outputs: raw_tx
                .output
                .into_iter()
                .enumerate()
                .map(
                    |(i, output)| -> Result<RawTransactionOutput, JsonRpcError> {
                        let index = i.try_into()?;
                        Ok(self.make_vout(output, index))
                    },
                )
                .collect::<Result<Vec<_>, JsonRpcError>>()?,
            block_hash,
            confirmations,
            block_time,
            transaction_time,
        })
    }
}

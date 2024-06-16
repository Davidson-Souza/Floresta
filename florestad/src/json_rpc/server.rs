use std::net::SocketAddr;
use std::sync::Arc;

use async_std::sync::RwLock;
use bitcoin::block::Header as BlockHeader;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::consensus::serialize;
use bitcoin::constants::genesis_block;
use bitcoin::hashes::hex::FromHex;
use bitcoin::hashes::Hash;
use bitcoin::hex::DisplayHex;
use bitcoin::Address;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::ScriptBuf;
use bitcoin::TxIn;
use bitcoin::TxOut;
use bitcoin::Txid;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_chain::ChainState;
use floresta_chain::KvChainStore;
use floresta_compact_filters::kv_filter_database::KvFilterStore;
use floresta_compact_filters::network_filters::NetworkFilters;
use floresta_watch_only::kv_database::KvDatabase;
use floresta_watch_only::AddressCache;
use floresta_watch_only::CachedTransaction;
use floresta_wire::node_interface::NodeInterface;
use floresta_wire::node_interface::NodeMethods;
use floresta_wire::node_interface::PeerInfo;
use futures::executor::block_on;
use jsonrpc_core::Result;
use jsonrpc_derive::rpc;
use jsonrpc_http_server::ServerBuilder;
use log::info;
use serde_json::json;
use serde_json::Value;

use super::res::BlockJson;
use super::res::Error;
use super::res::GetBlockchainInfoRes;
use super::res::RawTxJson;
use super::res::ScriptPubKeyJson;
use super::res::ScriptSigJson;
use super::res::TxInJson;
use super::res::TxOutJson;

#[rpc]
pub trait Rpc {
    #[rpc(name = "getblockfilter")]
    fn get_block_filter(&self, heigth: u32) -> Result<String>;
    #[rpc(name = "getblockchaininfo")]
    fn get_blockchain_info(&self) -> Result<GetBlockchainInfoRes>;
    #[rpc(name = "getblockhash")]
    fn get_block_hash(&self, height: u32) -> Result<BlockHash>;
    #[rpc(name = "getblockheader")]
    fn get_block_header(&self, hash: BlockHash) -> Result<BlockHeader>;
    #[rpc(name = "gettransaction")]
    fn get_transaction(&self, tx_id: Txid, verbosity: Option<bool>) -> Result<Value>;
    #[rpc(name = "gettxproof")]
    fn get_tx_proof(&self, tx_id: Txid) -> Result<Vec<String>>;
    #[rpc(name = "loaddescriptor")]
    fn load_descriptor(&self, descriptor: String, rescan: Option<u32>) -> Result<bool>;
    #[rpc(name = "rescan")]
    fn rescan(&self, rescan: u32) -> Result<bool>;
    #[rpc(name = "getheight")]
    fn get_height(&self) -> Result<u32>;
    #[rpc(name = "sendrawtransaction")]
    fn send_raw_transaction(&self, tx: String) -> Result<Txid>;
    #[rpc(name = "getroots")]
    fn get_roots(&self) -> Result<Vec<String>>;
    #[rpc(name = "getpeerinfo")]
    fn get_peer_info(&self) -> Result<Vec<PeerInfo>>;
    #[rpc(name = "getblock")]
    fn get_block(&self, hash: BlockHash, verbosity: Option<u8>) -> Result<Value>;
    #[rpc(name = "gettxout", returns = "TxOut")]
    fn get_tx_out(&self, tx_id: Txid, outpoint: u32) -> Result<Value>;
    #[rpc(name = "stop")]
    fn stop(&self) -> Result<bool>;
    #[rpc(name = "addnode")]
    fn add_node(&self, node: String) -> Result<bool>;
}

pub struct RpcImpl {
    block_filter_storage: Option<Arc<NetworkFilters<KvFilterStore>>>,
    network: Network,
    chain: Arc<ChainState<KvChainStore<'static>>>,
    wallet: Arc<RwLock<AddressCache<KvDatabase>>>,
    node: Arc<NodeInterface>,
    kill_signal: Arc<RwLock<bool>>,
}

impl Rpc for RpcImpl {
    fn get_tx_out(&self, tx_id: Txid, outpoint: u32) -> Result<Value> {
        fn has_input(block: &Block, expected_input: OutPoint) -> bool {
            block.txdata.iter().any(|tx| {
                tx.input
                    .iter()
                    .any(|input| input.previous_output == expected_input)
            })
        }
        // can't proceed without block filters
        if self.block_filter_storage.is_none() {
            return Err(jsonrpc_core::Error {
                code: Error::NoBlockFilters.into(),
                message: Error::NoBlockFilters.to_string(),
                data: None,
            });
        }
        // this variable will be set to the UTXO iff (i) it have been created
        // (ii) it haven't been spent
        let mut txout = None;
        let tip = self.chain.get_height().unwrap();

        if let Some(ref cfilters) = self.block_filter_storage {
            let vout = OutPoint {
                txid: tx_id,
                vout: outpoint,
            };

            let filter_outpoint = bitcoin::consensus::serialize(&vout);
            let filter_txid = bitcoin::consensus::serialize(&tx_id);

            let candidates = cfilters.match_any(
                vec![filter_outpoint.as_slice(), filter_txid.as_slice()],
                1,
                tip,
                self.chain.clone(),
            );

            let candidates = candidates.into_iter().map(|hash| self.node.get_block(hash));

            for candidate in candidates {
                let candidate = match candidate {
                    Err(e) => {
                        return Err(jsonrpc_core::Error {
                            code: Error::Node.into(),
                            message: format!("error while downloading block {candidate:?}"),
                            data: Some(jsonrpc_core::Value::String(e.to_string())),
                        });
                    }
                    Ok(None) => {
                        return Err(jsonrpc_core::Error {
                            code: Error::Node.into(),
                            message: format!("BUG: block {candidate:?} is a match in our filters, but we can't get it?"),
                            data: None,
                        });
                    }
                    Ok(Some(candidate)) => candidate,
                };

                if let Some(tx) = candidate.txdata.iter().position(|tx| tx.txid() == tx_id) {
                    txout = candidate.txdata[tx].output.get(outpoint as usize).cloned();
                }

                if has_input(&candidate, vout) {
                    txout = None;
                }
            }
        }
        match txout {
            Some(txout) => Ok(json!({ "txout": txout })),
            None => Ok(json!({})),
        }
    }
    fn get_height(&self) -> Result<u32> {
        Ok(self.chain.get_best_block().unwrap().0)
    }

    fn get_block_filter(&self, heigth: u32) -> Result<String> {
        if let Some(ref cfilters) = self.block_filter_storage {
            return Ok(serialize_hex(
                &cfilters
                    .get_filter(heigth)
                    .ok_or(Error::BlockNotFound)?
                    .content,
            ));
        }
        Err(jsonrpc_core::Error {
            code: Error::NoBlockFilters.into(),
            message: Error::NoBlockFilters.to_string(),
            data: None,
        })
    }
    fn add_node(&self, node: String) -> Result<bool> {
        let node = node.split(':').collect::<Vec<&str>>();
        let (ip, port) = if node.len() == 2 {
            (node[0], node[1].parse().map_err(|_| Error::InvalidPort)?)
        } else {
            match self.network {
                Network::Bitcoin => (node[0], 8333),
                Network::Testnet => (node[0], 18333),
                Network::Regtest => (node[0], 18444),
                Network::Signet => (node[0], 38333),
                _ => return Err(Error::InvalidNetwork.into()),
            }
        };
        let node = ip.parse().map_err(|_| Error::InvalidAddress)?;
        self.node.connect(node, port).unwrap();
        Ok(true)
    }

    fn get_blockchain_info(&self) -> Result<GetBlockchainInfoRes> {
        let (height, hash) = self.chain.get_best_block().unwrap();
        let validated = self.chain.get_validation_index().unwrap();
        let ibd = self.chain.is_in_idb();
        let latest_header = self.chain.get_block_header(&hash).unwrap();
        let latest_work = latest_header.work();
        let latest_block_time = latest_header.time;
        let leaf_count = self.chain.acc().leaves as u32;
        let root_count = self.chain.acc().roots.len() as u32;
        let root_hashes = self
            .chain
            .acc()
            .roots
            .into_iter()
            .map(|r| r.to_string())
            .collect();
        let validated_blocks = self.chain.get_validation_index().unwrap();
        Ok(GetBlockchainInfoRes {
            best_block: hash.to_string(),
            height,
            ibd,
            validated,
            latest_work: latest_work.to_string(),
            latest_block_time,
            leaf_count,
            root_count,
            root_hashes,
            chain: self.network.to_string(),
            difficulty: latest_header.difficulty() as u64,
            progress: validated_blocks as f32 / height as f32,
        })
    }

    fn get_roots(&self) -> Result<Vec<String>> {
        let hashes = self.chain.get_root_hashes();
        return Ok(hashes.iter().map(|h| h.to_string()).collect());
    }

    fn get_block_hash(&self, height: u32) -> Result<BlockHash> {
        if let Ok(hash) = self.chain.get_block_hash(height) {
            return Ok(hash);
        }
        Err(Error::BlockNotFound.into())
    }

    fn get_block_header(&self, hash: BlockHash) -> Result<BlockHeader> {
        if let Ok(header) = self.chain.get_block_header(&hash) {
            return Ok(header);
        }
        Err(Error::BlockNotFound.into())
    }

    fn get_transaction(&self, tx_id: Txid, verbosity: Option<bool>) -> Result<Value> {
        let wallet = block_on(self.wallet.read());
        if verbosity == Some(true) {
            if let Some(tx) = wallet.get_transaction(&tx_id) {
                return Ok(serde_json::to_value(serialize(&tx.tx)).unwrap());
            }
            return Err(Error::TxNotFound.into());
        }
        if let Some(tx) = wallet.get_transaction(&tx_id) {
            return Ok(serde_json::to_value(self.make_raw_transaction(tx)).unwrap());
        }
        Err(Error::TxNotFound.into())
    }

    fn load_descriptor(&self, descriptor: String, rescan: Option<u32>) -> Result<bool> {
        let wallet = block_on(self.wallet.write());
        let result = wallet.push_descriptor(&descriptor);
        if let Some(rescan) = rescan {
            self.chain
                .rescan(rescan)
                .map_err(|_| jsonrpc_core::Error::internal_error())?;
        }
        if result.is_err() {
            return Err(Error::InvalidDescriptor.into());
        }
        Ok(true)
    }

    fn rescan(&self, rescan: u32) -> Result<bool> {
        let result = self.chain.rescan(rescan);
        if result.is_err() {
            return Err(Error::Chain.into());
        }
        Ok(true)
    }

    fn send_raw_transaction(&self, tx: String) -> Result<Txid> {
        let tx_hex = Vec::from_hex(&tx).map_err(|_| jsonrpc_core::Error {
            code: 3.into(),
            message: "Invalid hex".into(),
            data: None,
        })?;
        let tx = deserialize(&tx_hex).map_err(|e| jsonrpc_core::Error {
            code: 2.into(),
            message: format!("{:?}", e),
            data: None,
        })?;
        if self.chain.broadcast(&tx).is_ok() {
            return Ok(tx.txid());
        }
        Err(Error::Chain.into())
    }

    fn get_tx_proof(&self, tx_id: Txid) -> Result<Vec<String>> {
        if let Some((proof, _)) = block_on(self.wallet.read()).get_merkle_proof(&tx_id) {
            return Ok(proof);
        }
        Err(Error::TxNotFound.into())
    }

    fn get_block(&self, hash: BlockHash, verbosity: Option<u8>) -> Result<Value> {
        let verbosity = verbosity.unwrap_or(1);

        let block = if self.chain.get_block_hash(0).unwrap().eq(&hash) {
            Some(genesis_block(self.network))
        } else {
            self.node.get_block(hash).map_err(|_| Error::Chain)?
        };

        if let Some(block) = block {
            if verbosity == 1 {
                let tip = self.chain.get_height().map_err(|_| Error::Chain)?;
                let height = self
                    .chain
                    .get_block_height(&hash)
                    .map_err(|_| Error::Chain)?
                    .unwrap();

                let median_time_past = if height > 11 {
                    let mut last_block_times: Vec<_> = ((height - 11)..height)
                        .map(|h| {
                            self.chain
                                .get_block_header(&self.chain.get_block_hash(h).unwrap())
                                .unwrap()
                                .time
                        })
                        .collect();
                    last_block_times.sort();
                    last_block_times[5]
                } else {
                    block.header.time
                };

                let block = BlockJson {
                    bits: serialize_hex(&block.header.bits),
                    chainwork: block.header.work().to_string(),
                    confirmations: (tip - height) + 1,
                    difficulty: block.header.difficulty(),
                    hash: block.header.block_hash().to_string(),
                    height,
                    merkleroot: block.header.merkle_root.to_string(),
                    nonce: block.header.nonce,
                    previousblockhash: block.header.prev_blockhash.to_string(),
                    size: block.total_size(),
                    time: block.header.time,
                    tx: block
                        .txdata
                        .iter()
                        .map(|tx| tx.txid().to_string())
                        .collect(),
                    version: block.header.version.to_consensus(),
                    version_hex: serialize_hex(&block.header.version),
                    weight: block.weight().to_wu() as usize,
                    mediantime: median_time_past,
                    n_tx: block.txdata.len(),
                    nextblockhash: self
                        .chain
                        .get_block_hash(height + 1)
                        .ok()
                        .map(|h| h.to_string()),
                    strippedsize: block.strippedsize(),
                };

                return Ok(serde_json::to_value(block).unwrap());
            }
            return Ok(json!(serialize(&block).to_vec()));
        }
        Err(Error::BlockNotFound.into())
    }

    fn get_peer_info(&self) -> Result<Vec<PeerInfo>> {
        let peers = self.node.get_peer_info();
        if let Ok(peers) = peers {
            return Ok(peers);
        }
        Err(Error::TxNotFound.into())
    }

    fn stop(&self) -> Result<bool> {
        *async_std::task::block_on(self.kill_signal.write()) = true;
        Ok(true)
    }
}

impl RpcImpl {
    fn make_vin(&self, input: TxIn) -> TxInJson {
        let txid = serialize_hex(&input.previous_output.txid);
        let vout = input.previous_output.vout;
        let sequence = input.sequence.0;
        TxInJson {
            txid,
            vout,
            script_sig: ScriptSigJson {
                asm: input.script_sig.to_asm_string(),
                hex: input.script_sig.to_hex_string(),
            },
            witness: input
                .witness
                .iter()
                .map(|w| w.to_hex_string(bitcoin::hex::Case::Upper))
                .collect(),
            sequence,
        }
    }
    fn get_script_type(script: ScriptBuf) -> Option<&'static str> {
        if script.is_p2pkh() {
            return Some("p2pkh");
        }
        if script.is_p2sh() {
            return Some("p2sh");
        }
        if script.is_p2wpkh() {
            return Some("v0_p2wpkh");
        }
        if script.is_p2wsh() {
            return Some("v0_p2wsh");
        }
        None
    }
    fn make_vout(&self, output: TxOut, n: u32) -> TxOutJson {
        let value = output.value;
        TxOutJson {
            value: value.to_sat(),
            n,
            script_pub_key: ScriptPubKeyJson {
                asm: output.script_pubkey.to_asm_string(),
                hex: output.script_pubkey.to_hex_string(),
                req_sigs: 0, // This field is deprecated
                address: Address::from_script(&output.script_pubkey, self.network)
                    .map(|a| a.to_string())
                    .unwrap(),
                type_: Self::get_script_type(output.script_pubkey)
                    .unwrap_or("nonstandard")
                    .to_string(),
            },
        }
    }
    fn make_raw_transaction(&self, tx: CachedTransaction) -> RawTxJson {
        let raw_tx = tx.tx;
        let in_active_chain = tx.height != 0;
        let hex = serialize_hex(&raw_tx);
        let txid = serialize_hex(&raw_tx.txid());
        let block_hash = self
            .chain
            .get_block_hash(tx.height)
            .unwrap_or(BlockHash::all_zeros());
        let tip = self.chain.get_height().unwrap();
        let confirmations = if in_active_chain {
            tip - tx.height + 1
        } else {
            0
        };

        RawTxJson {
            in_active_chain,
            hex,
            txid,
            hash: serialize_hex(&raw_tx.wtxid()),
            size: raw_tx.total_size() as u32,
            vsize: raw_tx.vsize() as u32,
            weight: raw_tx.weight().to_wu() as u32,
            version: raw_tx.version.0 as u32,
            locktime: raw_tx.lock_time.to_consensus_u32(),
            vin: raw_tx
                .input
                .iter()
                .map(|input| self.make_vin(input.clone()))
                .collect(),
            vout: raw_tx
                .output
                .into_iter()
                .enumerate()
                .map(|(i, output)| self.make_vout(output, i as u32))
                .collect(),
            blockhash: serialize_hex(&block_hash),
            confirmations,
            blocktime: self
                .chain
                .get_block_header(&block_hash)
                .map(|h| h.time)
                .unwrap_or(0),
            time: self
                .chain
                .get_block_header(&block_hash)
                .map(|h| h.time)
                .unwrap_or(0),
        }
    }
    fn get_port(net: &Network) -> u16 {
        match net {
            Network::Bitcoin => 8332,
            Network::Testnet => 18332,
            Network::Signet => 38332,
            Network::Regtest => 18442,
            _ => 8332,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        chain: Arc<ChainState<KvChainStore<'static>>>,
        wallet: Arc<RwLock<AddressCache<KvDatabase>>>,
        node: Arc<NodeInterface>,
        kill_signal: Arc<RwLock<bool>>,
        network: Network,
        block_filter_storage: Option<Arc<NetworkFilters<KvFilterStore>>>,
        address: Option<SocketAddr>,
    ) -> jsonrpc_http_server::Server {
        let mut io = jsonrpc_core::IoHandler::new();
        let rpc_impl = RpcImpl {
            network,
            chain,
            wallet,
            node,
            kill_signal,
            block_filter_storage,
        };
        io.extend_with(rpc_impl.to_delegate());
        let address = address.unwrap_or_else(|| {
            format!("127.0.0.1:{}", Self::get_port(&network))
                .parse()
                .unwrap()
        });
        info!("Starting JSON-RPC server on {:?}", address);
        ServerBuilder::new(io)
            .threads(1)
            .start_http(&address)
            .unwrap()
    }
}

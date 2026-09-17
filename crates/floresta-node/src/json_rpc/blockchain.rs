// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::Address;
use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::MerkleBlock;
use bitcoin::Network;
use bitcoin::OutPoint;
use bitcoin::Script;
use bitcoin::ScriptBuf;
use bitcoin::Txid;
use bitcoin::VarInt;
use bitcoin::block::Header;
use bitcoin::consensus::Encodable;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::constants::genesis_block;
use bitcoin::hashes::Hash;
use bitcoin::hex::DisplayHex;
use corepc_types::ScriptPubKey;
use corepc_types::v29::GetTxOut;
use corepc_types::v30::DeploymentInfo;
use corepc_types::v30::GetBlockHeaderVerbose;
use corepc_types::v30::GetBlockVerboseOne;
use corepc_types::v30::GetBlockchainInfo;
use corepc_types::v30::GetDeploymentInfo;
use floresta_chain::buried_deployments_for;
use floresta_chain::extensions::HeaderExt;
use floresta_chain::extensions::WorkExt;
use floresta_wire::node_interface::ChainMethods;
use miniscript::descriptor::checksum;
use serde_json::Value;
use serde_json::json;
use tracing::debug;

use super::res::GetBlockHeaderRes;
use super::res::GetTxOutProof;
use super::res::jsonrpc_interface::JsonRpcError;
use super::server::RpcChain;
use super::server::RpcImpl;
use crate::json_rpc::res::GetBlockRes;
use crate::json_rpc::res::RescanConfidence;
use crate::json_rpc::server::SERIALIZATION_EXPECT_MSG;
use crate::json_rpc::server::to_core_asm_string;

impl<Blockchain: RpcChain> RpcImpl<Blockchain> {
    async fn get_block_inner(&self, hash: BlockHash) -> Result<Block, JsonRpcError> {
        let is_genesis = self
            .chain
            .get_block_hash(0)
            .map_err(|_| JsonRpcError::Chain)?
            .eq(&hash);

        if is_genesis {
            return Ok(genesis_block(self.network));
        }

        // Verify the block header is known before requesting the full block
        // from the network, otherwise the request will hang indefinitely.
        self.get_block_header_inner(hash)?;

        self.node
            .get_block(hash)
            .await
            .map_err(|e| JsonRpcError::Node(e.to_string()))
            .and_then(|block| block.ok_or(JsonRpcError::BlockNotFound))
    }

    /// Return the block that contains the given Txid
    pub fn get_block_by_txid(&self, txid: &Txid) -> Result<Block, JsonRpcError> {
        let height = self
            .wallet
            .get_height(txid)
            .ok_or(JsonRpcError::TxNotFound)?;
        let blockhash = self
            .chain
            .get_block_hash(height)
            .map_err(|_| JsonRpcError::BlockNotFound)?;

        self.chain
            .get_block(&blockhash)
            .map_err(|_| JsonRpcError::BlockNotFound)
    }

    pub fn get_rescan_interval(
        &self,
        use_timestamp: bool,
        start: u32,
        stop: u32,
        confidence: RescanConfidence,
    ) -> Result<(u32, u32), JsonRpcError> {
        if use_timestamp {
            let start_height = self.get_block_height_by_timestamp(start, &confidence)?;

            let stop_height = self.get_block_height_by_timestamp(stop, &RescanConfidence::Exact)?;

            return Ok((start_height, stop_height));
        }

        let (tip, _) = self
            .chain
            .get_best_block()
            .map_err(|_| JsonRpcError::Chain)?;

        if stop > tip {
            return Err(JsonRpcError::InvalidRescanVal);
        }

        Ok((start, stop))
    }

    /// Retrieves the height of the block that was mined in the given timestamp.
    ///
    /// `timestamp` has an alias, 0 will directly refer to the network's genesis timestamp.
    pub fn get_block_height_by_timestamp(
        &self,
        timestamp: u32,
        confidence: &RescanConfidence,
    ) -> Result<u32, JsonRpcError> {
        /// Simple helper to avoid code reuse.
        fn get_block_time<BlockChain: RpcChain>(
            provider: &RpcImpl<BlockChain>,
            at: u32,
        ) -> Result<u32, JsonRpcError> {
            let hash = provider.get_block_hash(at)?;
            let block = provider.get_block_header_inner(hash)?;
            Ok(block.time)
        }

        let genesis_timestamp = genesis_block(self.network).header.time;

        if timestamp == 0 || timestamp == genesis_timestamp {
            return Ok(0);
        };

        let (tip_height, _) = self
            .chain
            .get_best_block()
            .map_err(|_| JsonRpcError::BlockNotFound)?;

        let tip_time = get_block_time(self, tip_height)?;

        if timestamp < genesis_timestamp || timestamp > tip_time {
            return Err(JsonRpcError::InvalidTimestamp);
        }

        let adjusted_target = timestamp.saturating_sub(confidence.as_secs());

        let mut high = tip_height;
        let mut low = 0;
        let max_iters = tip_height.ilog2() + 1;
        for _ in 0..max_iters {
            let cut = (high + low) / 2;

            let block_timestamp = get_block_time(self, cut)?;

            if block_timestamp == adjusted_target {
                debug!("found a precise block; returning {cut}");
                return Ok(cut);
            }

            if high - low <= 2 {
                debug!("didn't find a precise block; returning {low}");
                return Ok(low);
            }

            if block_timestamp > adjusted_target {
                high = cut;
            } else {
                low = cut;
            }
        }

        // This is pretty much unreachable.
        Err(JsonRpcError::BlockNotFound)
    }
}

// blockchain rpcs
impl<Blockchain: RpcChain> RpcImpl<Blockchain> {
    // dumputxoutset

    // getbestblockhash
    pub(super) fn get_best_block_hash(&self) -> Result<BlockHash, JsonRpcError> {
        Ok(self
            .chain
            .get_best_block()
            .map_err(|_| JsonRpcError::Chain)?
            .1)
    }

    // getblock
    pub(super) async fn get_block(
        &self,
        hash: BlockHash,
        verbosity: u8,
    ) -> Result<GetBlockRes, JsonRpcError> {
        let block = self.get_block_inner(hash).await?;

        if verbosity == 0 {
            let hex = serialize_hex(&block);

            return Ok(GetBlockRes::Zero(hex));
        }
        if verbosity == 1 {
            let header_fields = self.get_block_header_verbose_inner(&block)?;

            // Stripped size is the size of the block without witness data
            // Header + VarInt for number of transactions + sum of base sizes of each transaction
            let tx_count_varint_size = VarInt::from(block.txdata.len()).size();
            let total_tx_base_size: usize = block.txdata.iter().map(|tx| tx.base_size()).sum();
            let stripped_size_bytes = Header::SIZE + tx_count_varint_size + total_tx_base_size;

            let stripped_size = Some(stripped_size_bytes.try_into()?);

            let tx = block
                .txdata
                .iter()
                .map(|tx| tx.compute_txid().to_string())
                .collect();

            let block = GetBlockVerboseOne {
                bits: header_fields.bits,
                chain_work: header_fields.chain_work,
                confirmations: header_fields.confirmations,
                difficulty: header_fields.difficulty,
                hash: header_fields.hash,
                height: header_fields.height,
                merkle_root: header_fields.merkle_root,
                nonce: header_fields.nonce,
                previous_block_hash: header_fields.previous_block_hash,
                size: block.total_size().try_into()?,
                time: header_fields.time,
                tx,
                version: header_fields.version,
                version_hex: header_fields.version_hex,
                weight: block.weight().to_wu(),
                median_time: Some(header_fields.median_time),
                n_tx: header_fields.n_tx.into(),
                next_block_hash: header_fields.next_block_hash,
                stripped_size,
                target: header_fields.target,
            };

            return Ok(GetBlockRes::One(Box::new(block)));
        }
        Err(JsonRpcError::InvalidVerbosityLevel)
    }

    // getblockchaininfo
    //
    // `headers` tracks the best-known header tip; `blocks` tracks the validated
    // tip. They can diverge mid-IBD and coincide once sync completes.
    pub(super) fn get_blockchain_info(&self) -> Result<GetBlockchainInfo, JsonRpcError> {
        let (height, hash) = self
            .chain
            .get_best_block()
            .map_err(|_| JsonRpcError::Chain)?;
        let validated = self
            .chain
            .get_validation_index()
            .map_err(|_| JsonRpcError::Chain)?;
        let initial_block_download = self.chain.is_in_ibd();
        let latest_header = self
            .chain
            .get_block_header(&hash)
            .map_err(|_| JsonRpcError::Chain)?;
        let chain_work = latest_header
            .calculate_chain_work(&self.chain)?
            .to_string_hex();

        let verification_progress = self.verification_progress(validated, &latest_header)?;

        let blocks = i64::from(validated);
        let headers = i64::from(height);
        let best_block_hash = hash.to_string();
        let bits = latest_header.get_bits_hex();
        let target = latest_header.get_target_hex();
        let difficulty = latest_header.get_difficulty();
        let time = i64::from(latest_header.time);
        let median_time = i64::from(latest_header.calculate_median_time_past(&self.chain)?);
        let size_on_disk = self.chain.size_on_disk().map_err(|_| JsonRpcError::Chain)?;
        let prune_height = Some(blocks + 1);
        let warnings = self
            .chain
            .get_warnings()
            .iter()
            .map(ToString::to_string)
            .collect();

        let chain = match self.network {
            Network::Bitcoin => "main",
            Network::Testnet => "test",
            Network::Testnet4 => "testnet4",
            Network::Signet => "signet",
            Network::Regtest => "regtest",
        }
        .to_string();

        Ok(GetBlockchainInfo {
            chain,
            blocks,
            headers,
            best_block_hash,
            bits,
            target,
            difficulty,
            time,
            median_time,
            verification_progress,
            initial_block_download,
            chain_work,
            size_on_disk,
            pruned: true,
            prune_height,
            automatic_pruning: Some(true),
            prune_target_size: Some(0),
            signet_challenge: None,
            warnings,
        })
    }

    // getblockcount
    pub(super) fn get_block_count(&self) -> Result<u32, JsonRpcError> {
        self.chain.get_height().map_err(|_| JsonRpcError::Chain)
    }

    // getblockfilter
    // getblockfrompeer (just call getblock)

    // getblockhash
    pub(super) fn get_block_hash(&self, height: u32) -> Result<BlockHash, JsonRpcError> {
        self.chain
            .get_block_hash(height)
            .map_err(|_| JsonRpcError::BlockNotFound)
    }

    // getblockheader
    pub(super) async fn get_block_header(
        &self,
        hash: BlockHash,
        verbosity: bool,
    ) -> Result<GetBlockHeaderRes, JsonRpcError> {
        let header = self.get_block_header_inner(hash)?;

        if !verbosity {
            let hex = serialize_hex(&header);
            return Ok(GetBlockHeaderRes::Raw(hex));
        }

        let block = self.get_block_inner(hash).await?;

        let get_block_header = self.get_block_header_verbose_inner(&block)?;

        Ok(GetBlockHeaderRes::Verbose(Box::new(get_block_header)))
    }

    // getblockstats
    // getchainstates
    // getchaintips
    // getchaintxstats

    // getdeploymentinfo
    pub(super) fn get_deployment_info(
        &self,
        hash: Option<BlockHash>,
    ) -> Result<GetDeploymentInfo, JsonRpcError> {
        let tip = self
            .chain
            .get_best_block()
            .map_err(|_| JsonRpcError::Chain)?
            .1;
        let target_hash = hash.unwrap_or(tip);

        let height = self
            .chain
            .get_block_height(&target_hash)
            .map_err(|_| JsonRpcError::Chain)?
            .ok_or(JsonRpcError::BlockNotFound)?;

        let mut deployments = BTreeMap::new();

        for &(name, activation_height) in buried_deployments_for(self.network) {
            deployments.insert(
                name.to_string(),
                DeploymentInfo {
                    deployment_type: "buried".to_string(),
                    height: Some(activation_height),
                    active: height >= activation_height,
                    bip9: None,
                },
            );
        }

        Ok(GetDeploymentInfo {
            hash: target_hash.to_string(),
            height,
            deployments,
        })
    }

    // getdifficulty
    pub(super) fn get_difficulty(&self) -> Result<f64, JsonRpcError> {
        let (_, hash) = self
            .chain
            .get_best_block()
            .map_err(|_| JsonRpcError::Chain)?;
        let header = self
            .chain
            .get_block_header(&hash)
            .map_err(|_| JsonRpcError::BlockNotFound)?;
        Ok(header.difficulty_float())
    }

    // getmempoolancestors
    // getmempooldescendants
    // getmempoolentry
    // getmempoolinfo
    // getrawmempool

    /// Same as `get_block_header_inner` but verbose.
    fn get_block_header_verbose_inner(
        &self,
        block: &Block,
    ) -> Result<GetBlockHeaderVerbose, JsonRpcError> {
        let header = &block.header;
        let height = header.get_height(&self.chain)?;
        let median_time = header.calculate_median_time_past(&self.chain)?;
        let chain_work = header.calculate_chain_work(&self.chain)?.to_string_hex();
        let confirmations = header.get_confirmations(&self.chain)?;
        let version_hex = header.get_version_hex();

        let next_block_hash = header
            .get_next_block_hash(&self.chain)?
            .map(|h| h.to_string());

        let bits = header.get_bits_hex();
        let difficulty = header.get_difficulty();
        let target = header.get_target_hex();
        let previous_block_hash = (header.prev_blockhash != BlockHash::all_zeros())
            .then_some(header.prev_blockhash.to_string());

        Ok(GetBlockHeaderVerbose {
            bits,
            chain_work,
            confirmations: confirmations.into(),
            difficulty,
            hash: header.block_hash().to_string(),
            height: height.into(),
            median_time: median_time.into(),
            next_block_hash,
            version: header.version.to_consensus(),
            version_hex,
            previous_block_hash,
            merkle_root: header.merkle_root.to_string(),
            time: header.time.into(),
            target,
            nonce: header.nonce.into(),
            n_tx: block.txdata.len().try_into()?,
        })
    }

    /// Helper method to get a block header by its hash, used by multiple rpcs.
    fn get_block_header_inner(&self, hash: BlockHash) -> Result<Header, JsonRpcError> {
        self.chain
            .get_block_header(&hash)
            .map_err(|_| JsonRpcError::BlockNotFound)
    }

    /// Check if the script is anchor type
    fn is_anchor_type(script: &Script) -> bool {
        script.as_bytes().starts_with(&[0x51, 0x02, 0x4e, 0x73])
    }

    /// Returns a label about the scriptPubKey type
    /// (pubkey, pubkeyhash, multisig, nulldata, scripthash, witness_v0_keyhash, witness_v0_scripthash, witness_v1_taproot, anchor, nonstandard)
    pub(super) fn get_script_type_label(script: &Script) -> &'static str {
        if script.is_p2pk() {
            return "pubkey";
        }

        if script.is_p2pkh() {
            return "pubkeyhash";
        }

        if script.is_multisig() {
            return "multisig";
        }

        if script.is_op_return() {
            return "nulldata";
        }

        if script.is_p2sh() {
            return "scripthash";
        }

        if script.is_p2wpkh() {
            return "witness_v0_keyhash";
        }

        if script.is_p2wsh() {
            return "witness_v0_scripthash";
        }

        if script.is_p2tr() {
            return "witness_v1_taproot";
        }

        if Self::is_anchor_type(script) {
            return "anchor";
        }

        "nonstandard"
    }

    /// TODO: This function is not compliant with Bitcoin Core.
    /// See: <https://github.com/getfloresta/Floresta/issues/987>
    pub(super) fn get_script_type_descriptor(script: &Script, address: &Option<Address>) -> String {
        // Try script from the address
        if let Some(addr) = address {
            if script.is_p2pk() {
                return format!("pk({addr})");
            }

            return format!("addr({addr})");
        }

        if script.is_op_return() {
            let hex = script.to_hex_string();
            return format!("raw({hex})");
        }

        let hex = script.to_hex_string();
        format!("raw({hex})")
    }

    /// gettxout: returns details about an unspent transaction output.
    pub(super) fn get_tx_out(
        &self,
        txid: Txid,
        outpoint: u32,
        _include_mempool: bool,
    ) -> Result<Option<GetTxOut>, JsonRpcError> {
        let res = match (
            self.wallet.get_transaction(&txid),
            self.wallet.get_height(&txid),
            self.wallet.get_utxo(&OutPoint {
                txid,
                vout: outpoint,
            }),
        ) {
            (Some(cached_tx), Some(height), Some(txout)) => {
                let is_coinbase = cached_tx.tx.is_coinbase();
                let Ok((bestblock_height, bestblock_hash)) = self.chain.get_best_block() else {
                    return Err(JsonRpcError::BlockNotFound);
                };

                let script = txout.script_pubkey.as_script();
                let network = self.chain.get_params().network;
                let address = Address::from_script(script, network).ok();

                let base_descriptor = Self::get_script_type_descriptor(script, &address);
                let mut checksum_engine = checksum::Engine::new();
                let descriptor: Option<String> = match checksum_engine.input(&base_descriptor) {
                    Ok(()) => Some(format!("{base_descriptor}#{}", checksum_engine.checksum())),
                    Err(_) => None,
                };

                let asm = to_core_asm_string(&txout.script_pubkey, false);
                let script_pubkey = ScriptPubKey {
                    asm,
                    hex: txout.script_pubkey.to_hex_string(),
                    descriptor,
                    address: address.as_ref().map(ToString::to_string),
                    type_: Self::get_script_type_label(script).to_string(),
                    // Deprecated in Bitcoin Core v22, require flags in Bitcoin Core.
                    // Set to None as not required for consensus.
                    addresses: None,
                    required_signatures: None,
                };

                Some(GetTxOut {
                    best_block: bestblock_hash.to_string(),
                    confirmations: bestblock_height - height + 1,
                    value: txout.value.to_btc(),
                    script_pubkey,
                    coinbase: is_coinbase,
                })
            }
            _ => None,
        };
        Ok(res)
    }

    /// Computes the necessary information for the RPC `gettxoutproof [txids] blockhash (optional)`
    ///
    /// This function has two paths, when blockhash is inserted and when isn't.
    ///
    /// Specifying the blockhash will make this function go after the block and search
    /// for the transactions inside it, building a merkle proof from the block with its
    /// indexes. Not specifying will redirect it to search for the merkle proof on our
    /// watch-only wallet which may not have the transaction cached.
    ///
    /// Not finding one of the specified transactions will raise [`JsonRpcError::TxNotFound`].
    pub(super) async fn get_txout_proof(
        &self,
        tx_ids: &[Txid],
        blockhash: Option<BlockHash>,
    ) -> Result<GetTxOutProof, JsonRpcError> {
        let block = match blockhash {
            Some(blockhash) => self.get_block_inner(blockhash).await?,
            // Using the first Txid to get the block should be fine since they are expected to all
            // live in the same block, otherwise, theres no way they have a common proof.
            None => self.get_block_by_txid(&tx_ids[0])?,
        };

        // Before building the merkle block we try to remove all txids
        // that aren't present in the block we found, meaning that
        // at least one of the txids doesn't belong to the block which
        // in case should make the command fail.
        //
        // this makes the use MerkleBlock::from_block_with_predicate useless.
        let targeted_txids: Vec<Txid> = block
            .txdata
            .iter()
            .filter_map(|tx| {
                let txid = tx.compute_txid();
                if tx_ids.contains(&txid) {
                    Some(txid)
                } else {
                    None
                }
            })
            .collect();

        if targeted_txids.len() != tx_ids.len() {
            return Err(JsonRpcError::TxNotFound);
        };

        let merkle_block = MerkleBlock::from_block_with_predicate(&block, |tx| tx_ids.contains(tx));
        let mut bytes: Vec<u8> = Vec::new();
        merkle_block
            .consensus_encode(&mut bytes)
            .expect("This will raise if a writer error happens");
        Ok(GetTxOutProof(bytes.to_lower_hex_string()))
    }

    // gettxoutsetinfo
    // gettxspendigprevout
    // importmempool
    // loadtxoutset
    // preciousblock
    // pruneblockchain
    // savemempool
    // scanblocks
    // scantxoutset
    // verifychain
    // verifytxoutproof

    // floresta flavored rpcs. These are not part of the bitcoin rpc spec
    // findtxout
    pub(super) async fn find_tx_out(
        &self,
        txid: Txid,
        vout: u32,
        script: ScriptBuf,
        height: u32,
    ) -> Result<Value, JsonRpcError> {
        if let Some(txout) = self.wallet.get_utxo(&OutPoint { txid, vout }) {
            return Ok(serde_json::to_value(txout).expect(SERIALIZATION_EXPECT_MSG));
        }
        if self.chain.is_in_ibd() {
            return Err(JsonRpcError::InInitialBlockDownload);
        }

        self.wallet.cache_address(script.clone());
        let filters = self.filters.clone().ok_or(JsonRpcError::NoBlockFilters)?;
        Self::rescan_with_block_filters(
            vec![script],
            self.chain.clone(),
            self.wallet.clone(),
            filters,
            Some(height),
            None,
        )
        .await?;

        Ok(match self.get_tx_out(txid, vout, false)? {
            Some(txout) => json!(txout),
            None => json!({}),
        })
    }

    // getroots
    pub(super) fn get_roots(&self) -> Result<Vec<String>, JsonRpcError> {
        let hashes = self.chain.get_root_hashes();
        Ok(hashes.iter().map(|h| h.to_string()).collect())
    }

    pub(super) fn list_descriptors(&self) -> Result<Vec<String>, JsonRpcError> {
        let descriptors = self
            .wallet
            .get_descriptors()
            .map_err(|e| JsonRpcError::Wallet(e.to_string()))?;
        Ok(descriptors)
    }

    /// Time-based estimate of validated-tip progress in `[0.0, 1.0]`.
    ///
    /// Ratio of chain-time elapsed between genesis and the validated block,
    /// over chain-time between genesis and the later of `now` or the tip
    /// header's time. Returns `0.0` for cold start, clamps to `1.0` at tip.
    fn verification_progress(
        &self,
        validated: u32,
        latest_header: &Header,
    ) -> Result<f64, JsonRpcError> {
        let header_tip_time = latest_header.time;

        let validated_block_time = self
            .chain
            .get_block_hash(validated)
            .and_then(|hash| self.chain.get_block_header(&hash))
            .map_err(|_| JsonRpcError::Chain)?
            .time;

        let genesis_time = genesis_block(self.network).header.time;
        let now: u32 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| JsonRpcError::InvalidSystemTime)?
            .as_secs()
            .try_into()
            .map_err(|_| JsonRpcError::InvalidSystemTime)?;

        let elapsed_validated = validated_block_time.saturating_sub(genesis_time);
        let elapsed_total = now.max(header_tip_time).saturating_sub(genesis_time);

        // Cold start: the tip is still genesis and the clock has not reached
        // genesis time, so there is no span to measure against.
        if elapsed_total == 0 {
            return Ok(0.0);
        }

        Ok((f64::from(elapsed_validated) / f64::from(elapsed_total)).clamp(0.0, 1.0))
    }
}

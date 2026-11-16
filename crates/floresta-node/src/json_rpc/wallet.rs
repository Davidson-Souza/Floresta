// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use bitcoin::ScriptBuf;
use bitcoin::constants::genesis_block;
use floresta_compact_filters::flat_filters_store::FlatFiltersStore;
use floresta_compact_filters::network_filters::NetworkFilters;
use floresta_domain::node_interface::ChainMethods;
use floresta_rpc::rpc_interfaces::WalletRpc;
use floresta_rpc::rpc_types::RescanConfidence;
use floresta_watch_only::AddressCache;
use floresta_watch_only::kv_database::KvDatabase;
use floresta_wire::node_handle::NodeHandle;
use tracing::debug;
use tracing::info;

use super::server::RpcChain;
use super::server::RpcImpl;
use crate::json_rpc::res::jsonrpc_interface::JsonRpcError;

impl<Blockchain: RpcChain> WalletRpc for RpcImpl<Blockchain> {
    type Error = JsonRpcError;

    // rescanblockchain
    async fn rescan_blockchain(
        &self,
        start: Option<u32>,
        stop: Option<u32>,
        use_timestamp: Option<bool>,
        confidence: Option<RescanConfidence>,
    ) -> Result<bool, JsonRpcError> {
        let start = start.unwrap_or(0);
        let stop = stop.unwrap_or(0);
        let use_timestamp = use_timestamp.unwrap_or(false);
        let confidence = confidence.unwrap_or(RescanConfidence::Medium);
        let (start_height, stop_height) =
            self.get_rescan_interval(use_timestamp, start, stop, confidence)?;

        if stop_height != 0 && start_height >= stop_height {
            // When stop height is a non zero value it needs atleast to be greater than start_height.
            return Err(JsonRpcError::InvalidRescanVal);
        }

        // if we are on ibd, we don't have any filters to rescan
        if self.chain.is_in_ibd() {
            return Err(JsonRpcError::InInitialBlockDownload);
        }

        let addresses = self.wallet.get_cached_addresses()?;

        if addresses.is_empty() {
            return Err(JsonRpcError::NoAddressesToRescan);
        }

        let wallet = self.wallet.clone();

        let cfilters = self
            .block_filter_storage
            .as_ref()
            .ok_or(JsonRpcError::NoBlockFilters)?
            .clone();

        let node = self.node.clone();

        let chain = self.chain.clone();

        tokio::task::spawn(Self::rescan_with_block_filters(
            addresses,
            chain,
            wallet,
            cfilters,
            node,
            (start_height != 0).then_some(start_height), // Its ugly but to maintain the API here its necessary to recast to a Option.
            (stop_height != 0).then_some(stop_height),
        ));
        Ok(true)
    }

    // listdescriptors
    async fn list_descriptors(&self) -> Result<Vec<String>, JsonRpcError> {
        let descriptors = self
            .wallet
            .get_descriptors()
            .map_err(|e| JsonRpcError::Wallet(e.to_string()))?;
        Ok(descriptors)
    }

    // loaddescriptor
    async fn load_descriptor(&self, descriptor: String) -> Result<bool, JsonRpcError> {
        let addresses = self.wallet.push_descriptor(&descriptor)?;
        info!("Descriptor pushed: {descriptor}");
        debug!("Rescanning with block filters for addresses: {addresses:?}");

        let addresses = self.wallet.get_cached_addresses()?;
        let wallet = self.wallet.clone();
        let cfilters = self
            .block_filter_storage
            .as_ref()
            .ok_or(JsonRpcError::NoBlockFilters)?
            .clone();
        let node = self.node.clone();
        let chain = self.chain.clone();

        tokio::task::spawn(Self::rescan_with_block_filters(
            addresses, chain, wallet, cfilters, node, None, None,
        ));

        Ok(true)
    }
}

impl<Blockchain: RpcChain> RpcImpl<Blockchain> {
    async fn rescan_with_block_filters(
        addresses: Vec<ScriptBuf>,
        chain: Blockchain,
        wallet: Arc<AddressCache<KvDatabase>>,
        cfilters: Arc<NetworkFilters<FlatFiltersStore>>,
        node: NodeHandle,
        start_height: Option<u32>,
        stop_height: Option<u32>,
    ) -> Result<(), JsonRpcError> {
        let blocks = cfilters
            .match_any(
                addresses.iter().map(|a| a.as_bytes()).collect(),
                start_height,
                stop_height,
                chain.clone(),
            )
            .map_err(|e| JsonRpcError::Filters(e.to_string()))?;

        info!("rescan filter hits: {blocks:?}");

        for block in blocks {
            if let Ok(block) = node.get_block(block).await {
                let height = chain
                    .get_block_height(&block.block_hash())
                    .map_err(|_| JsonRpcError::Chain)?
                    .ok_or(JsonRpcError::BlockNotFound)?;

                wallet.block_process(&block, height)?;
            }
        }

        Ok(())
    }

    fn get_rescan_interval(
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
    fn get_block_height_by_timestamp(
        &self,
        timestamp: u32,
        confidence: &RescanConfidence,
    ) -> Result<u32, JsonRpcError> {
        /// Simple helper to avoid code reuse.
        fn get_block_time<BlockChain: RpcChain>(
            provider: &RpcImpl<BlockChain>,
            at: u32,
        ) -> Result<u32, JsonRpcError> {
            let hash = provider.get_block_hash_inner(at)?;
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

/// After a node catches-up with the network, we can start listening for new blocks, handing any
/// request our user might make and keep our peers alive. This mode requires way less bandwidth and
/// CPU to run, being bound by the number of blocks found in a given period.
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bitcoin::p2p::address::AddrV2;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::message_blockdata::Inventory;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::pruned_utreexo::partial_chain::PartialChainState;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use floresta_chain::UtreexoBlock;
use log::debug;
use log::error;
use log::info;
use log::warn;
use rustreexo::accumulator::stump::Stump;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::RwLock;
use tokio::time::timeout;

use super::error::WireError;
use super::peer::PeerMessages;
use crate::address_man::AddressState;
use crate::address_man::LocalAddress;
use crate::node::periodic_job;
use crate::node::try_and_log;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::RescanStatus;
use crate::node::UtreexoNode;
use crate::node_context::NodeContext;
use crate::node_interface::NodeInterface;
use crate::node_interface::NodeResponse;
use crate::node_interface::UserRequest;
use crate::p2p_wire::chain_selector::ChainSelector;
use crate::p2p_wire::sync_node::SyncNode;

#[derive(Debug, Clone)]
pub struct RunningNode {
    pub(crate) last_rescan_request: RescanStatus,
    pub(crate) last_feeler: Instant,
    pub(crate) last_address_rearrange: Instant,
    pub(crate) user_requests: Arc<NodeInterface>,
}

impl NodeContext for RunningNode {
    const REQUEST_TIMEOUT: u64 = 30;
    fn get_required_services(&self) -> ServiceFlags {
        ServiceFlags::UTREEXO
            | ServiceFlags::NETWORK
            | ServiceFlags::WITNESS
            | ServiceFlags::COMPACT_FILTERS
    }
}

impl<Chain> UtreexoNode<RunningNode, Chain>
where
    WireError: From<<Chain as BlockchainInterface>::Error>,
    Chain: BlockchainInterface + UpdatableChainstate + 'static,
{
    /// Returns a handle to the node interface that we can use to request data from our
    /// node. This struct is thread safe, so we can use it from multiple threads and have
    /// multiple handles. It also doesn't require a mutable reference to the node, or any
    /// synchronization mechanism.
    pub fn get_handle(&self) -> Arc<NodeInterface> {
        self.1.user_requests.clone()
    }

    #[allow(clippy::result_large_err)]
    fn check_request_timeout(&mut self) -> Result<(), SendError<NodeResponse>> {
        let mutex = self.1.user_requests.requests.lock().unwrap();
        let mut to_remove = Vec::new();
        for req in mutex.iter() {
            if req.time.elapsed() > Duration::from_secs(10) {
                to_remove.push(req.req);
            }
        }
        drop(mutex);
        for request in to_remove {
            self.1.user_requests.send_answer(request, None);
        }

        Ok(())
    }

    async fn handle_user_request(&mut self) {
        let mut requests = Vec::new();

        for request in self.1.user_requests.requests.lock().unwrap().iter() {
            if !self
                .inflight
                .contains_key(&InflightRequests::UserRequest(request.req))
            {
                requests.push(request.req);
            }
        }
        self.perform_user_request(requests).await;
    }

    fn handle_get_peer_info(&self) {
        let mut peers = Vec::new();
        for peer in self.peer_ids.iter() {
            peers.push(self.get_peer_info(peer));
        }
        let peers = peers.into_iter().flatten().collect();
        self.1.user_requests.send_answer(
            UserRequest::GetPeerInfo,
            Some(NodeResponse::GetPeerInfo(peers)),
        );
    }

    async fn perform_user_request(&mut self, user_req: Vec<UserRequest>) {
        for user_req in user_req {
            let req = match user_req {
                UserRequest::Block(block) => NodeRequest::GetBlock((vec![block], false)),
                UserRequest::UtreexoBlock(block) => NodeRequest::GetBlock((vec![block], true)),
                UserRequest::MempoolTransaction(txid) => NodeRequest::MempoolTransaction(txid),
                UserRequest::GetPeerInfo => {
                    self.handle_get_peer_info();
                    continue;
                }
                UserRequest::Connect((addr, port)) => {
                    let addr_v2 = match addr {
                        IpAddr::V4(addr) => AddrV2::Ipv4(addr),
                        IpAddr::V6(addr) => AddrV2::Ipv6(addr),
                    };
                    let local_addr = LocalAddress::new(
                        addr_v2,
                        0,
                        AddressState::NeverTried,
                        0.into(),
                        port,
                        self.peer_id_count as usize,
                    );
                    self.open_connection(false, 0, local_addr).await;
                    self.peer_id_count += 1;
                    self.1.user_requests.send_answer(
                        UserRequest::Connect((addr, port)),
                        Some(NodeResponse::Connect(true)),
                    );
                    continue;
                }
            };
            let peer = self.send_to_random_peer(req, ServiceFlags::NONE).await;
            if let Ok(peer) = peer {
                self.inflight.insert(
                    InflightRequests::UserRequest(user_req),
                    (peer, Instant::now()),
                );
            }
        }
    }

    async fn send_addresses(&mut self) -> Result<(), WireError> {
        let addresses = self
            .address_man
            .get_addresses_to_send()
            .into_iter()
            .map(|(addr, time, services, port)| AddrV2Message {
                services,
                addr,
                port,
                time: time as u32,
            })
            .collect();

        self.send_to_random_peer(NodeRequest::SendAddresses(addresses), ServiceFlags::NONE)
            .await?;
        Ok(())
    }

    async fn check_for_timeout(&mut self) -> Result<(), WireError> {
        let mut timed_out = Vec::new();
        for request in self.inflight.keys() {
            let (_, time) = self.inflight.get(request).unwrap();
            if time.elapsed() > Duration::from_secs(RunningNode::REQUEST_TIMEOUT) {
                timed_out.push(request.clone());
                debug!("Request {:?} timed out", request);
            }
        }

        for request in timed_out {
            let Some((peer, _)) = self.inflight.remove(&request) else {
                warn!(
                    "POSSIBLE BUG: Request {:?} timed out, but it wasn't in the inflight list",
                    request
                );
                continue;
            };

            if !matches!(request, InflightRequests::Connect(_)) {
                // Punnishing this peer for taking too long to respond
                self.increase_banscore(peer, 2).await?;
            }

            match request {
                InflightRequests::UtreexoState(_) => {}
                InflightRequests::Blocks(block) => {
                    if !self.has_utreexo_peers() {
                        continue;
                    }
                    let peer = self
                        .send_to_random_peer(
                            NodeRequest::GetBlock((vec![block], true)),
                            ServiceFlags::UTREEXO,
                        )
                        .await?;
                    self.inflight
                        .insert(InflightRequests::Blocks(block), (peer, Instant::now()));
                }
                InflightRequests::RescanBlock(block) => {
                    let peer = self
                        .send_to_random_peer(
                            NodeRequest::GetBlock((vec![block], false)),
                            ServiceFlags::NONE,
                        )
                        .await?;
                    self.inflight
                        .insert(InflightRequests::RescanBlock(block), (peer, Instant::now()));
                }
                InflightRequests::Headers => {
                    let peer = self
                        .send_to_random_peer(NodeRequest::GetAddresses, ServiceFlags::NONE)
                        .await?;
                    self.last_headers_request = Instant::now();
                    self.inflight
                        .insert(InflightRequests::Headers, (peer, Instant::now()));
                }
                InflightRequests::UserRequest(req) => match req {
                    UserRequest::Block(block) => {
                        let peer = self
                            .send_to_random_peer(
                                NodeRequest::GetBlock((vec![block], true)),
                                ServiceFlags::NONE,
                            )
                            .await?;
                        self.inflight
                            .insert(InflightRequests::UserRequest(req), (peer, Instant::now()));
                    }
                    UserRequest::MempoolTransaction(txid) => {
                        let peer = self
                            .send_to_random_peer(
                                NodeRequest::MempoolTransaction(txid),
                                ServiceFlags::NONE,
                            )
                            .await?;
                        self.inflight
                            .insert(InflightRequests::UserRequest(req), (peer, Instant::now()));
                    }
                    UserRequest::UtreexoBlock(block) => {
                        let peer = self
                            .send_to_random_peer(
                                NodeRequest::GetBlock((vec![block], true)),
                                ServiceFlags::NONE,
                            )
                            .await?;
                        self.inflight
                            .insert(InflightRequests::UserRequest(req), (peer, Instant::now()));
                    }
                    _ => {}
                },
                InflightRequests::Connect(peer) => {
                    self.send_to_peer(peer, NodeRequest::Shutdown).await?
                }
                InflightRequests::GetFilters => {
                    if let Some(ref block_filters) = self.block_filters {
                        let last_success = block_filters.get_height()? + 1;
                        self.last_filter = self.chain.get_block_hash(last_success)?;
                        self.download_filters().await?;
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn catch_up(self, kill_signal: Arc<RwLock<bool>>) -> Self {
        let mut sync = UtreexoNode::<SyncNode, Chain>(self.0, SyncNode::default());
        sync.run(kill_signal, |_| {}).await;

        UtreexoNode(sync.0, self.1)
    }

    pub async fn run(
        mut self,
        kill_signal: Arc<RwLock<bool>>,
        stop_signal: futures::channel::oneshot::Sender<()>,
    ) {
        try_and_log!(self.init_peers().await);
        let startup_tip = self.chain.get_height().unwrap();

        // Use this node state to Initial Block download
        let mut ibd = UtreexoNode(self.0, ChainSelector::default());
        try_and_log!(UtreexoNode::<ChainSelector, Chain>::run(&mut ibd, kill_signal.clone()).await);

        if *kill_signal.read().await {
            self = UtreexoNode(ibd.0, self.1);
            self.shutdown().await;
            return;
        }

        self = UtreexoNode(ibd.0, self.1);

        // download all blocks from the network
        if self.config.backfill && startup_tip == 0 {
            let end = self.0.chain.get_validation_index().unwrap();
            let chain = self
                .chain
                .get_partial_chain(startup_tip, end, Stump::default())
                .unwrap();

            let mut backfill = UtreexoNode::<SyncNode, PartialChainState>::new(
                self.config.clone(),
                chain,
                self.mempool.clone(),
                None,
            );

            UtreexoNode::<SyncNode, PartialChainState>::run(
                &mut backfill,
                kill_signal.clone(),
                |chain: &PartialChainState| {
                    if chain.has_invalid_blocks() {
                        panic!(
                            "We assumed a chain with invalid blocks, something went really wrong"
                        );
                    }

                    for block in chain.list_valid_blocks() {
                        self.chain
                            .mark_block_as_valid(block.block_hash())
                            .expect("Failed to mark block as valid");
                    }
                },
            )
            .await;
        }

        self = self.catch_up(kill_signal.clone()).await;

        self.last_block_request = self.chain.get_validation_index().unwrap_or(0);

        if let Some(ref cfilters) = self.block_filters {
            self.last_filter = self
                .chain
                .get_block_hash(cfilters.get_height().unwrap_or(1))
                .unwrap();
        }

        info!("starting running node...");
        loop {
            while let Some(notification) = timeout(Duration::from_millis(100), self.node_rx.recv())
                .await
                .unwrap()
            {
                try_and_log!(self.handle_notification(Some(notification)).await);
            }

            if *kill_signal.read().await {
                self.shutdown().await;
                break;
            }

            // Jobs that don't need a connected peer

            // Save our peers db
            periodic_job!(
                self.save_peers(),
                self.last_peer_db_dump,
                PEER_DB_DUMP_INTERVAL,
                RunningNode
            );

            // Rework our address database
            periodic_job!(
                self.address_man.rearrange_buckets(),
                self.1.last_address_rearrange,
                ADDRESS_REARRANGE_INTERVAL,
                RunningNode,
                true
            );

            // Perhaps we need more connections
            periodic_job!(
                self.maybe_open_connection().await,
                self.last_connection,
                TRY_NEW_CONNECTION,
                RunningNode
            );

            // Requests using the node handle
            try_and_log!(self.check_request_timeout());
            self.handle_user_request().await;

            // Check if some of our peers have timed out a request
            try_and_log!(self.check_for_timeout().await);

            // Those jobs bellow needs a connected peer to work
            if self.peer_ids.is_empty() {
                continue;
            }
            // Aks our peers for new addresses
            periodic_job!(
                self.ask_for_addresses().await,
                self.last_get_address_request,
                ASK_FOR_PEERS_INTERVAL,
                RunningNode
            );
            // Open new feeler connection periodically
            periodic_job!(
                self.open_feeler_connection().await,
                self.1.last_feeler,
                FEELER_INTERVAL,
                RunningNode
            );
            // Try broadcast transactions
            periodic_job!(
                self.handle_broadcast().await,
                self.last_broadcast,
                BROADCAST_DELAY,
                RunningNode
            );
            // Send our addresses to our peers
            periodic_job!(
                self.send_addresses().await,
                self.last_send_addresses,
                SEND_ADDRESSES_INTERVAL,
                RunningNode
            );

            // Check whether we are in a stale tip
            periodic_job!(
                self.check_for_stale_tip().await,
                self.last_tip_update,
                ASSUME_STALE,
                RunningNode
            );

            try_and_log!(self.request_rescan_block().await);
            try_and_log!(self.download_filters().await);

            // requests that need a utreexo peer
            if !self.has_utreexo_peers() {
                continue;
            }

            // Check if we haven't missed any block
            if self.inflight.len() < 10 {
                try_and_log!(self.ask_missed_block().await);
            }
        }

        stop_signal.send(()).unwrap();
    }

    async fn download_filters(&mut self) -> Result<(), WireError> {
        if self.inflight.contains_key(&InflightRequests::GetFilters) {
            return Ok(());
        }

        if !self.has_compact_filters_peer() {
            return Ok(());
        }

        let Some(ref filters) = self.block_filters else {
            return Ok(());
        };

        let mut height = filters.get_height()?;
        let best_height = self.chain.get_height()?;

        if height == 0 {
            let user_height = self.config.filter_start_height.unwrap_or(0);

            height = if user_height < 0 {
                best_height.saturating_sub(user_height.unsigned_abs())
            } else {
                user_height as u32
            };

            filters.save_height(height)?;
        }

        if height >= best_height {
            return Ok(());
        }

        info!("Downloading filters from height {}", filters.get_height()?);
        let stop = if height + 1000 > best_height {
            best_height
        } else {
            height + 1000
        };

        let stop_hash = self.chain.get_block_hash(stop)?;
        self.last_filter = stop_hash;

        let peer = self
            .send_to_random_peer(
                NodeRequest::GetFilter((stop_hash, height + 1)),
                ServiceFlags::COMPACT_FILTERS,
            )
            .await?;

        self.inflight
            .insert(InflightRequests::GetFilters, (peer, Instant::now()));
        Ok(())
    }

    async fn ask_missed_block(&mut self) -> Result<(), WireError> {
        let tip = self.chain.get_height().unwrap();
        let next = self.chain.get_validation_index().unwrap();
        if tip == next {
            return Ok(());
        }

        let mut blocks = Vec::new();
        for i in (next + 1)..=tip {
            let hash = self.chain.get_block_hash(i)?;
            // already requested
            if self.inflight.contains_key(&InflightRequests::Blocks(hash)) {
                continue;
            }

            // already downloaded
            if self.blocks.contains_key(&hash) {
                continue;
            }

            blocks.push(hash);
        }

        if blocks.is_empty() {
            return Ok(());
        }

        self.request_blocks(blocks).await?;
        Ok(())
    }

    async fn request_rescan_block(&mut self) -> Result<(), WireError> {
        let tip = self.chain.get_height().unwrap();
        if self.inflight.len() + 10 > RunningNode::MAX_INFLIGHT_REQUESTS {
            return Ok(());
        }
        // We use a grace period to avoid looping at the end of rescan
        if let RescanStatus::Completed(time) = self.1.last_rescan_request {
            if time.elapsed() > Duration::from_secs(60) {
                self.1.last_rescan_request = RescanStatus::None;
            }
        }
        if self.1.last_rescan_request == RescanStatus::None
            && self.chain.get_rescan_index().is_some()
        {
            self.1.last_rescan_request =
                RescanStatus::InProgress(self.chain.get_rescan_index().unwrap());
        }
        if let RescanStatus::InProgress(height) = self.1.last_rescan_request {
            for i in (height + 1)..=(height + 10) {
                if i > tip {
                    self.1.last_rescan_request = RescanStatus::Completed(Instant::now());
                    break;
                }
                self.1.last_rescan_request = RescanStatus::InProgress(i);
                let hash = self.chain.get_block_hash(i)?;
                let peer = self
                    .send_to_random_peer(
                        NodeRequest::GetBlock((vec![hash], false)),
                        ServiceFlags::NONE,
                    )
                    .await?;
                self.inflight
                    .insert(InflightRequests::RescanBlock(hash), (peer, Instant::now()));
            }
        }

        Ok(())
    }

    /// This function checks how many time has passed since our last tip update, if it's
    /// been more than 15 minutes, try to update it.
    async fn check_for_stale_tip(&mut self) -> Result<(), WireError> {
        warn!("Potential stale tip detected, trying extra peers");

        // this catches an edge-case where all our utreexo peers are gone, and the GetData
        // times-out. That yields an error, but doesn't ask the block again. Our last_block_request
        // will be pointing to a block that will never arrive, so we basically deadlock.
        self.last_block_request = self.chain.get_validation_index().unwrap();
        // update this or we'll get this warning every second after 15 minutes without a block,
        // until we get a new block.
        self.last_tip_update = Instant::now();
        self.create_connection(false).await;
        self.send_to_random_peer(
            NodeRequest::GetHeaders(self.chain.get_block_locator().unwrap()),
            ServiceFlags::NONE,
        )
        .await?;
        Ok(())
    }

    async fn handle_new_block(&mut self) -> Result<(), WireError> {
        if self.inflight.contains_key(&InflightRequests::Headers) {
            return Ok(());
        }

        let locator = self.0.chain.get_block_locator().unwrap();

        let peer = self
            .send_to_random_peer(NodeRequest::GetHeaders(locator), ServiceFlags::NONE)
            .await?;

        self.inflight
            .insert(InflightRequests::Headers, (peer, Instant::now()));

        Ok(())
    }

    /// This function is called every time we get a Block message from a peer.
    /// This block may be a rescan block, a user request or a new block that we
    /// need to process.
    async fn handle_block_data(&mut self, block: UtreexoBlock, peer: u32) -> Result<(), WireError> {
        // Rescan block, a block that the wallet is interested in to check if it contains
        // any transaction that we are interested in.
        if self
            .inflight
            .remove(&InflightRequests::RescanBlock(block.block.block_hash()))
            .is_some()
        {
            self.request_rescan_block().await?;
            return Ok(self.chain.process_rescan_block(&block.block)?);
        }
        // If this block is a request made through the user interface, send it back to the
        // user.
        if self
            .inflight
            .remove(&InflightRequests::UserRequest(UserRequest::Block(
                block.block.block_hash(),
            )))
            .is_some()
        {
            if block.udata.is_some() {
                self.1.user_requests.send_answer(
                    UserRequest::UtreexoBlock(block.block.block_hash()),
                    Some(NodeResponse::UtreexoBlock(block)),
                );
                return Ok(());
            }
            self.1.user_requests.send_answer(
                UserRequest::Block(block.block.block_hash()),
                Some(NodeResponse::Block(block.block)),
            );
            return Ok(());
        }

        // If none of the above, it means that this block is a new block that we need to
        // process.

        // Check if we actually requested this block. If a peer sends a block we didn't
        // request, we should disconnect it.
        if self
            .inflight
            .remove(&InflightRequests::Blocks(block.block.block_hash()))
            .is_none()
        {
            // We didn't request this block, so we should disconnect the peer.
            if let Some(peer) = self.peers.get(&peer).cloned() {
                self.address_man.update_set_state(
                    peer.address_id as usize,
                    AddressState::Banned(RunningNode::BAN_TIME),
                );
            }
            error!(
                "Peer {peer} sent us block {} which we didn't request",
                block.block.block_hash()
            );
            self.increase_banscore(peer, 5).await?;
            return Ok(());
        }

        let validation_index = self.chain.get_validation_index()?;
        let mut next_block = self.chain.get_block_hash(validation_index + 1)?;

        debug!(
            "Block {} received, waiting for block {}",
            block.block.block_hash(),
            next_block
        );

        self.blocks.insert(block.block.block_hash(), (peer, block));
        while let Some((peer, block)) = self.blocks.remove(&next_block) {
            debug!("processing block {}", block.block.block_hash(),);
            let (proof, del_hashes, inputs) = floresta_chain::proof_util::process_proof(
                &block.udata.unwrap(),
                &block.block.txdata,
                &self.chain,
            )?;
            tokio::task::yield_now().await;
            if let Err(e) = self
                .chain
                .connect_block(&block.block, proof, inputs, del_hashes)
            {
                error!("Invalid block received by peer {} reason: {:?}", peer, e);
                if let BlockchainError::BlockValidation(e) = e {
                    // Because the proof isn't committed to the block, we can't invalidate
                    // it if the proof is invalid. Any other error should cause the block
                    // to be invalidated.
                    match e {
                        BlockValidationErrors::InvalidCoinbase(_)
                        | BlockValidationErrors::UtxoAlreadySpent(_)
                        | BlockValidationErrors::ScriptValidationError(_)
                        | BlockValidationErrors::InvalidOutput
                        | BlockValidationErrors::ScriptError
                        | BlockValidationErrors::BlockTooBig
                        | BlockValidationErrors::NotEnoughPow
                        | BlockValidationErrors::TooManyCoins
                        | BlockValidationErrors::BadMerkleRoot
                        | BlockValidationErrors::BadWitnessCommitment
                        | BlockValidationErrors::NotEnoughMoney
                        | BlockValidationErrors::FirstTxIsnNotCoinbase
                        | BlockValidationErrors::BadCoinbaseOutValue
                        | BlockValidationErrors::EmptyBlock
                        | BlockValidationErrors::BlockExtendsAnOrphanChain
                        | BlockValidationErrors::BadBip34
                        | BlockValidationErrors::CoinbaseNotMatured => {
                            self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                            try_and_log!(self.chain.invalidate_block(block.block.block_hash()));
                        }
                        BlockValidationErrors::InvalidProof => {}
                    }
                }

                // Disconnect the peer and ban it.
                if let Some(peer) = self.peers.get(&peer).cloned() {
                    self.address_man.update_set_state(
                        peer.address_id as usize,
                        AddressState::Banned(RunningNode::BAN_TIME),
                    );
                }
                self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                return Err(WireError::PeerMisbehaving);
            }

            if !self.chain.is_in_idb() {
                let mempool_delta = self.mempool.write().await.consume_block(&block.block);
                debug!(
                    "Block {} accepted, confirmed transactions: {:?}",
                    block.block.block_hash(),
                    mempool_delta
                );
            }

            let next = self.chain.get_validation_index()? + 1;

            match self.chain.get_block_hash(next) {
                Ok(_next_block) => next_block = _next_block,
                Err(_) => break,
            }
            debug!("accepted block {}", block.block.block_hash());
        }

        // Remove confirmed transactions from the mempool.
        self.last_tip_update = Instant::now();

        Ok(())
    }

    pub(crate) async fn handle_notification(
        &mut self,
        notification: Option<NodeNotification>,
    ) -> Result<(), WireError> {
        match notification {
            Some(NodeNotification::FromPeer(peer, message)) => match message {
                PeerMessages::NewBlock(block) => {
                    debug!("We got an inv with block {block} requesting it");
                    self.handle_new_block().await?;
                }
                PeerMessages::Block(block) => {
                    debug!(
                        "Got data for block {} from peer {peer}",
                        block.block.block_hash()
                    );
                    self.handle_block_data(block, peer).await?;
                }
                PeerMessages::Headers(headers) => {
                    debug!(
                        "Got headers from peer {peer} with {} headers",
                        headers.len()
                    );
                    self.inflight.remove(&InflightRequests::Headers);
                    for header in headers.iter() {
                        self.chain.accept_header(*header)?;
                    }

                    if self.chain.is_in_idb() {
                        let blocks = headers.iter().map(|header| header.block_hash()).collect();
                        self.request_blocks(blocks).await?;
                    }
                }
                PeerMessages::Ready(version) => {
                    debug!(
                        "handshake with peer={peer} succeeded feeler={}",
                        version.feeler
                    );
                    self.handle_peer_ready(peer, &version).await?;
                }
                PeerMessages::Disconnected(idx) => {
                    self.handle_disconnection(peer, idx).await?;
                }
                PeerMessages::Addr(addresses) => {
                    debug!("Got {} addresses from peer {}", addresses.len(), peer);
                    let addresses: Vec<_> =
                        addresses.iter().cloned().map(|addr| addr.into()).collect();
                    self.address_man.push_addresses(&addresses);
                }
                PeerMessages::BlockFilter((hash, filter)) => {
                    debug!("Got a block filter for block {hash} from peer {peer}");

                    if let Some(filters) = self.block_filters.as_ref() {
                        let current_height = filters.get_height()?;
                        let Some(this_height) = self.chain.get_block_height(&hash)? else {
                            warn!("Filter for block {} received, but we don't have it", hash);
                            return Ok(());
                        };

                        // we expect to receive them in order
                        if current_height + 1 != this_height {
                            warn!(
                                "Expected filter for height {}, got filter for height {}",
                                current_height + 1,
                                this_height
                            );
                            self.increase_banscore(peer, 10).await?;
                            return Ok(());
                        }

                        filters.save_height(current_height + 1)?;
                        filters.push_filter(filter)?;

                        if self.last_filter == hash {
                            self.inflight.remove(&InflightRequests::GetFilters);
                            self.download_filters().await?;
                        }
                    }
                }
                PeerMessages::NotFound(inv) => match inv {
                    Inventory::Error => {}
                    Inventory::Block(block)
                    | Inventory::WitnessBlock(block)
                    | Inventory::UtreexoBlock(block)
                    | Inventory::UtreexoWitnessBlock(block)
                    | Inventory::CompactBlock(block) => {
                        self.1
                            .user_requests
                            .send_answer(UserRequest::Block(block), None);
                    }

                    Inventory::WitnessTransaction(tx) | Inventory::Transaction(tx) => {
                        self.1
                            .user_requests
                            .send_answer(UserRequest::MempoolTransaction(tx), None);
                    }
                    _ => {}
                },
                PeerMessages::Transaction(tx) => {
                    debug!("saw a mempool transaction with txid={}", tx.txid());
                    self.1.user_requests.send_answer(
                        UserRequest::MempoolTransaction(tx.txid()),
                        Some(NodeResponse::MempoolTransaction(tx)),
                    );
                }
                PeerMessages::UtreexoState(_) => {
                    warn!(
                        "Utreexo state received from peer {}, but we didn't ask",
                        peer
                    );
                    self.increase_banscore(peer, 5).await?;
                }
            },
            None => {
                // We got a kill signal
                return Ok(());
            }
        }
        Ok(())
    }
}

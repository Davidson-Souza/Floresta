//! After a node catches-up with the network, we can start listening for new blocks, handing any
//! request our user might make and keep our peers alive. This mode requires way less bandwidth and
//! CPU to run, being bound by the number of blocks found in a given period.
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::time::Duration;
use std::time::Instant;

use bitcoin::bip158::BlockFilter;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::message_blockdata::Inventory;
use bitcoin::p2p::ServiceFlags;
use bitcoin::BlockHash;
use floresta_chain::proof_util::UtreexoLeafError;
use floresta_chain::pruned_utreexo::partial_chain::PartialChainState;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_chain::ChainBackend;
use floresta_common::service_flags;
use floresta_common::service_flags::UTREEXO;
use rand::random;
use rustreexo::accumulator::stump::Stump;
use tokio::time::timeout;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use super::error::WireError;
use super::peer::PeerMessages;
use crate::node::periodic_job;
use crate::node::try_and_log;
use crate::node::try_and_warn;
use crate::node::ConnectionKind;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::UtreexoNode;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;
use crate::node_interface::NodeResponse;
use crate::node_interface::UserRequest;
use crate::p2p_wire::chain_selector::ChainSelector;
use crate::p2p_wire::sync_node::SyncNode;

#[derive(Debug, Clone)]
pub struct RunningNode {
    pub(crate) last_address_rearrange: Instant,
    /// To find peers with a good connectivity, keep track of what peers sent us an inv message
    /// for a block, in the first 5 seconds after we get the first inv message. If we ever decide
    /// to disconnect a peer, we should disconnect the ones that didn't send us an inv message
    /// in a timely manner, but keep the ones that notified us of a new blocks the fastest.
    /// We also keep the moment we received the first inv message
    pub(crate) last_invs: HashMap<BlockHash, (Instant, Vec<PeerId>)>,
    #[allow(dead_code)]
    pub(crate) inflight_filters: BTreeMap<u32, BlockFilter>,
}

impl NodeContext for RunningNode {
    const REQUEST_TIMEOUT: u64 = 2 * 60;
    fn get_required_services(&self) -> ServiceFlags {
        ServiceFlags::NETWORK
            | service_flags::UTREEXO.into()
            | ServiceFlags::WITNESS
            | ServiceFlags::COMPACT_FILTERS
    }
}

impl<Chain> UtreexoNode<Chain, RunningNode>
where
    Chain: ChainBackend + Clone + Send + 'static,
    Chain::Error: From<UtreexoLeafError>,
    WireError: From<Chain::Error>,
{
    fn send_addresses(&mut self) -> Result<(), WireError> {
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

        self.send_to_random_peer(NodeRequest::SendAddresses(addresses), ServiceFlags::NONE)?;
        Ok(())
    }

    /// Every time we restart the node, we'll be a few blocks behind the tip. This function
    /// will start a sync node that will request, download and validate all blocks from the
    /// last validation index to the tip. This function will block until the sync node is
    /// finished.
    ///
    /// On the first startup, if we use either assumeutreexo or pow fraud proofs, this function
    /// will only download the blocks that are after the one that got assumed. So, for PoW fraud
    /// proofs, this means the last 100 blocks, and for assumeutreexo, this means however many
    /// blocks from the hard-coded value in the config file.
    pub async fn catch_up(self) -> Result<Self, WireError> {
        let sync = UtreexoNode {
            common: self.common,
            context: SyncNode::default(),
        };

        let sync = sync.run(|_| {}).await;

        Ok(UtreexoNode {
            common: sync.common,
            context: self.context,
        })
    }

    /// This function is called periodically to check if we have:
    /// - 10 connections
    /// - At least one utreexo peer
    /// - At least one compact filters peer
    ///
    /// If we are missing the special peers but have 10 connections, we should disconnect one
    /// random peer and try to connect to a utreexo and a compact filters peer.
    fn check_connections(&mut self) -> Result<(), WireError> {
        // retry the added peers connections
        self.maybe_open_connection_with_added_peers()?;

        // if we have 10 connections, but not a single utreexo or CBF one, disconnect one random
        // peer and create a utreexo and CBS connection
        if !self.has_utreexo_peers() {
            if self.peer_ids.len() == 10 {
                let peer = random::<usize>() % self.peer_ids.len();
                let peer = self
                    .peer_ids
                    .get(peer)
                    .expect("we've modulo before, we should have it");
                self.send_to_peer(*peer, NodeRequest::Shutdown)?;
            }

            self.maybe_open_connection(UTREEXO.into())?;
        }

        if !self.has_compact_filters_peer() {
            if self.peer_ids.len() == 10 {
                let peer = random::<usize>() % self.peer_ids.len();
                let peer = self
                    .peer_ids
                    .get(peer)
                    .expect("we've modulo before, we should have it");
                self.send_to_peer(*peer, NodeRequest::Shutdown)?;
            }

            self.maybe_open_connection(ServiceFlags::COMPACT_FILTERS)?;
        }

        self.maybe_open_connection(ServiceFlags::NONE)?;

        Ok(())
    }

    /// If either PoW fraud proofs or assumeutreexo are enabled, we will "skip" IBD for all
    /// historical blocks. This allow us to start the node faster, making it usable in a few
    /// minutes. If you still want to validate all blocks, you can enable the backfill option.
    ///
    /// This function will spawn a background task that will download and validate all blocks
    /// that got assumed. After completion, the task will shutdown and the node will continue
    /// running normally. If we ever assume an invalid chain, the node will [halt and catch fire].
    ///
    /// [halt and catch fire]: https://en.wikipedia.org/wiki/Halt_and_Catch_Fire_(computing)
    pub fn backfill(&self, done_flag: std::sync::mpsc::Sender<()>) -> Result<bool, WireError> {
        // try finding the last state of the sync node
        let state = std::fs::read(self.config.datadir.clone() + "/.sync_node_state");
        // try to recover from the disk state, if it exists. Otherwise, start from genesis
        let (chain, end) = match state {
            Ok(state) => {
                // if this file is empty, this means we've finished backfilling
                if state.is_empty() {
                    return Ok(false);
                }

                let acc = Stump::deserialize(&state[..(state.len() - 8)]).unwrap();
                let tip = u32::from_le_bytes(
                    state[(state.len() - 8)..(state.len() - 4)]
                        .try_into()
                        .unwrap(),
                );

                let end = u32::from_le_bytes(state[(state.len() - 4)..].try_into().unwrap());
                info!("Recovering backfill node from state tip={tip}, end={tip}");

                (
                    self.chain
                        .get_partial_chain(tip, end, acc)
                        .expect("Failed to get partial chain"),
                    end,
                )
            }
            Err(_) => {
                // if the file doesn't exist or got corrupted, start from genesis
                let end = self
                    .chain
                    .get_validation_index()
                    .expect("can get the validation index");
                (
                    self.chain
                        .get_partial_chain(0, end, Stump::default())
                        .unwrap(),
                    end,
                )
            }
        };

        let backfill = UtreexoNode::<PartialChainState, SyncNode>::new(
            self.config.clone(),
            chain,
            self.mempool.clone(),
            self.kill_signal.clone(),
            #[cfg(feature = "compact-filters")]
            None,
            self.address_man.clone(),
        )
        .unwrap();

        let datadir = self.config.datadir.clone();
        let outer_chain = self.chain.clone();

        let fut = UtreexoNode::<PartialChainState, SyncNode>::run(
            backfill,
            move |chain: &PartialChainState| {
                if chain.has_invalid_blocks() {
                    panic!("We assumed a chain with invalid blocks, something went really wrong");
                }

                done_flag.send(()).unwrap();

                // we haven't finished the backfill yet, save the current state for the next run
                if chain.is_in_ibd() {
                    let acc = chain.get_acc();
                    let tip = chain.get_height().unwrap();
                    let mut ser_acc = Vec::new();
                    acc.serialize(&mut ser_acc).unwrap();
                    ser_acc.extend_from_slice(&tip.to_le_bytes());
                    ser_acc.extend_from_slice(&end.to_le_bytes());
                    std::fs::write(datadir + "/.sync_node_state", ser_acc)
                        .expect("Failed to write sync node state");
                    return;
                }

                // empty the file if we're done
                std::fs::write(datadir + "/.sync_node_state", Vec::new())
                    .expect("Failed to write sync node state");

                for block in chain.list_valid_blocks() {
                    outer_chain
                        .mark_block_as_valid(block.block_hash())
                        .expect("Failed to mark block as valid");
                }

                info!("Backfilling task shutting down...");
            },
        );

        tokio::task::spawn(fut);
        Ok(true)
    }

    pub async fn run(mut self, stop_signal: tokio::sync::oneshot::Sender<()>) {
        try_and_warn!(self.init_peers());

        // Use this node state to Initial Block download
        let mut ibd = UtreexoNode {
            common: self.common,
            context: ChainSelector::default(),
        };

        try_and_log!(UtreexoNode::<Chain, ChainSelector>::run(&mut ibd).await);

        self = UtreexoNode {
            common: ibd.common,
            context: self.context,
        };

        if *self.kill_signal.read().await {
            self.shutdown();
            try_and_log!(stop_signal.send(()));
            return;
        }

        // download blocks from the network before our validation index, probably because we've
        // assumed it somehow.
        let (sender, recv) = std::sync::mpsc::channel();
        let is_backfilling = match self.config.backfill {
            true => {
                info!("Starting backfill task...");
                self.backfill(sender)
                    .expect("Failed to spawn backfill thread")
            }
            false => false,
        };

        // Catch up with the network, downloading blocks from our last validation index to the tip
        info!("Catching up with the network...");
        self = match self.catch_up().await {
            Ok(node) => node,
            Err(e) => {
                error!("An error happened while trying to catch-up with the network: {e:?}",);
                return;
            }
        };

        if *self.kill_signal.read().await {
            self.shutdown();
            stop_signal.send(()).unwrap();
            return;
        }

        self.last_block_request = self.chain.get_validation_index().unwrap_or(0);
        info!("starting running node...");
        loop {
            if *self.kill_signal.read().await {
                break;
            }

            while let Ok(Some(notification)) =
                timeout(Duration::from_millis(100), self.node_rx.recv()).await
            {
                try_and_log!(self.handle_notification(notification));
            }

            // Jobs that don't need a connected peer
            try_and_log!(self.process_pending_blocks());

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
                self.context.last_address_rearrange,
                ADDRESS_REARRANGE_INTERVAL,
                RunningNode,
                true
            );

            // Perhaps we need more connections
            periodic_job!(
                self.check_connections(),
                self.last_connection,
                TRY_NEW_CONNECTION,
                RunningNode
            );

            // Check if some of our peers have timed out a request
            try_and_log!(self.check_for_timeout());

            // Open new feeler connection periodically
            periodic_job!(
                self.open_feeler_connection(),
                self.last_feeler,
                FEELER_INTERVAL,
                RunningNode
            );

            // Those jobs bellow needs a connected peer to work
            if self.peer_ids.is_empty() {
                continue;
            }

            // Ask our peers for new addresses
            periodic_job!(
                self.ask_for_addresses(),
                self.last_get_address_request,
                ASK_FOR_PEERS_INTERVAL,
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
                self.send_addresses(),
                self.last_send_addresses,
                SEND_ADDRESSES_INTERVAL,
                RunningNode
            );

            // Check whether we are in a stale tip
            periodic_job!(
                self.check_for_stale_tip(),
                self.last_tip_update,
                ASSUME_STALE,
                RunningNode
            );

            // requests that need a utreexo peer
            if !self.has_utreexo_peers() {
                continue;
            }

            // Check if we haven't missed any block
            if self.inflight.len() < RunningNode::MAX_INFLIGHT_REQUESTS {
                try_and_log!(self.ask_missed_block());
            }
        }

        // ignore the error here because if the backfill task already
        // finished, this channel will be closed
        if is_backfilling {
            let _ = recv.recv();
        }

        self.shutdown();
        let _ = stop_signal.send(());
    }

    #[cfg(feature = "compact-filters")]
    #[allow(dead_code)]
    fn download_filter_headers(&mut self) -> Result<(), WireError> {
        use floresta_compact_filters::FilterHeadersStore;

        let Some(block_filters) = &self.common.filter_store else {
            return Ok(());
        };

        if self
            .inflight
            .contains_key(&InflightRequests::GetFilterHeaders)
        {
            return Ok(());
        }

        let filter_height = block_filters.get_height().unwrap_or(0);
        let checkpoint = self
            .config
            .assume_utreexo
            .as_ref()
            .map(|assume_utreexo_value| assume_utreexo_value.height)
            .unwrap_or(0);
        // We only download stuff up to our checkpoint, after that we will download
        // those blocks and build the filters ourselves
        if filter_height >= self.chain.get_height()? || filter_height > checkpoint {
            return Ok(());
        }

        debug!(
            "Requesting filter headers from height {}",
            filter_height + 1
        );

        let best_block_hash = self.chain.get_best_block()?.1;
        let peer = self.send_to_random_peer(
            NodeRequest::GetFilterHeaders(filter_height + 1, best_block_hash),
            ServiceFlags::COMPACT_FILTERS,
        )?;

        self.inflight
            .insert(InflightRequests::GetFilterHeaders, (peer, Instant::now()));

        Ok(())
    }

    fn ask_missed_block(&mut self) -> Result<(), WireError> {
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

            if blocks.len() >= RunningNode::MAX_INFLIGHT_REQUESTS {
                break;
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

        self.request_blocks(blocks)?;
        Ok(())
    }

    /// If we think our tip is stale, we may disconnect one peer and try to get a new one.
    /// In this process, if the extra peer gives us a new block, we should drop one of our
    /// already connected peers to keep the number of connections stable. This function
    /// decides which peer to drop based on whether they've timely inv-ed us about the last
    /// 6 blocks.
    fn get_peer_score(&self, peer: PeerId) -> u32 {
        let mut score = 0;
        for block in self.context.last_invs.keys() {
            if self.context.last_invs[block].1.contains(&peer) {
                score += 1;
            }
        }

        score
    }

    /// This function checks how many time has passed since our last tip update, if it's
    /// been more than 15 minutes, try to update it.
    fn check_for_stale_tip(&mut self) -> Result<(), WireError> {
        warn!("Potential stale tip detected, trying extra peers");

        // this catches an edge-case where all our utreexo peers are gone, and the GetData
        // times-out. That yields an error, but doesn't ask the block again. Our last_block_request
        // will be pointing to a block that will never arrive, so we basically deadlock.
        self.last_block_request = self.chain.get_validation_index().unwrap();
        // update this or we'll get this warning every second after 15 minutes without a block,
        // until we get a new block.
        self.last_tip_update = Instant::now();
        self.create_connection(ConnectionKind::Extra)?;
        self.send_to_random_peer(
            NodeRequest::GetHeaders(self.chain.get_block_locator().unwrap()),
            ServiceFlags::NONE,
        )?;
        Ok(())
    }

    fn handle_new_block(&mut self, block: BlockHash, peer: u32) -> Result<(), WireError> {
        if self.inflight.contains_key(&InflightRequests::Headers) {
            return Ok(());
        }

        if self.chain.get_block_header(&block).is_ok() {
            return Ok(());
        }

        let locator = self.chain.get_block_locator().unwrap();
        self.send_to_peer(peer, NodeRequest::GetHeaders(locator))?;

        self.inflight
            .insert(InflightRequests::Headers, (peer, Instant::now()));

        Ok(())
    }

    fn handle_notification(&mut self, notification: NodeNotification) -> Result<(), WireError> {
        match notification {
            NodeNotification::FromUser(request, responder) => {
                self.perform_user_request(request, responder);
            }

            NodeNotification::DnsSeedAddresses(addresses) => {
                self.address_man.push_addresses(&addresses);
            }

            NodeNotification::FromPeer(peer, message) => {
                self.register_message_time(&message, peer);

                match message {
                    PeerMessages::UtreexoProof(uproof) => {
                        self.attach_proof(uproof, peer)?;
                        self.process_pending_blocks()?;
                    }

                    PeerMessages::FilterHeader(_) => {
                        // Running node doesn't handle filter headers
                    }

                    PeerMessages::NewBlock(block) => {
                        debug!("We got an inv with block {block} requesting it");
                        self.context
                            .last_invs
                            .entry(block)
                            .and_modify(|(when, peers)| {
                                if peers.contains(&peer) {
                                    return;
                                }

                                // if it's been less than 5 seconds since we got the first inv message
                                // for this block, we should mark as this peer sent us in a timely manner
                                if when.elapsed() < Duration::from_secs(5) {
                                    peers.push(peer);
                                }
                            })
                            .or_insert_with(|| (Instant::now(), Vec::new()));

                        if self.chain.get_block_header(&block).is_ok() {
                            return Ok(());
                        }

                        if self.inflight.contains_key(&InflightRequests::Blocks(block)) {
                            return Ok(());
                        }

                        let p = self
                            .peers
                            .get(&peer)
                            .cloned()
                            .ok_or(WireError::PeerNotFound)?;

                        // if this is a utreexo peer, we should ask for the block if we don't
                        // have it
                        if p.services.has(UTREEXO.into()) {
                            self.handle_new_block(block, peer)?;
                        }
                    }

                    PeerMessages::Block(block) => {
                        self.request_block_proof(block, peer)?;
                    }

                    PeerMessages::Headers(headers) => {
                        debug!(
                            "Got headers from peer {peer} with {} headers",
                            headers.len()
                        );
                        self.inflight.remove(&InflightRequests::Headers);

                        let peer_info = self.peers.get(&peer).cloned().expect("Peer not found");
                        let is_extra = matches!(peer_info.kind, ConnectionKind::Extra);

                        if is_extra {
                            // if this is an extra peer, and the headers message is empty, disconnect it
                            if headers.is_empty() {
                                self.increase_banscore(peer, 5)?;
                                return Ok(());
                            }

                            // this peer got us a new block, we should disconnect one of our regular peers
                            // and keep this one.
                            let peer_to_disconnect = self
                                .peers
                                .iter()
                                .filter(|(_, info)| matches!(info.kind, ConnectionKind::Regular(_)))
                                .min_by_key(|(k, _)| self.get_peer_score(**k))
                                .map(|(peer, _)| *peer);

                            // disconnect the peer with the lowest score
                            if let Some(peer) = peer_to_disconnect {
                                self.send_to_peer(peer, NodeRequest::Shutdown)?;
                            }

                            // update the peer info
                            self.peers.entry(peer).and_modify(|info| {
                                info.kind = ConnectionKind::Regular(peer_info.services);
                            });
                        }

                        for header in headers.iter() {
                            self.chain.accept_header(*header)?;

                            self.send_to_peer(
                                peer,
                                NodeRequest::GetBlock(vec![header.block_hash()]),
                            )?;

                            self.inflight.insert(
                                InflightRequests::Blocks(header.block_hash()),
                                (peer, Instant::now()),
                            );
                        }

                        // update the peer info
                        self.peers.entry(peer).and_modify(|info| {
                            info.kind = ConnectionKind::Regular(peer_info.services);
                        });
                    }

                    PeerMessages::Ready(version) => {
                        debug!(
                            "handshake with peer={peer} succeeded feeler={:?}",
                            version.kind
                        );
                        self.handle_peer_ready(peer, &version)?;
                    }

                    PeerMessages::Disconnected(idx) => {
                        self.handle_disconnection(peer, idx)?;
                    }

                    PeerMessages::Addr(addresses) => {
                        debug!("Got {} addresses from peer {}", addresses.len(), peer);
                        let addresses: Vec<_> =
                            addresses.into_iter().map(|addr| addr.into()).collect();

                        self.address_man.push_addresses(&addresses);
                    }

                    PeerMessages::NotFound(inv) => match inv {
                        Inventory::Error => {}
                        Inventory::Block(block)
                        | Inventory::WitnessBlock(block)
                        | Inventory::CompactBlock(block) => {
                            if let Some(request) = self
                                .inflight_user_requests
                                .remove(&UserRequest::Block(block))
                            {
                                request.2.send(NodeResponse::Block(None)).unwrap();
                            }
                        }

                        Inventory::WitnessTransaction(tx) | Inventory::Transaction(tx) => {
                            if let Some(request) = self
                                .inflight_user_requests
                                .remove(&UserRequest::MempoolTransaction(tx))
                            {
                                request
                                    .2
                                    .send(NodeResponse::MempoolTransaction(None))
                                    .unwrap();
                            }
                        }

                        i => {
                            warn!("Got NotFound for an unrequested inv {i:?} from peer {peer}");
                            self.increase_banscore(peer, 1)?;
                        }
                    },

                    PeerMessages::Transaction(tx) => {
                        debug!("saw a mempool transaction with txid={}", tx.compute_txid());
                        if let Some(request) = self
                            .inflight_user_requests
                            .remove(&UserRequest::MempoolTransaction(tx.compute_txid()))
                        {
                            request
                                .2
                                .send(NodeResponse::MempoolTransaction(Some(tx)))
                                .unwrap();
                        }
                    }

                    PeerMessages::UtreexoState(_) => {
                        warn!("Utreexo state received from peer {peer}, but we didn't ask",);
                        self.increase_banscore(peer, 5)?;
                    }

                    PeerMessages::BlockFilter(_) => {
                        warn!("Block filter received from peer {peer}, but we didn't ask",);
                        self.increase_banscore(peer, 5)?;
                    }
                }
            }
        }
        Ok(())
    }
}

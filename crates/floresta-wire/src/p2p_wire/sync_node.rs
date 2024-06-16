//! A node that downlaods and validates the blockchain.

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use async_std::future::timeout;
use async_std::sync::RwLock;
use bitcoin::p2p::utreexo::UtreexoBlock;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_chain::pruned_utreexo::UpdatableChainstate;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
use log::debug;
use log::error;
use log::info;

use super::error::WireError;
use super::peer::PeerMessages;
use crate::address_man::AddressState;
use crate::node::periodic_job;
use crate::node::try_and_log;
use crate::node::InflightRequests;
use crate::node::NodeNotification;
use crate::node::NodeRequest;
use crate::node::UtreexoNode;
use crate::node_context::NodeContext;
use crate::node_context::PeerId;

#[derive(Clone, Debug, Default)]
pub struct SyncNode {
    last_block_requested: u32,
}

impl NodeContext for SyncNode {
    fn get_required_services(&self) -> bitcoin::p2p::ServiceFlags {
        ServiceFlags::WITNESS | ServiceFlags::UTREEXO | ServiceFlags::NETWORK
    }

    const MAX_OUTGOING_PEERS: usize = 5; // don't need many peers, half the default
    const TRY_NEW_CONNECTION: u64 = 10; // ten seconds
    const REQUEST_TIMEOUT: u64 = 30; // 30 seconds
    const MAX_INFLIGHT_REQUESTS: usize = 100; // double the default
}

impl<Chain> UtreexoNode<SyncNode, Chain>
where
    WireError: From<<Chain as BlockchainInterface>::Error>,
    Chain: BlockchainInterface + UpdatableChainstate + 'static,
{
    pub async fn run(&mut self, kill_signal: Arc<RwLock<bool>>, done_cb: impl FnOnce(&Chain)) {
        info!("Starting sync node");
        self.1.last_block_requested = self.chain.get_validation_index().unwrap();
        loop {
            while let Ok(Ok(msg)) = timeout(Duration::from_secs(1), self.node_rx.recv()).await {
                self.handle_message(msg).await;
            }

            if *kill_signal.read().await {
                break;
            }

            if !self.chain.is_in_idb() {
                break;
            }

            periodic_job!(
                self.maybe_open_connection().await,
                self.last_connection,
                TRY_NEW_CONNECTION,
                SyncNode
            );

            self.handle_timeout().await;

            if self.has_utreexo_peers() {
                continue;
            }

            if self.inflight.len() < SyncNode::MAX_INFLIGHT_REQUESTS {
                let mut blocks = Vec::with_capacity(100);
                for _ in 0..100 {
                    let next_block = self.1.last_block_requested + 1;
                    let next_block = self.chain.get_block_hash(next_block);
                    match next_block {
                        Ok(next_block) => {
                            blocks.push(next_block);
                            self.1.last_block_requested += 1;
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
                try_and_log!(self.request_blocks(blocks).await);
            }
        }

        done_cb(&self.chain);
    }

    async fn handle_timeout(&mut self) {
        let mut to_remove = Vec::new();
        for (block, (peer, when)) in self.inflight.iter() {
            if when.elapsed().as_secs() > SyncNode::REQUEST_TIMEOUT {
                to_remove.push((*peer, block.clone()));
            }
        }

        for (peer, block) in to_remove {
            self.inflight.remove(&block);
            try_and_log!(self.increase_banscore(peer, 1).await);

            let InflightRequests::Blocks(block) = block else {
                continue;
            };
            try_and_log!(self.request_blocks(vec![block]).await);
        }
    }

    async fn handle_block_data(
        &mut self,
        peer: PeerId,
        block: UtreexoBlock,
    ) -> Result<(), WireError> {
        self.inflight
            .remove(&InflightRequests::Blocks(block.block.block_hash()));

        self.blocks.insert(block.block.block_hash(), (peer, block));

        let next_block = self.chain.get_validation_index()? + 1;
        let mut next_block = self.chain.get_block_hash(next_block)?;

        while let Some((peer, block)) = self.blocks.remove(&next_block) {
            if block.udata.is_none() {
                error!("Block without proof received from peer {}", peer);
                self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                self.send_to_random_peer(
                    NodeRequest::GetBlock((vec![block.block.block_hash()], true)),
                    ServiceFlags::UTREEXO,
                )
                .await?;
                self.inflight
                    .insert(InflightRequests::Blocks(next_block), (peer, Instant::now()));
                return Err(WireError::PeerMisbehaving);
            }

            debug!("processing block {}", block.block.block_hash(),);
            let (proof, del_hashes, inputs) = floresta_chain::proof_util::process_proof(
                &block.udata.unwrap(),
                &block.block.txdata,
                &self.chain,
            )?;

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
                        BlockValidationErrors::InvalidTx(_)
                        | BlockValidationErrors::NotEnoughPow
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
                        AddressState::Banned(SyncNode::BAN_TIME),
                    );
                }
                self.send_to_peer(peer, NodeRequest::Shutdown).await?;
                return Err(WireError::PeerMisbehaving);
            }

            let next = self.chain.get_validation_index()? + 1;

            match self.chain.get_block_hash(next) {
                Ok(_next_block) => next_block = _next_block,
                Err(_) => break,
            }
            debug!("accepted block {}", block.block.block_hash());
        }

        Ok(())
    }

    async fn handle_message(&mut self, msg: NodeNotification) {
        match msg {
            NodeNotification::FromPeer(peer, notification) => match notification {
                PeerMessages::Block(block) => {
                    if let Err(e) = self.handle_block_data(peer, block).await {
                        error!("Error processing block: {:?}", e);
                    }
                }
                PeerMessages::Ready(version) => {
                    try_and_log!(self.handle_peer_ready(peer, &version).await);
                }
                PeerMessages::Disconnected(idx) => {
                    try_and_log!(self.handle_disconnection(peer, idx));
                }
                _ => {}
            },
        }
    }
}

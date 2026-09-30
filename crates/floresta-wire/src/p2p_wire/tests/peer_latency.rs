// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression tests for shared peer latency and timeout handling.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use std::time::Instant;

    use bitcoin::Network;
    use bitcoin::bip158::BlockFilter;
    use bitcoin::p2p::ServiceFlags;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;
    use floresta_chain::pruned_utreexo::BlockchainInterface;
    use tokio::sync::mpsc::unbounded_channel;
    use tokio::sync::oneshot;

    use crate::node::ConnectionKind;
    use crate::node::InflightRequests;
    use crate::node::NodeRequest;
    use crate::node::PeerStatus;
    use crate::node::UtreexoNode;
    use crate::node::chain_selector_ctx::ChainSelector;
    use crate::node::running_ctx::RunningNode;
    use crate::node::sync_ctx::SyncNode;
    use crate::node_context::NodeContext;
    use crate::node_handle::UserRequest;
    use crate::p2p_wire::peer::PeerMessages;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::SetupNodeArgs;
    use crate::p2p_wire::tests::utils::setup_node;
    use crate::p2p_wire::tests::utils::signet_blocks;

    const NUM_BLOCKS: usize = 9;

    /// Creates two simulated peers with initial latency samples of 1 ms.
    fn latency_node<T: 'static + Default + NodeContext>()
    -> UtreexoNode<Arc<ChainState<FlatChainStore>>, T> {
        let peer = PeerData::new(Vec::new(), signet_blocks(), HashMap::new());
        let args = SetupNodeArgs::new(
            vec![peer; 2],
            false,
            Network::Signet,
            format!("./tmp-db/{}.sync_latency", rand::random::<u32>()),
            NUM_BLOCKS,
        );
        let mut node = setup_node::<T>(args);
        // Set the initial latency and clear handshake requests ourselves because these
        // tests call node methods directly, without running the node's event loop.
        for peer in node.peers.values_mut() {
            peer.message_times.add(1.0);
            peer.state = PeerStatus::Ready;
        }
        node.inflight.clear();
        node
    }

    #[tokio::test]
    // Verifies that failed retries survive disconnection and can be retried later.
    async fn test_failed_retries() {
        let mut node = latency_node::<SyncNode>();
        let requests = [
            InflightRequests::Blocks(node.chain.get_block_hash(1).unwrap()),
            InflightRequests::Headers,
            InflightRequests::UtreexoState(0),
        ];

        // No peer can accept a retry, regardless of the requested service
        for peer in node.peers.values_mut() {
            peer.state = PeerStatus::Awaiting;
        }

        let expired_time = Instant::now() - Duration::from_secs(SyncNode::REQUEST_TIMEOUT + 1);
        for req in &requests {
            node.inflight.insert(req.clone(), (0, expired_time));
        }

        // Without a ready replacement, keep every request for a later retry
        let original_inflight = node.inflight.clone();
        for _ in 0..2 {
            assert!(node.check_for_timeout().is_err());
            assert_eq!(node.inflight, original_inflight);
            assert!(!node.peers.contains_key(&0));
            assert_eq!(node.peers[&1].banscore, 0);
        }

        // A replacement becomes ready, so all requests must move to it
        node.peers.get_mut(&1).unwrap().state = PeerStatus::Ready;
        node.check_for_timeout().unwrap();
        assert_eq!(node.inflight.len(), requests.len());
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::UtreexoState(0))
        );
        assert!(
            node.inflight
                .contains_key(&InflightRequests::UtreexoState(1))
        );
        assert!(
            node.inflight
                .values()
                .all(|&(peer, time)| peer == 1 && time > expired_time)
        );
    }

    #[tokio::test]
    // Verifies that a failed block retry doesn't stop a header retry for the same peer.
    async fn test_mixed_retries() {
        let mut node = latency_node::<SyncNode>();
        let request = InflightRequests::Blocks(node.chain.get_block_hash(1).unwrap());
        let expired_time = Instant::now() - Duration::from_secs(SyncNode::REQUEST_TIMEOUT + 1);
        node.inflight.insert(request.clone(), (0, expired_time));
        node.inflight
            .insert(InflightRequests::Headers, (0, expired_time));

        // Peer 1 can serve headers, but doesn't offer block downloads
        node.peers.get_mut(&1).unwrap().services = ServiceFlags::NONE;
        assert!(node.check_for_timeout().is_err());
        assert!(!node.peers.contains_key(&0));
        assert_eq!(node.inflight[&request], (0, expired_time));
        assert_eq!(node.inflight[&InflightRequests::Headers].0, 1);
        assert_eq!(node.peers[&1].banscore, 0);
    }

    #[tokio::test]
    // Verifies that address timeouts preserve ordinary peers and manual data requests
    // are retried, while feeler and handshake timeouts still disconnect in every context.
    async fn test_address_timeout() {
        fn check<T: 'static + Default + NodeContext>() {
            let mut node = latency_node::<T>();
            let (sender, mut messages) = unbounded_channel();
            node.peers.get_mut(&0).unwrap().channel = sender;
            let expired = Instant::now() - Duration::from_secs(T::REQUEST_TIMEOUT + 1);

            // Missing addresses alone must not close a useful connection.
            for kind in [
                ConnectionKind::Regular(ServiceFlags::NETWORK),
                ConnectionKind::Manual,
                ConnectionKind::Extra,
            ] {
                node.peers.get_mut(&0).unwrap().kind = kind;
                node.inflight
                    .insert(InflightRequests::GetAddresses, (0, expired));
                node.check_for_timeout().unwrap();
                assert!(node.inflight.is_empty());
                assert_eq!(node.peers[&0].banscore, 0);
                assert!(matches!(
                    messages.try_recv(),
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty)
                ));
            }

            // With no ready alternative, retry the manual peer without disconnecting it.
            node.peers.get_mut(&0).unwrap().kind = ConnectionKind::Manual;
            node.peers.get_mut(&1).unwrap().state = PeerStatus::Awaiting;
            node.inflight
                .insert(InflightRequests::GetAddresses, (0, expired));
            node.inflight
                .insert(InflightRequests::Headers, (0, expired));
            node.check_for_timeout().unwrap();
            assert!(node.peers.contains_key(&0));
            assert_eq!(node.inflight.len(), 1);
            let (peer, sent_at) = node.inflight[&InflightRequests::Headers];
            assert_eq!(peer, 0);
            assert!(sent_at > expired);
            assert!(matches!(
                messages.try_recv().unwrap(),
                NodeRequest::GetHeaders(_)
            ));

            // Manual peers must still complete the handshake before its deadline.
            node.peers.get_mut(&0).unwrap().state = PeerStatus::Awaiting;
            node.inflight.clear();
            node.inflight.insert(
                InflightRequests::Connect(0),
                (
                    0,
                    Instant::now() - Duration::from_secs(T::CONNECTION_TIMEOUT + 1),
                ),
            );
            node.check_for_timeout().unwrap();
            assert!(!node.peers.contains_key(&0));
            assert!(matches!(
                messages.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
            ));
            assert!(node.inflight.is_empty());

            // Feelers are temporary address-discovery connections, so close them on timeout.
            node.inflight.clear();
            let (sender, mut messages) = unbounded_channel();
            let peer = node.peers.get_mut(&1).unwrap();
            peer.channel = sender;
            peer.state = PeerStatus::Ready;
            peer.kind = ConnectionKind::Feeler;
            node.inflight
                .insert(InflightRequests::GetAddresses, (1, expired));
            node.check_for_timeout().unwrap();
            assert!(!node.peers.contains_key(&1));
            assert!(matches!(
                messages.try_recv().unwrap(),
                NodeRequest::Shutdown
            ));
            assert!(node.inflight.is_empty());
        }

        check::<SyncNode>();
        check::<RunningNode>();
        check::<ChainSelector>();
    }

    #[tokio::test]
    // Verifies that replies from the wrong peer or with a stale timestamp don't change
    // latency. A valid reply updates only the peer assigned to the request.
    async fn test_reply_latency() {
        let mut node = latency_node::<SyncNode>();
        let hash = node.chain.get_block_hash(1).unwrap();
        let block = signet_blocks().remove(&hash).unwrap();
        let requests = [
            (InflightRequests::Blocks(hash), PeerMessages::Block(block)),
            (InflightRequests::Headers, PeerMessages::Headers(Vec::new())),
            (
                InflightRequests::GetFilters,
                PeerMessages::BlockFilter((hash, BlockFilter::new(&[]))),
            ),
            (
                InflightRequests::UtreexoState(1),
                PeerMessages::UtreexoState(Vec::new()),
            ),
        ];

        let mut expected_lat = node.peers[&1].message_times.clone();
        assert_eq!(expected_lat.value().unwrap(), 1.0, "initial latency is 1ms");

        for (request, message) in requests {
            // Prepare a request to peer 1 and a valid reply timestamp 5 seconds later
            let sent_at = Instant::now();
            let read_at = sent_at + Duration::from_secs(5);
            node.inflight.insert(request, (1, sent_at));

            // Peer 0 wasn't assigned this request, so its reply must not count
            assert_eq!(node.register_message_time(&message, 0, read_at), None);

            // Peer 1 is correct, but a reply read before `sent_at` is a stale sample
            let stale_time = sent_at - Duration::from_millis(1);
            assert_eq!(node.register_message_time(&message, 1, stale_time), None);

            // Neither rejected reply should change either peer's latency
            assert_eq!(node.peers[&0].message_times.value(), Some(1.0));
            assert_eq!(node.peers[&1].message_times.value(), expected_lat.value());

            // The valid reply adds a 5000ms sample to peer 1, while peer 0 stays at 1ms
            node.register_message_time(&message, 1, read_at).unwrap();

            expected_lat.add(5_000.0);
            assert_eq!(node.peers[&1].message_times.value(), expected_lat.value());
            assert_eq!(node.peers[&0].message_times.value(), Some(1.0));
            node.inflight.clear();
        }
    }

    #[tokio::test]
    // Verifies that a timeout disconnects the peer, moves all its requests to another
    // peer, and ignores queued replies without disturbing the new requests.
    async fn test_timeout_disconnect() {
        let mut node = latency_node::<RunningNode>();
        let (sender, mut messages) = unbounded_channel();
        node.peers.get_mut(&0).unwrap().channel = sender;
        let hash = node.chain.get_block_hash(1).unwrap();
        let block = signet_blocks().remove(&hash).unwrap();
        let request = InflightRequests::Blocks(hash);
        let expired_time = Instant::now() - Duration::from_secs(RunningNode::REQUEST_TIMEOUT + 1);
        node.inflight.insert(request.clone(), (0, expired_time));
        node.inflight
            .insert(InflightRequests::Headers, (0, expired_time));

        // Even a fresh request needs a replacement when its peer is disconnected
        let fresh_request = InflightRequests::Blocks(node.chain.get_block_hash(2).unwrap());
        node.inflight
            .insert(fresh_request.clone(), (0, Instant::now()));
        let (sender, mut user_reply) = oneshot::channel();
        node.inflight_user_requests
            .insert(UserRequest::Block(hash), (0, Instant::now(), sender));
        node.check_for_timeout().unwrap();
        assert!(matches!(
            messages.try_recv().unwrap(),
            NodeRequest::Shutdown
        ));
        assert!(!node.peers.contains_key(&0));
        assert_eq!(
            user_reply.try_recv().unwrap_err(),
            oneshot::error::TryRecvError::Closed
        );
        assert_eq!(node.inflight.len(), 3);
        assert!(
            node.inflight
                .values()
                .all(|&(peer, time)| peer == 1 && time > expired_time)
        );
        assert_eq!(node.peers[&1].banscore, 0);

        // Rechecking fresh retries must leave the replacement peer connected
        let retried = node.inflight.clone();
        node.check_for_timeout().unwrap();
        assert_eq!(node.inflight, retried);
        assert!(node.peers.contains_key(&1));

        // The old peer may already have queued replies before receiving Shutdown
        for message in [
            PeerMessages::Block(block.clone()),
            PeerMessages::Headers(Vec::new()),
        ] {
            assert!(node.handle_peer_msg_common(message, 0).unwrap().is_none());
        }
        let message = node
            .handle_peer_msg_common(PeerMessages::Disconnected(0), 0)
            .unwrap();
        let Some(PeerMessages::Disconnected(idx)) = message else {
            panic!("disconnection notifications must still reach the context handler");
        };
        node.handle_disconnection(0, idx).unwrap();
        assert_eq!(node.inflight, retried);
        assert!(!node.blocks.contains_key(&hash));

        // The replacement's reply is accepted and doesn't receive any banscore
        let message = node
            .handle_peer_msg_common(PeerMessages::Block(block), 1)
            .unwrap();
        let Some(PeerMessages::Block(block)) = message else {
            panic!("the requested block should reach the block handler");
        };
        node.request_block_proof(block, 1).unwrap();
        assert!(!node.inflight.contains_key(&request));
        assert!(node.inflight.contains_key(&fresh_request));
        assert!(node.blocks.contains_key(&hash));
        assert_eq!(node.peers[&1].banscore, 0);
    }
}

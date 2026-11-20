// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use bitcoin::Block;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;
    use floresta_chain::extensions::BlockExt;
    use floresta_chain::extensions::BlockExtError;
    use tokio::sync::mpsc::error::TryRecvError;
    use tokio::sync::mpsc::unbounded_channel;
    use tokio::sync::oneshot;
    use tokio::sync::oneshot::Receiver;

    use crate::node::BlockStructureCheckOutcome;
    use crate::node::ConnectionKind;
    use crate::node::InflightRequests;
    use crate::node::NodeRequest;
    use crate::node::PeerStatus;
    use crate::node::UtreexoNode;
    use crate::node::sync_ctx::SyncNode;
    use crate::node_handle::NodeResponse;
    use crate::node_handle::UserRequest;
    use crate::p2p_wire::error::WireError;
    use crate::p2p_wire::tests::utils::BlockDefect;
    use crate::p2p_wire::tests::utils::PEER_TEST;
    use crate::p2p_wire::tests::utils::mutate_block;
    use crate::p2p_wire::tests::utils::setup_unit_node;
    use crate::p2p_wire::tests::utils::signet_blocks;
    use crate::p2p_wire::tests::utils::signet_headers;
    use crate::p2p_wire::tests::utils::synthetic_block;

    type TestNode = UtreexoNode<Arc<ChainState<FlatChainStore>>, SyncNode>;

    type TestSetup = (TestNode, Block, Option<Receiver<NodeResponse>>);

    fn setup_test(is_user_request: bool, is_mutated_block: bool) -> TestSetup {
        let mut node = setup_unit_node();
        let blocks = signet_blocks();
        let headers = signet_headers();

        let block_hash = headers[1].block_hash();
        let mut block = blocks.get(&block_hash).unwrap().clone();
        if is_mutated_block {
            mutate_block(&mut block);
        }

        let block_hash = block.block_hash();

        // Pre-populate inflight
        node.inflight.insert(
            InflightRequests::Blocks(block_hash),
            (PEER_TEST, Instant::now()),
        );

        let mut response = None;
        if is_user_request {
            let (tx, rx) = oneshot::channel::<NodeResponse>();
            node.inflight_user_requests.insert(
                UserRequest::Block(block_hash),
                (PEER_TEST, Instant::now(), tx),
            );

            response = Some(rx);
        }

        (node, block, response)
    }

    /// Runs the received-block orchestration under test: notification-level handling
    /// followed by storage/proof processing for accepted blocks.
    fn receive_block(node: &mut TestNode, block: Block, peer: u32) -> Result<(), WireError> {
        if let Some(block) = node.handle_received_block(block, peer)? {
            node.request_block_proof(block, peer)?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_unmutated_block_is_ok() {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::None);

        node.enforce_block_structure_check(&block, PEER_TEST, true)
            .unwrap();

        // The peer shouldn't be punished
        assert_eq!(node.peers.get(&PEER_TEST).unwrap().state, PeerStatus::Ready);
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_mutated_block_bans_peer() {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::MerkleRoot);
        assert!(!block.check_merkle_root());

        let result = node
            .enforce_block_structure_check(&block, PEER_TEST, true)
            .unwrap_err();

        assert!(matches!(result, WireError::PeerMisbehaving));
        assert_eq!(
            node.peers.get(&PEER_TEST).unwrap().state,
            PeerStatus::Banned
        );
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_bad_witness_commitment_bans_peer() {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::WitnessCommitment);

        // The merkle root matches, so only the witness commitment is invalid
        assert!(block.check_merkle_root());
        assert!(!block.check_witness_commitment());

        let result = node
            .enforce_block_structure_check(&block, PEER_TEST, true)
            .unwrap_err();

        assert!(matches!(result, WireError::PeerMisbehaving));
        assert_eq!(
            node.peers.get(&PEER_TEST).unwrap().state,
            PeerStatus::Banned
        );
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_too_many_sigops_is_ok() {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::TooManySigops);
        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::TooManySigops)
        ));

        // Authentic consensus-invalid blocks invalidate but don't ban.
        node.enforce_block_structure_check(&block, PEER_TEST, true)
            .unwrap();

        assert_eq!(node.peers.get(&PEER_TEST).unwrap().state, PeerStatus::Ready);
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_block_too_big_is_ok() {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::BlockTooBig);
        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::BlockTooBig)
        ));

        // Authentic consensus-invalid blocks invalidate but don't ban.
        node.enforce_block_structure_check(&block, PEER_TEST, true)
            .unwrap();

        assert_eq!(node.peers.get(&PEER_TEST).unwrap().state, PeerStatus::Ready);
    }

    #[tokio::test]
    async fn test_block_proof_coinbase_only_block_stores_and_skips_proof_request() {
        let (mut node, block, _) = setup_test(false, false);
        let block_hash = block.block_hash();

        receive_block(&mut node, block, PEER_TEST).unwrap();

        // The block should be stored
        assert!(node.blocks.contains_key(&block_hash));
        // The Blocks inflight entry should have been removed
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );

        // Coinbase-only: no proof needed, aux_data is ready
        let inflight_block = node.blocks.get(&block_hash).unwrap();
        assert!(inflight_block.aux_data.is_some());
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::UtreexoProof(block_hash))
        );
    }

    #[tokio::test]
    async fn test_block_proof_unrequested_block_is_ignored() {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::TooManySigops);
        let block_hash = block.block_hash();

        receive_block(&mut node, block, PEER_TEST).unwrap();

        // The block is discarded before structure validation, so it is not stored,
        // no proof is requested and the peer is only penalized with a banscore.
        assert!(!node.blocks.contains_key(&block_hash));
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::UtreexoProof(block_hash))
        );
        let peer = node.peers.get(&PEER_TEST).unwrap();
        assert_eq!(peer.state, PeerStatus::Ready);
        assert_eq!(peer.banscore, 5);
    }

    #[tokio::test]
    async fn test_block_proof_valid_block_user_request_replies_to_user() {
        let (mut node, block, response) = setup_test(true, false);
        let block_hash = block.block_hash();

        receive_block(&mut node, block, PEER_TEST).unwrap();

        // The user should have received the block
        let response = response.unwrap().await.unwrap();
        match response {
            NodeResponse::Block(Some(b)) => assert_eq!(b.block_hash(), block_hash),
            other => panic!("expected NodeResponse::Block(Some(_)), got {other:?}"),
        }

        // The block should NOT be stored in self.blocks (consumed by the user reply)
        assert!(!node.blocks.contains_key(&block_hash));

        // The user request should have been removed
        assert!(
            !node
                .inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );

        // The proof request should NOT happen.
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::UtreexoProof(block_hash))
        );
    }

    #[tokio::test]
    async fn test_block_proof_mutated_block_not_user_request_bans_peer() {
        let (mut node, mutated_block, _) = setup_test(false, true);
        let block_hash = mutated_block.block_hash();

        let result = receive_block(&mut node, mutated_block, PEER_TEST);

        assert!(matches!(result, Err(WireError::PeerMisbehaving)));

        // Block should NOT be stored
        assert!(!node.blocks.contains_key(&block_hash));
        // Peer should be banned
        assert_eq!(
            node.peers.get(&PEER_TEST).unwrap().state,
            PeerStatus::Banned
        );
        // The proof request should NOT happen.
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::UtreexoProof(block_hash))
        );
    }

    #[tokio::test]
    async fn test_block_proof_mutated_block_user_request_retries() {
        let (mut node, mutated_block, _) = setup_test(true, true);
        let block_hash = mutated_block.block_hash();

        receive_block(&mut node, mutated_block, PEER_TEST).unwrap();

        assert_eq!(
            node.peers.get(&PEER_TEST).unwrap().state,
            PeerStatus::Banned
        );

        // A new Blocks inflight entry should exist (the retry)
        assert!(
            node.inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );

        // The user request should still be open (not removed)
        assert!(
            node.inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );

        // The proof request should NOT happen.
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::UtreexoProof(block_hash))
        );
    }

    #[tokio::test]
    async fn test_block_proof_mutated_block_user_request_survives_offender_disconnect() {
        let (mut node, mutated_block, response) = setup_test(true, true);
        let block_hash = mutated_block.block_hash();

        assert!(receive_block(&mut node, mutated_block, PEER_TEST).is_ok());

        // The retry goes to a replacement peer, not the offender
        let retry_peer = node
            .inflight
            .get(&InflightRequests::Blocks(block_hash))
            .map(|req| req.0);
        assert!(matches!(retry_peer, Some(peer) if peer != PEER_TEST));

        // The request must track the replacement peer after the retry
        assert_eq!(
            node.inflight_user_requests
                .get(&UserRequest::Block(block_hash))
                .map(|req| req.0),
            retry_peer
        );

        // Race: the banned offender's disconnection is processed before the replacement
        // replies. Unrelated inflight redos of the fixture may error; that's not what
        // this test measures, so the result is ignored.
        let idx = match node.peers.get(&PEER_TEST) {
            Some(peer) => peer.address.id,
            None => 0,
        };
        let _ = node.handle_disconnection(PEER_TEST, idx);

        // The request must survive the offender's disconnection
        assert!(
            node.inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );

        // The mutation keeps the header, so the replacement's valid block matches the hash
        let valid = signet_blocks().get(&block_hash).cloned();
        if let (Some(retry_peer), Some(valid)) = (retry_peer, valid) {
            assert!(receive_block(&mut node, valid, retry_peer).is_ok());
        }

        // The replacement's valid block completes the user request
        let reply = match response {
            Some(rx) => rx.await.ok(),
            None => None,
        };
        assert!(matches!(
            &reply,
            Some(NodeResponse::Block(Some(block))) if block.block_hash() == block_hash
        ));

        assert!(
            !node
                .inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );
        // Consumed by the user reply
        assert!(!node.blocks.contains_key(&block_hash));
    }

    #[tokio::test]
    async fn test_block_proof_mutated_block_user_request_replacement_disconnect_fails_request() {
        let (mut node, mutated_block, response) = setup_test(true, true);
        let block_hash = mutated_block.block_hash();

        assert!(receive_block(&mut node, mutated_block, PEER_TEST).is_ok());

        // The retry goes to a replacement peer, not the offender
        let retry_peer = node
            .inflight
            .get(&InflightRequests::Blocks(block_hash))
            .map(|req| req.0);
        assert!(matches!(retry_peer, Some(peer) if peer != PEER_TEST));

        // After the retry, the request is registered against the replacement peer.
        // User requests get no further automatic retry when that peer is lost,
        // so the request fails instead of staying open forever.
        if let Some(retry_peer) = retry_peer {
            // Unrelated fixture inflight redos may error; they are not what this
            // test measures, so the result is ignored.
            let idx = match node.peers.get(&retry_peer) {
                Some(peer) => peer.address.id,
                None => 0,
            };
            let _ = node.handle_disconnection(retry_peer, idx);
        }

        assert!(
            !node
                .inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );

        // Responder dropped without a reply, notably not `Block(None)`.
        if let Some(rx) = response {
            assert!(rx.await.is_err());
        }
    }

    #[tokio::test]
    async fn test_block_proof_mutated_block_user_request_without_peers_fails_request() {
        let (mut node, mutated_block, response) = setup_test(true, true);
        let block_hash = mutated_block.block_hash();

        // No peer is left to retry the request with
        node.peers.clear();

        let result = receive_block(&mut node, mutated_block, PEER_TEST);

        // The misbehavior is still reported
        assert!(matches!(result, Err(WireError::PeerMisbehaving)));

        // The user request was failed, instead of being left open forever
        assert!(
            !node
                .inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );

        // The user got a `None` block, and no retry was armed
        match response.unwrap().await.unwrap() {
            NodeResponse::Block(None) => {}
            other => panic!("expected NodeResponse::Block(None), got {other:?}"),
        }
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );
    }

    #[tokio::test]
    async fn test_block_proof_mutated_block_manual_peer_retries_with_another_peer() {
        let (mut node, mutated_block, response) = setup_test(true, true);
        let block_hash = mutated_block.block_hash();

        // The offender is manual (exempt from bans, so still `Ready` and eligible) AND
        // the strictly fastest peer, so only an explicit exclusion can skip it. Message
        // level proof: the offender receives nothing, and the retry `GetBlock` goes to
        // the other peer.
        for (id, other) in node.peers.iter_mut() {
            if *id != PEER_TEST {
                for _ in 0..200 {
                    other.message_times.add(10_000.0);
                }
            }
        }
        node.peers.get_mut(&PEER_TEST).unwrap().kind = ConnectionKind::Manual;

        // Precondition: the offender has the lowest latency
        let offender_latency = node
            .peers
            .get(&PEER_TEST)
            .unwrap()
            .message_times
            .value()
            .expect("offender has latency samples");
        for (id, peer) in node.peers.iter() {
            if *id != PEER_TEST {
                let latency = peer
                    .message_times
                    .value()
                    .expect("peer has latency samples");
                assert!(offender_latency < latency);
            }
        }

        // Capture everything sent to each peer
        let mut receivers = Vec::new();
        for (id, peer) in node.peers.iter_mut() {
            let (sender, receiver) = unbounded_channel();
            peer.channel = sender;
            receivers.push((*id, receiver));
        }

        let result = receive_block(&mut node, mutated_block, PEER_TEST);

        assert!(result.is_ok());

        // The retry went to another peer
        let inflight = node
            .inflight
            .get(&InflightRequests::Blocks(block_hash))
            .expect("retry armed");
        assert_ne!(inflight.0, PEER_TEST);

        // The user request stays open
        assert!(
            node.inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );

        // Nothing was sent to the offender, despite being manual (no ban) and fastest
        let retry_peer = inflight.0;
        for (id, receiver) in receivers.iter_mut() {
            if *id == PEER_TEST {
                assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
            }
        }

        // Exactly one `GetBlock` was sent, to the retry peer
        let (_, receiver) = receivers
            .iter_mut()
            .find(|(id, _)| *id == retry_peer)
            .expect("retry peer has a receiver");
        assert!(matches!(
            receiver.try_recv(),
            Ok(NodeRequest::GetBlock(blocks, _)) if blocks == vec![block_hash]
        ));
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        // Keep the responder alive for the whole test
        assert!(response.is_some());
    }

    #[tokio::test]
    async fn test_block_proof_mutated_block_only_manual_peer_fails_request() {
        let (mut node, mutated_block, response) = setup_test(true, true);
        let block_hash = mutated_block.block_hash();

        // The offending peer is manual and excluded from retries, and there's
        // no other peer: the request is failed instead of hanging forever
        node.peers.get_mut(&PEER_TEST).unwrap().kind = ConnectionKind::Manual;
        node.peers.retain(|id, _| *id == PEER_TEST);

        // Capture everything sent to the offender
        let (sender, mut messages) = unbounded_channel();
        node.peers.get_mut(&PEER_TEST).unwrap().channel = sender;

        let result = receive_block(&mut node, mutated_block, PEER_TEST);

        // Nothing was sent to the offender (manual peers don't get a `Shutdown` either)
        assert!(matches!(messages.try_recv(), Err(TryRecvError::Empty)));

        assert!(matches!(result, Err(WireError::PeerMisbehaving)));

        // The user request was failed with a `None` block
        assert!(
            !node
                .inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );

        match response.unwrap().await.unwrap() {
            NodeResponse::Block(None) => {}
            other => panic!("expected NodeResponse::Block(None), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_and_retry_user_request_unmutated_block_returns_continue()
     {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::None);
        let block_hash = block.block_hash();

        let outcome = node
            .enforce_block_structure_check_and_retry_user_request(&block, PEER_TEST)
            .unwrap();

        match outcome {
            BlockStructureCheckOutcome::Continue => {}
            BlockStructureCheckOutcome::UserRequestRetried => {
                panic!("expected Continue, got UserRequestRetried")
            }
        }

        // Nothing should change: no inflight entry and no ban
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );
        assert_eq!(node.peers.get(&PEER_TEST).unwrap().state, PeerStatus::Ready);
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_and_retry_user_request_mutated_user_request_retries()
     {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::MerkleRoot);
        let block_hash = block.block_hash();

        let (tx, _rx) = oneshot::channel::<NodeResponse>();
        node.inflight_user_requests.insert(
            UserRequest::Block(block_hash),
            (PEER_TEST, Instant::now(), tx),
        );

        let outcome = node
            .enforce_block_structure_check_and_retry_user_request(&block, PEER_TEST)
            .unwrap();

        match outcome {
            BlockStructureCheckOutcome::UserRequestRetried => {}
            BlockStructureCheckOutcome::Continue => {
                panic!("expected UserRequestRetried, got Continue")
            }
        }

        // The peer is banned and the block is retried elsewhere
        assert_eq!(
            node.peers.get(&PEER_TEST).unwrap().state,
            PeerStatus::Banned
        );
        assert!(
            node.inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );

        // The user request stays open
        assert!(
            node.inflight_user_requests
                .contains_key(&UserRequest::Block(block_hash))
        );
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_and_retry_user_request_mutated_no_user_request_errors()
     {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::MerkleRoot);
        let block_hash = block.block_hash();

        let result = node
            .enforce_block_structure_check_and_retry_user_request(&block, PEER_TEST)
            .unwrap_err();

        assert!(matches!(result, WireError::PeerMisbehaving));
        assert_eq!(
            node.peers.get(&PEER_TEST).unwrap().state,
            PeerStatus::Banned
        );

        // No retry should happen
        assert!(
            !node
                .inflight
                .contains_key(&InflightRequests::Blocks(block_hash))
        );
    }

    #[tokio::test]
    async fn test_block_proof_authentic_invalid_user_request_replies_to_user() {
        let mut node: TestNode = setup_unit_node();

        // An authentic block that is consensus-invalid
        let block = synthetic_block(BlockDefect::TooManySigops);
        assert!(matches!(
            block.check_block_structure(),
            Err(BlockExtError::TooManySigops)
        ));

        let block_hash = block.block_hash();
        node.inflight.insert(
            InflightRequests::Blocks(block_hash),
            (PEER_TEST, Instant::now()),
        );

        let (tx, rx) = oneshot::channel::<NodeResponse>();
        node.inflight_user_requests.insert(
            UserRequest::Block(block_hash),
            (PEER_TEST, Instant::now(), tx),
        );

        receive_block(&mut node, block, PEER_TEST).unwrap();

        // The user gets the block as is, and the peer isn't punished: the block is authentic
        match rx.await.unwrap() {
            NodeResponse::Block(Some(b)) => assert_eq!(b.block_hash(), block_hash),
            other => panic!("expected NodeResponse::Block(Some(_)), got {other:?}"),
        }
        assert_eq!(node.peers.get(&PEER_TEST).unwrap().state, PeerStatus::Ready);
    }
}

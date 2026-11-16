// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use bitcoin::Block;
    use bitcoin::Network;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;
    use floresta_chain::extensions::BlockExt;
    use floresta_chain::extensions::BlockExtError;
    use tokio::sync::oneshot;
    use tokio::sync::oneshot::Receiver;

    use crate::node::InflightRequests;
    use crate::node::PeerStatus;
    use crate::node::UtreexoNode;
    use crate::node::sync_ctx::SyncNode;
    use crate::node_handle::NodeResponse;
    use crate::node_handle::UserRequest;
    use crate::p2p_wire::error::WireError;
    use crate::p2p_wire::tests::utils::BlockDefect;
    use crate::p2p_wire::tests::utils::PEER_TEST;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::SetupNodeArgs;
    use crate::p2p_wire::tests::utils::setup_node;
    use crate::p2p_wire::tests::utils::setup_unit_node;
    use crate::p2p_wire::tests::utils::signet_blocks;
    use crate::p2p_wire::tests::utils::signet_headers;
    use crate::p2p_wire::tests::utils::synthetic_block;

    type TestNode = UtreexoNode<Arc<ChainState<FlatChainStore>>, SyncNode>;

    type TestSetup = (TestNode, Block, Option<Receiver<NodeResponse>>);

    fn setup_test(is_user_request: bool) -> TestSetup {
        let datadir = format!("./tmp-db/{}.blocks", rand::random::<u32>());
        let blocks = signet_blocks();
        let headers = signet_headers();

        let peers = vec![
            PeerData::new(Vec::new(), blocks.clone(), HashMap::new()),
            PeerData::new(headers.clone(), blocks.clone(), HashMap::new()),
        ];
        let args = SetupNodeArgs::new(peers, false, Network::Signet, datadir, 9);
        let mut node = setup_node(args);

        let block_hash = headers[1].block_hash();
        let block = blocks.get(&block_hash).unwrap().clone();

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
    fn receive_block(
        node: &mut UtreexoNode<Arc<ChainState<FlatChainStore>>, SyncNode>,
        block: Block,
        peer: u32,
    ) -> Result<(), WireError> {
        if let Some(block) = node.handle_received_block(block, peer)? {
            node.request_block_proof(block, peer)?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_enforce_block_structure_check_unmutated_block_is_ok() {
        let mut node: TestNode = setup_unit_node();
        let block = synthetic_block(BlockDefect::None);

        node.enforce_block_structure_check(&block, PEER_TEST)
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
            .enforce_block_structure_check(&block, PEER_TEST)
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
            .enforce_block_structure_check(&block, PEER_TEST)
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
        node.enforce_block_structure_check(&block, PEER_TEST)
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
        node.enforce_block_structure_check(&block, PEER_TEST)
            .unwrap();

        assert_eq!(node.peers.get(&PEER_TEST).unwrap().state, PeerStatus::Ready);
    }

    #[tokio::test]
    async fn test_block_proof_coinbase_only_block_stores_and_skips_proof_request() {
        let (mut node, block, _) = setup_test(false);
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
        let (mut node, _, _) = setup_test(false);
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
        let (mut node, block, response) = setup_test(true);
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
}

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
    use tokio::sync::oneshot;
    use tokio::sync::oneshot::Receiver;

    use crate::node::InflightRequests;
    use crate::node::UtreexoNode;
    use crate::node::sync_ctx::SyncNode;
    use crate::node_handle::NodeResponse;
    use crate::node_handle::UserRequest;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::SetupNodeArgs;
    use crate::p2p_wire::tests::utils::setup_node;
    use crate::p2p_wire::tests::utils::signet_blocks;
    use crate::p2p_wire::tests::utils::signet_headers;

    const PEER_TEST: u32 = 0;

    type TestSetup = (
        UtreexoNode<Arc<ChainState<FlatChainStore>>, SyncNode>,
        Block,
        Option<Receiver<NodeResponse>>,
    );

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

    #[tokio::test]
    async fn test_block_proof_coinbase_only_block_stores_and_skips_proof_request() {
        let (mut node, block, _) = setup_test(false);
        let block_hash = block.block_hash();

        node.request_block_proof(block, PEER_TEST).unwrap();

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
    async fn test_block_proof_valid_block_user_request_replies_to_user() {
        let (mut node, block, response) = setup_test(true);
        let block_hash = block.block_hash();

        node.request_block_proof(block, PEER_TEST).unwrap();

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

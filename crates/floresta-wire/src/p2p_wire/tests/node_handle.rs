// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use bitcoin::Network;
    use bitcoin::p2p::ServiceFlags;
    use floresta_common::service_flags;
    use tokio::time::Duration;
    use tokio::time::timeout;

    use crate::node::PeerStatus;
    use crate::node_interface::ChainMethods;
    use crate::node_interface::NetworkMethods;
    use crate::node_interface::NodeConfigMethods;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::setup_node_handle_test;
    use crate::p2p_wire::tests::utils::signet_blocks;
    use crate::p2p_wire::tests::utils::signet_headers;
    use crate::p2p_wire::transport::TransportProtocol;

    #[tokio::test]
    async fn node_handle_get_config_returns_running_node_config() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let harness = setup_node_handle_test(Vec::new(), false, Network::Signet, &datadir, 0).await;

        let config = timeout(Duration::from_secs(5), harness.handle.get_config())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(config.network, Network::Signet);
        assert_eq!(config.datadir, PathBuf::from(datadir));
        assert!(!config.pow_fraud_proofs);

        harness.shutdown().await;
    }

    #[tokio::test]
    async fn node_handle_get_peer_info_returns_mocked_peer_data() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new());
        let harness = setup_node_handle_test(vec![peer], false, Network::Signet, &datadir, 0).await;
        harness.wait_for_peers(1).await;

        let connection_count = timeout(
            Duration::from_secs(5),
            harness.handle.get_connection_count(),
        )
        .await
        .unwrap()
        .unwrap();
        let peer_info = timeout(Duration::from_secs(5), harness.handle.get_peer_info())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(connection_count, 1);
        assert_eq!(peer_info.len(), 1);

        let peer = &peer_info[0];
        assert_eq!(peer.id, 0);
        assert_eq!(peer.user_agent, "node_test");
        assert_eq!(peer.state, PeerStatus::Ready);
        assert_eq!(peer.transport_protocol, TransportProtocol::V2);
        assert!(peer.services.has(ServiceFlags::NETWORK));
        assert!(peer.services.has(service_flags::UTREEXO.into()));

        harness.shutdown().await;
    }

    #[tokio::test]
    async fn node_handle_get_block_returns_mocked_peer_block() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let headers = signet_headers();
        let blocks = signet_blocks();
        let expected_block = blocks.get(&headers[1].block_hash()).unwrap().clone();
        let peer = PeerData::new(Vec::new(), blocks, HashMap::new());
        let harness = setup_node_handle_test(vec![peer], false, Network::Signet, &datadir, 0).await;
        harness.wait_for_peers(1).await;

        let block = timeout(
            Duration::from_secs(5),
            harness.handle.get_block(expected_block.block_hash()),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();

        assert_eq!(block, expected_block);

        harness.shutdown().await;
    }
}

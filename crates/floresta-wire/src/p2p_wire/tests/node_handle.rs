// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use bitcoin::Amount;
    use bitcoin::FilterHash;
    use bitcoin::FilterHeader;
    use bitcoin::Network;
    use bitcoin::OutPoint;
    use bitcoin::ScriptBuf;
    use bitcoin::Sequence;
    use bitcoin::Transaction;
    use bitcoin::TxIn;
    use bitcoin::TxOut;
    use bitcoin::Txid;
    use bitcoin::absolute;
    use bitcoin::hashes::Hash;
    use bitcoin::p2p::message_filter::CFHeaders;
    use bitcoin::transaction::Version;
    use tokio::sync::mpsc::unbounded_channel;
    use tokio::time::Duration;
    use tokio::time::timeout;

    use crate::bitcoin_socket_addr::BitcoinSocketAddr;
    use crate::node::NodeNotification;
    use crate::node::PeerStatus;
    use crate::node_handle::NodeHandle;
    use crate::node_interface::ChainMethods;
    use crate::node_interface::MempoolMethods;
    use crate::node_interface::NetworkMethods;
    use crate::node_interface::NodeConfigMethods;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::setup_node_handle_test;
    use crate::p2p_wire::tests::utils::signet_blocks;
    use crate::p2p_wire::tests::utils::signet_headers;
    use crate::p2p_wire::transport::TransportProtocol;

    fn sample_transaction() -> Transaction {
        Transaction {
            version: Version::ONE,
            lock_time: absolute::LockTime::ZERO,
            input: vec![TxIn {
                previous_output: OutPoint {
                    txid: Txid::all_zeros(),
                    vout: 0,
                },
                script_sig: ScriptBuf::new(),
                sequence: Sequence::MAX,
                witness: bitcoin::Witness::new(),
            }],
            output: vec![TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: ScriptBuf::new(),
            }],
        }
    }

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
        assert_eq!(
            peer.services_names,
            vec![
                "NETWORK",
                "WITNESS",
                "COMPACT_FILTERS",
                "UTREEXO",
                "UTREEXO_ARCHIVE"
            ]
        );
        assert_eq!(peer.services, "0000000000003049");

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

    #[tokio::test]
    async fn node_handle_get_cfilters_headers_returns_mocked_peer_headers() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let stop_hash = signet_headers()[1].block_hash();
        let cfheaders = CFHeaders {
            filter_type: 0,
            stop_hash,
            previous_filter_header: FilterHeader::all_zeros(),
            filter_hashes: vec![FilterHash::all_zeros()],
        };
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new())
            .with_cfilter_headers(cfheaders.clone());
        let harness = setup_node_handle_test(vec![peer], false, Network::Signet, &datadir, 0).await;
        harness.wait_for_peers(1).await;

        let response = timeout(
            Duration::from_secs(5),
            harness.handle.get_cfilters_headers(1, stop_hash),
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(response, cfheaders);

        harness.shutdown().await;
    }

    #[tokio::test]
    async fn node_handle_mempool_methods_use_local_mempool_and_mocked_peer() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let transaction = sample_transaction();
        let txid = transaction.compute_txid();
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new())
            .with_transaction(transaction.clone());
        let harness = setup_node_handle_test(vec![peer], false, Network::Signet, &datadir, 0).await;
        harness.wait_for_peers(1).await;

        let broadcast_txid = timeout(
            Duration::from_secs(5),
            harness.handle.broadcast_transaction(transaction.clone()),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();
        let fetched_transaction = timeout(
            Duration::from_secs(5),
            harness.handle.get_mempool_transaction(txid),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();

        assert_eq!(broadcast_txid, txid);
        assert_eq!(fetched_transaction, transaction);

        harness.shutdown().await;
    }

    #[tokio::test]
    async fn node_handle_network_methods_return_node_status_and_control_peers() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new());
        let harness = setup_node_handle_test(vec![peer], false, Network::Signet, &datadir, 0).await;
        harness.wait_for_peers(1).await;

        let add_addr: BitcoinSocketAddr = "127.0.0.1:18444".parse().unwrap();
        let onetry_addr: BitcoinSocketAddr = "127.0.0.1:18445".parse().unwrap();
        let connected_addr = harness.handle.get_peer_info().await.unwrap()[0]
            .address
            .clone();

        let ping = timeout(Duration::from_secs(5), harness.handle.ping())
            .await
            .unwrap()
            .unwrap();
        let add = timeout(
            Duration::from_secs(5),
            harness.handle.add_peer(add_addr.clone(), false),
        )
        .await
        .unwrap()
        .unwrap();
        let stats = timeout(Duration::from_secs(5), harness.handle.get_addrman_info())
            .await
            .unwrap()
            .unwrap();
        let remove = timeout(Duration::from_secs(5), harness.handle.remove_peer(add_addr))
            .await
            .unwrap()
            .unwrap();
        let onetry = timeout(
            Duration::from_secs(5),
            harness.handle.onetry_peer(onetry_addr, false),
        )
        .await
        .unwrap()
        .unwrap();
        let disconnect = timeout(
            Duration::from_secs(5),
            harness.handle.disconnect_peer(connected_addr),
        )
        .await
        .unwrap()
        .unwrap();

        assert!(ping);
        assert!(add);
        assert_eq!(stats.ipv4.total(), stats.ipv4.new + stats.ipv4.tried);
        assert!(remove);
        assert!(onetry);
        assert!(disconnect);

        harness.wait_for_peers(0).await;
        harness.shutdown().await;
    }

    #[tokio::test]
    async fn node_handle_returns_error_when_node_receiver_is_dropped() {
        let (node_sender, node_receiver) = unbounded_channel::<NodeNotification>();
        drop(node_receiver);

        let handle = NodeHandle::new(node_sender);

        let err = timeout(Duration::from_secs(5), handle.get_config())
            .await
            .unwrap()
            .unwrap_err();

        assert_eq!(err.to_string(), "channel closed");
    }

    #[tokio::test]
    async fn node_handle_block_request_stays_pending_when_peer_disconnects() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let headers = signet_headers();
        let block_hash = headers[1].block_hash();
        let peer =
            PeerData::disconnecting_on_block_request(Vec::new(), HashMap::new(), HashMap::new());
        let harness = setup_node_handle_test(vec![peer], false, Network::Signet, &datadir, 0).await;
        harness.wait_for_peers(1).await;

        let request = timeout(
            Duration::from_millis(500),
            harness.handle.get_block(block_hash),
        )
        .await;

        assert!(request.is_err());

        harness.shutdown().await;
    }

    #[tokio::test]
    async fn node_handle_block_request_stays_pending_when_peer_ignores_request() {
        let datadir = format!("./tmp-db/{}.node_handle", rand::random::<u32>());
        let headers = signet_headers();
        let block_hash = headers[1].block_hash();
        let peer = PeerData::ignoring_block_requests(Vec::new(), HashMap::new(), HashMap::new());
        let harness = setup_node_handle_test(vec![peer], false, Network::Signet, &datadir, 0).await;
        harness.wait_for_peers(1).await;

        let request = timeout(
            Duration::from_millis(500),
            harness.handle.get_block(block_hash),
        )
        .await;

        assert!(request.is_err());

        harness.shutdown().await;
    }
}

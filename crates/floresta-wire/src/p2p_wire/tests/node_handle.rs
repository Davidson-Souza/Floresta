// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future::Future;
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
    use tokio::sync::oneshot;
    use tokio::sync::oneshot::error::RecvError;
    use tokio::task;
    use tokio::task::JoinHandle;
    use tokio::time::Duration;
    use tokio::time::sleep;
    use tokio::time::timeout;

    use crate::bitcoin_socket_addr::BitcoinSocketAddr;
    use crate::node::NodeNotification;
    use crate::node::PeerStatus;
    use crate::node::chain_selector_ctx::ChainSelector;
    use crate::node::running_ctx::RunningNode;
    use crate::node::sync_ctx::SyncNode;
    use crate::node_handle::NodeHandle;
    use crate::node_interface::ChainMethods;
    use crate::node_interface::MempoolMethods;
    use crate::node_interface::NetworkMethods;
    use crate::node_interface::NodeConfigMethods;
    use crate::p2p_wire::tests::utils::PeerData;
    use crate::p2p_wire::tests::utils::SetupNodeArgs;
    use crate::p2p_wire::tests::utils::setup_node;
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

    const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
    const START_TIMEOUT: Duration = Duration::from_secs(100);

    #[derive(Clone, Copy, Debug)]
    enum NodeFlavor {
        ChainSelector,
        Sync,
        Running,
    }

    impl NodeFlavor {
        // Finish the running node's startup before starting the deliberately stalled contexts.
        const ALL: [Self; 3] = [Self::Running, Self::ChainSelector, Self::Sync];

        fn name(self) -> &'static str {
            match self {
                Self::ChainSelector => "chain_selector",
                Self::Sync => "sync",
                Self::Running => "running",
            }
        }
    }

    struct NodeHandleTestHarness {
        flavor: NodeFlavor,
        handle: NodeHandle,
        datadir: PathBuf,
        node_task: JoinHandle<()>,
        ready_receiver: Option<oneshot::Receiver<()>>,
    }

    impl NodeHandleTestHarness {
        fn start(flavor: NodeFlavor, peers: Vec<PeerData>, num_blocks: usize) -> Self {
            let datadir = format!(
                "./tmp-db/{}.{}.node_handle",
                rand::random::<u32>(),
                flavor.name()
            );
            let args =
                SetupNodeArgs::new(peers, false, Network::Signet, datadir.clone(), num_blocks);

            let (handle, node_task, ready_receiver) = match flavor {
                NodeFlavor::ChainSelector => {
                    let mut node = setup_node::<ChainSelector>(args);
                    let handle = node.get_handle();
                    let node_task = task::spawn(async move {
                        node.run().await.unwrap();
                    });
                    (handle, node_task, None)
                }
                NodeFlavor::Sync => {
                    let node = setup_node::<SyncNode>(args);
                    let handle = node.get_handle();
                    let node_task = task::spawn(async move {
                        node.run(|_| {}).await;
                    });
                    (handle, node_task, None)
                }
                NodeFlavor::Running => {
                    let node = setup_node::<RunningNode>(args);
                    let handle = node.get_handle();
                    let (stop_sender, _stop_receiver) = oneshot::channel();
                    let (ready_sender, ready_receiver) = oneshot::channel();
                    let node_task = task::spawn(async move {
                        node.run_with_ready_signal(stop_sender, ready_sender).await;
                    });
                    (handle, node_task, Some(ready_receiver))
                }
            };

            Self {
                flavor,
                handle,
                datadir: datadir.into(),
                node_task,
                ready_receiver,
            }
        }

        async fn wait_until_ready(&mut self) {
            let Some(ready_receiver) = self.ready_receiver.take() else {
                return;
            };

            timeout(START_TIMEOUT, ready_receiver)
                .await
                .unwrap_or_else(|_| panic!("{:?} did not enter its main loop", self.flavor))
                .unwrap_or_else(|_| panic!("{:?} stopped during startup", self.flavor));
        }

        async fn wait_for_peers(&self, expected: usize) {
            timeout(REQUEST_TIMEOUT, async {
                loop {
                    if request(self.handle.get_peer_info()).await.len() == expected {
                        break;
                    }
                    sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap_or_else(|_| {
                panic!("{:?} did not reach {expected} connected peers", self.flavor)
            });
        }

        async fn wait_for_peer_disconnect(&self, address: &BitcoinSocketAddr) {
            timeout(REQUEST_TIMEOUT, async {
                loop {
                    let peers = request(self.handle.get_peer_info()).await;
                    if peers.iter().all(|peer| &peer.address != address) {
                        break;
                    }
                    sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap_or_else(|_| panic!("{:?} did not disconnect peer {address}", self.flavor));
        }

        async fn shutdown(self) {
            self.node_task.abort();
            let result = timeout(REQUEST_TIMEOUT, self.node_task)
                .await
                .unwrap_or_else(|_| panic!("{:?} task did not abort", self.flavor));

            if let Err(error) = result {
                assert!(error.is_cancelled(), "{:?}: {error}", self.flavor);
            }
        }
    }

    async fn setup_node_flavors(
        peers: Vec<PeerData>,
        num_blocks: usize,
    ) -> Vec<NodeHandleTestHarness> {
        let mut nodes = Vec::with_capacity(NodeFlavor::ALL.len());

        for flavor in NodeFlavor::ALL {
            let (peers, num_blocks) = match flavor {
                NodeFlavor::ChainSelector => (
                    peers
                        .clone()
                        .into_iter()
                        .map(PeerData::ignoring_header_requests)
                        .collect(),
                    num_blocks,
                ),
                NodeFlavor::Sync => (peers.clone(), num_blocks),
                NodeFlavor::Running if peers.is_empty() => (
                    vec![PeerData::new(Vec::new(), HashMap::new(), HashMap::new())],
                    0,
                ),
                NodeFlavor::Running => (peers.clone(), 0),
            };

            let mut node = NodeHandleTestHarness::start(flavor, peers, num_blocks);
            node.wait_until_ready().await;
            nodes.push(node);
        }

        nodes
    }

    async fn shutdown_nodes(nodes: Vec<NodeHandleTestHarness>) {
        for node in nodes {
            node.shutdown().await;
        }
    }

    async fn request<T>(request: impl Future<Output = Result<T, RecvError>>) -> T {
        timeout(REQUEST_TIMEOUT, request).await.unwrap().unwrap()
    }

    #[tokio::test]
    async fn node_handle_get_config_from_each_node_flavor() {
        let nodes = setup_node_flavors(Vec::new(), 1).await;

        for node in &nodes {
            let config = request(node.handle.get_config()).await;

            assert_eq!(config.network, Network::Signet, "{:?}", node.flavor);
            assert_eq!(config.datadir, node.datadir, "{:?}", node.flavor);
            assert!(!config.pow_fraud_proofs, "{:?}", node.flavor);
        }

        shutdown_nodes(nodes).await;
    }

    #[tokio::test]
    async fn node_handle_get_peer_info_from_each_node_flavor() {
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new());
        let nodes = setup_node_flavors(vec![peer], 1).await;

        for node in &nodes {
            node.wait_for_peers(1).await;

            let connection_count = request(node.handle.get_connection_count()).await;
            let peer_info = request(node.handle.get_peer_info()).await;

            assert_eq!(connection_count, 1, "{:?}", node.flavor);
            assert_eq!(peer_info.len(), 1, "{:?}", node.flavor);

            let peer = &peer_info[0];
            assert_eq!(peer.id, 0, "{:?}", node.flavor);
            assert_eq!(peer.user_agent, "node_test", "{:?}", node.flavor);
            assert_eq!(peer.state, PeerStatus::Ready, "{:?}", node.flavor);
            assert_eq!(
                peer.transport_protocol,
                TransportProtocol::V2,
                "{:?}",
                node.flavor
            );
            assert_eq!(
                peer.services_names,
                vec![
                    "NETWORK",
                    "WITNESS",
                    "COMPACT_FILTERS",
                    "UTREEXO",
                    "UTREEXO_ARCHIVE"
                ],
                "{:?}",
                node.flavor
            );
            assert_eq!(peer.services, "0000000000003049", "{:?}", node.flavor);
        }

        shutdown_nodes(nodes).await;
    }

    #[tokio::test]
    async fn node_handle_get_block_from_each_node_flavor() {
        let headers = signet_headers();
        let block_hash = headers[1].block_hash();
        let expected_block = signet_blocks().remove(&block_hash).unwrap();
        let blocks = HashMap::from([(block_hash, expected_block.clone())]);
        let peer = PeerData::new(Vec::new(), blocks, HashMap::new());
        let nodes = setup_node_flavors(vec![peer], 2).await;

        for node in &nodes {
            node.wait_for_peers(1).await;

            let block = request(node.handle.get_block(block_hash)).await.unwrap();
            assert_eq!(block, expected_block, "{:?}", node.flavor);
        }

        shutdown_nodes(nodes).await;
    }

    #[tokio::test]
    async fn node_handle_get_cfilters_headers_from_each_node_flavor() {
        let stop_hash = signet_headers()[1].block_hash();
        let cfheaders = CFHeaders {
            filter_type: 0,
            stop_hash,
            previous_filter_header: FilterHeader::all_zeros(),
            filter_hashes: vec![FilterHash::all_zeros()],
        };
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new())
            .with_cfilter_headers(cfheaders.clone());
        let nodes = setup_node_flavors(vec![peer], 1).await;

        for node in &nodes {
            node.wait_for_peers(1).await;

            let response = request(node.handle.get_cfilters_headers(1, stop_hash)).await;
            assert_eq!(response, cfheaders, "{:?}", node.flavor);
        }

        shutdown_nodes(nodes).await;
    }

    #[tokio::test]
    async fn node_handle_mempool_methods_work_with_each_node_flavor() {
        let transaction = sample_transaction();
        let txid = transaction.compute_txid();
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new())
            .with_transaction(transaction.clone());
        let nodes = setup_node_flavors(vec![peer], 1).await;

        for node in &nodes {
            node.wait_for_peers(1).await;

            let broadcast_txid = request(node.handle.broadcast_transaction(transaction.clone()))
                .await
                .unwrap();
            let fetched_transaction = request(node.handle.get_mempool_transaction(txid))
                .await
                .unwrap();

            assert_eq!(broadcast_txid, txid, "{:?}", node.flavor);
            assert_eq!(fetched_transaction, transaction, "{:?}", node.flavor);
        }

        shutdown_nodes(nodes).await;
    }

    #[tokio::test]
    async fn node_handle_network_methods_work_with_each_node_flavor() {
        let peer = PeerData::new(Vec::new(), HashMap::new(), HashMap::new());
        let nodes = setup_node_flavors(vec![peer], 1).await;
        let add_addr: BitcoinSocketAddr = "127.0.0.1:18444".parse().unwrap();
        let onetry_addr: BitcoinSocketAddr = "127.0.0.1:18445".parse().unwrap();

        for node in &nodes {
            node.wait_for_peers(1).await;

            let connected_addr = request(node.handle.get_peer_info()).await[0]
                .address
                .clone();
            let ping = request(node.handle.ping()).await;
            let add = request(node.handle.add_peer(add_addr.clone(), false)).await;
            let stats = request(node.handle.get_addrman_info()).await;
            let remove = request(node.handle.remove_peer(add_addr.clone())).await;
            let onetry = request(node.handle.onetry_peer(onetry_addr.clone(), false)).await;
            let disconnect = request(node.handle.disconnect_peer(connected_addr.clone())).await;

            assert!(ping, "{:?}", node.flavor);
            assert!(add, "{:?}", node.flavor);
            assert_eq!(
                stats.ipv4.total(),
                stats.ipv4.new + stats.ipv4.tried,
                "{:?}",
                node.flavor
            );
            assert!(remove, "{:?}", node.flavor);
            assert!(onetry, "{:?}", node.flavor);
            assert!(disconnect, "{:?}", node.flavor);

            node.wait_for_peer_disconnect(&connected_addr).await;
        }

        shutdown_nodes(nodes).await;
    }

    #[tokio::test]
    async fn node_handle_returns_error_when_node_receiver_is_dropped() {
        let (node_sender, node_receiver) = unbounded_channel::<NodeNotification>();
        drop(node_receiver);

        let handle = NodeHandle::new(node_sender);
        let err = timeout(REQUEST_TIMEOUT, handle.get_config())
            .await
            .unwrap()
            .unwrap_err();

        assert_eq!(err.to_string(), "channel closed");
    }

    #[tokio::test]
    async fn node_handle_block_request_errors_when_each_node_flavor_loses_peer() {
        let block_hash = signet_headers()[2].block_hash();
        let peer = PeerData::disconnecting_on_block_request(
            Vec::new(),
            HashMap::new(),
            HashMap::new(),
            block_hash,
        );
        let nodes = setup_node_flavors(vec![peer], 1).await;

        for node in &nodes {
            node.wait_for_peers(1).await;

            let response = timeout(REQUEST_TIMEOUT, node.handle.get_block(block_hash))
                .await
                .unwrap();
            assert!(response.is_err(), "{:?}", node.flavor);
        }

        shutdown_nodes(nodes).await;
    }

    #[tokio::test]
    async fn node_handle_block_request_stays_pending_when_each_node_flavor_peer_ignores_it() {
        let block_hash = signet_headers()[2].block_hash();
        let peer = PeerData::ignoring_block_requests(
            Vec::new(),
            HashMap::new(),
            HashMap::new(),
            block_hash,
        );
        let nodes = setup_node_flavors(vec![peer], 1).await;

        for node in &nodes {
            node.wait_for_peers(1).await;

            let response = timeout(
                Duration::from_millis(500),
                node.handle.get_block(block_hash),
            )
            .await;
            assert!(response.is_err(), "{:?}", node.flavor);
        }

        shutdown_nodes(nodes).await;
    }
}

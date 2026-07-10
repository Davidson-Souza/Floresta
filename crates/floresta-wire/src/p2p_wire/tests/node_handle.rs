// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bitcoin::Network;
    use tokio::time::Duration;
    use tokio::time::timeout;

    use crate::node_interface::NodeConfigMethods;
    use crate::p2p_wire::tests::utils::setup_node_handle_test;

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
}

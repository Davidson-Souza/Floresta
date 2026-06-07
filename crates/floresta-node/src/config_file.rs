// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::FlorestadError;

#[derive(Default, Debug, Deserialize)]
pub struct Wallet {
    pub xpubs: Option<Vec<String>>,
    pub descriptors: Option<Vec<String>>,
    pub addresses: Option<Vec<String>>,
}

#[cfg(feature = "json-rpc")]
#[derive(Default, Debug, Deserialize)]
pub struct Rpc {
    pub user: Option<String>,
    pub password: Option<String>,
    #[serde(default)]
    pub auth: Vec<String>,
    pub cookie_file: Option<String>,
}

#[derive(Default, Debug, Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    pub wallet: Wallet,
    #[cfg(feature = "json-rpc")]
    #[serde(default)]
    pub rpc: Rpc,
}

impl ConfigFile {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, FlorestadError> {
        let config_file = fs::read_to_string(path.as_ref())?;

        Ok(toml::from_str(&config_file)?)
    }
}

#[cfg(all(test, feature = "json-rpc"))]
mod tests {
    use super::ConfigFile;

    /// Both sections are optional, so a config that only sets `[rpc]` parses
    /// instead of being thrown away for a missing `[wallet]`.
    #[test]
    fn parses_a_config_with_only_an_rpc_section() {
        let config: ConfigFile = toml::from_str(
            r#"
            [rpc]
            user = "alice"
            password = "hunter2"
            "#,
        )
        .expect("an `[rpc]`-only config should parse");

        assert_eq!(config.rpc.user.as_deref(), Some("alice"));
        assert_eq!(config.rpc.password.as_deref(), Some("hunter2"));
        assert!(config.wallet.xpubs.is_none());
    }

    /// The reverse still holds: `[wallet]` alone is fine without `[rpc]`.
    #[test]
    fn parses_a_config_with_only_a_wallet_section() {
        let config: ConfigFile = toml::from_str(
            r#"
            [wallet]
            xpubs = ["xpub661MyMwAqRbcFtXgS5sYJABqqG9YLmC4Q1Rdap9gSE8NqtwybGhePY2gZ29ESFjqJoCu1Rupje8YtGqsefD265TMg7usUDFdp6W1EGMcet8"]
            "#,
        )
        .expect("a `[wallet]`-only config should parse");

        assert_eq!(config.wallet.xpubs.as_ref().map(Vec::len), Some(1));
        assert!(config.rpc.user.is_none());
    }
}

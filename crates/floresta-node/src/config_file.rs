// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use bitcoin::ScriptBuf;
use serde::Deserialize;

use crate::error::FlorestadError;

#[derive(Default, Debug, Deserialize)]
pub struct Wallet {
    pub xpubs: Option<Vec<String>>,
    pub descriptors: Option<Vec<String>>,
    pub addresses: Option<Vec<String>>,
}

#[derive(Default, Debug, Deserialize)]
pub struct ConfigFile {
    pub signet_challenge: Option<ScriptBuf>,

    #[serde(default)]
    pub wallet: Wallet,
}

impl ConfigFile {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, FlorestadError> {
        let config_file = fs::read_to_string(path.as_ref())?;

        Ok(toml::from_str(&config_file)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_signet_challenge_as_hex_script() {
        let config: ConfigFile =
            toml::from_str("signet_challenge = \"51\"").expect("valid configuration");

        assert_eq!(
            config.signet_challenge,
            Some(ScriptBuf::from_bytes(vec![0x51]))
        );
    }

    #[test]
    fn rejects_non_hex_signet_challenge() {
        let result = toml::from_str::<ConfigFile>("signet_challenge = \"not-hex\"");

        assert!(result.is_err());
    }
}

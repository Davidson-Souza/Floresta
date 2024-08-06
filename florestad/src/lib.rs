mod config_file;
mod error;
mod florestad;
#[cfg(feature = "json-rpc")]
mod json_rpc;
mod slip132;
mod wallet_input;
#[cfg(feature = "zmq-server")]
mod zmq;

#[derive(Debug, Default, Clone, Copy)]
/// Which network should we use
///
/// Use this to select one of the supported networks. They are either Bitcoin mainnet or some form
/// of test network, like testnet or regtest. The default is Bitcoin mainnet.
pub enum Network {
    #[default]
    /// Mainnet (default)
    Bitcoin,
    /// Testnet 3
    Testnet,
    /// Signet
    Signet,
    /// Regtest
    Regtest,
}

pub use florestad::Config;
pub use florestad::Florestad;
pub use florestad::AssumeUtreexoValue;

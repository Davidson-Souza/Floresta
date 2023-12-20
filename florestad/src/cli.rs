use bitcoin::BlockHash;
use clap::{arg, command, Parser, Subcommand, ValueEnum};

#[derive(Clone, Debug, ValueEnum, Default)]
pub enum Network {
    #[default]
    Bitcoin,
    Signet,
    Testnet,
    Regtest,
}

impl std::fmt::Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Network::Bitcoin => write!(f, "bitcoin"),
            Network::Signet => write!(f, "signet"),
            Network::Testnet => write!(f, "testnet"),
            Network::Regtest => write!(f, "regtest"),
        }
    }
}
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    pub config_file: Option<String>,
    /// Which network should we use
    #[arg(short, long, default_value_t=Network::Bitcoin)]
    pub network: Network,
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub debug: u8,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    #[cfg(not(feature = "experimental-p2p"))]
    /// Starts your wallet and server
    Run {
        /// Where should we store data
        #[arg(long)]
        data_dir: Option<String>,
        /// Add a xpub to our wallet
        #[arg(long)]
        wallet_xpub: Option<Vec<String>>,
        /// Add individual addresses to your wallet
        #[arg(long)]
        wallet_addresses: Option<Vec<String>>,
        /// Your rpc user, as set in Utreexod
        #[arg(long)]
        rpc_user: Option<String>,
        /// Your rpc password, as set in Utreexod
        #[arg(long)]
        rpc_password: Option<String>,
        /// The hostname:port of Utreexod
        #[arg(short, long)]
        rpc_host: Option<String>,
        #[arg(long)]
        rpc_port: Option<u32>,
        /// Whether or not we want to sync with a external provider
        #[arg(long, default_value_t = false)]
        use_batch_sync: bool,
        /// If use_batch_sync is set, this option provides which server we use
        #[arg(long)]
        batch_sync: Option<String>,
        /// Assume blocks before this one as having valid signatures, same with bitcoin core
        #[arg(long)]
        assume_valid: Option<BlockHash>,
        /// Electrum server url with port
        #[arg(short, long, default_value = "0.0.0.0:50001")]
        electrum_host: String,
    },
    #[cfg(feature = "experimental-p2p")]
    /// Starts your wallet and server
    #[command(author, version, about, long_about = None)]
    Run {
        /// Where should we store data
        #[arg(long)]
        data_dir: Option<String>,
        /// Whether to build Compact Block Filters
        ///
        /// Those filters let you query for chain data after IBD, like wallet rescan,
        /// finding an utxo, finding specific tx_ids.
        /// Will cause more disk usage
        #[arg(long = "cfilters", short = 'c', default_value_t = false)]
        cfilters: bool,
        /// What types of filters we should build. Keep in mind that each filter
        /// type you add, will eat up more disk.
        #[arg(long = "cfilter-types")]
        cfilter_types: Option<Vec<FilterType>>,
        #[arg(long, short, default_value = None)]
        /// The url of a proxy we should open p2p connections through (e.g. 127.0.0.1:9050)
        proxy: Option<String>,
        #[arg(long, short, default_value = None)]
        rescan: Option<u32>,
        /// Add a xpub to our wallet
        #[arg(long)]
        wallet_xpub: Option<Vec<String>>,
        #[arg(long)]
        wallet_descriptor: Option<Vec<String>>,
        /// Assume blocks before this one as having valid signatures, same with bitcoin core
        #[arg(long)]
        assume_valid: Option<BlockHash>,
        #[arg(long, short)]
        zmq_address: Option<String>,
        /// Electrum server url with port
        #[arg(short, long, default_value = "0.0.0.0:50001")]
        electrum_host: String,
    },
}

#[derive(Clone, Debug, ValueEnum)]
pub enum FilterType {
    Inputs,
    TxId,
    SpkPKH,
    SpkPSH,
    SpkWPKH,
    SpkWSH,
    SpkTR,
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! This is a modular-(ish) utreexo powered wallet backend and fully validating node, it's
//! developed as an experiment to showcase utreexo. This wallet also comes with an Electrum
//! server out-of-the-box, for people to try out with their favorite wallet.
//! This codebase consists of three main parts: a blockchain backend, that gets all information
//! we need from the network. An Electrum Server that talks full Electrum protocol and can be
//! used with any wallet that understands this protocol. Finally, it has the `AddressCache`,
//! a watch-only wallet that keeps track of your wallet's transactions.

// Coding conventions (lexicographically sorted)
#![deny(arithmetic_overflow)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![deny(non_camel_case_types)]
#![deny(non_snake_case)]
#![deny(non_upper_case_globals)]

mod cli;
#[cfg(unix)]
mod daemonize;
mod logger;

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::exit;
use std::sync::Arc;
use std::time::Duration;

use bitcoin::Network;
use bitcoin::Script;
use bitcoin::ScriptBuf;
use clap::Parser;
use cli::Cli;
use floresta_node::Config;
use floresta_node::Florestad;
use floresta_node::read_signet_challenge;
use floresta_node::signet_data_dir_name;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tokio::time::timeout;
use tracing::Level;
use tracing::info;

#[cfg(unix)]
use crate::daemonize::Daemon;
use crate::logger::start_logger;

fn main() {
    let params = Cli::parse();
    params.validate();

    let (datadir, config_file, signet_challenge) = resolve_data_paths(
        params.data_dir.as_deref(),
        params.network,
        params.signet_challenge,
        params.config_file,
    );

    // Create the data directory if it doesn't exist
    fs::create_dir_all(&datadir).unwrap_or_else(|e| {
        eprintln!("Could not create data dir {datadir:?}: {e}");
        exit(1);
    });

    let config = Config {
        datadir,
        disable_dns_seeds: !params.connect.is_empty() || params.disable_dns_seeds,
        network: params.network,
        signet_challenge,
        debug: params.debug,
        cfilters: !params.no_cfilters,
        proxy: params.proxy,
        assume_utreexo: !params.no_assume_utreexo,
        connect: params.connect,
        wallet_xpub: params.wallet_xpub,
        config_file,
        #[cfg(unix)]
        log_to_stdout: !params.daemon,
        #[cfg(not(unix))]
        log_to_stdout: true,
        #[cfg(unix)]
        log_to_file: params.log_to_file || params.daemon,
        #[cfg(not(unix))]
        log_to_file: params.log_to_file,
        assume_valid: params.assume_valid,
        #[cfg(feature = "zmq-server")]
        zmq_address: params.zmq_address,
        #[cfg(feature = "json-rpc")]
        json_rpc_address: params.rpc_address,
        generate_cert: params.generate_cert,
        wallet_descriptor: params.wallet_descriptor,
        filters_start_height: params.filters_start_height,
        user_agent: env!("USER_AGENT").to_owned(),
        assumeutreexo_value: None,
        electrum_address: params.electrum_address,
        enable_electrum_tls: params.enable_electrum_tls,
        electrum_address_tls: params.electrum_address_tls,
        tls_cert_path: params.tls_cert_path,
        tls_key_path: params.tls_key_path,
        allow_v1_fallback: params.allow_v1_fallback,
        backfill: !params.no_backfill,
    };

    #[cfg(unix)]
    if params.daemon {
        let mut daemon = Daemon::new(&config.datadir);
        if let Some(pid_file) = params.pid_file {
            daemon = daemon.pid_file(pid_file);
        }

        daemon.fork().expect("failed to daemonize");
    }

    let log_level = match config.debug {
        true => Level::DEBUG,
        false => Level::INFO,
    };

    // The guard must stay alive until the end of `main` to flush file logs when dropped.
    let _logger_guard = start_logger(
        &config.datadir,
        config.log_to_file,
        config.log_to_stdout,
        log_level,
    );

    let _rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .max_blocking_threads(2)
        .thread_keep_alive(Duration::from_secs(60))
        .thread_name("florestad")
        .build()
        .unwrap();

    let signal = Arc::new(RwLock::new(false));
    let _signal = signal.clone();

    _rt.spawn(async move {
        // This is used to signal the runtime to stop gracefully.
        // It will be set to true when we receive a Ctrl-C or a stop signal.
        tokio::signal::ctrl_c().await.unwrap();
        let mut sig = signal.write().await;
        *sig = true;
    });

    let florestad = Florestad::from(config);
    _rt.block_on(async {
        florestad.start().await.unwrap_or_else(|e| {
            eprintln!("Failed to start florestad: {e}");
            exit(1);
        });

        // wait for shutdown
        loop {
            if florestad.should_stop().await || *_signal.read().await {
                info!("Stopping Floresta");
                florestad.stop().await;
                let _ = timeout(Duration::from_secs(10), florestad.wait_shutdown()).await;
                break;
            }

            sleep(Duration::from_secs(5)).await;
        }
    });

    // Drop `florestad` and the runtime.
    // They are dropped outside the async block to avoid a nested
    // drop of the runtime due to the RPC server, which panics.
    drop(florestad);
    drop(_rt);
    // Flush logs to the file system when dropped.
    drop(_logger_guard);
}

fn resolve_data_paths(
    base_dir: Option<&Path>,
    network: Network,
    cli_signet_challenge: Option<ScriptBuf>,
    config_file: Option<PathBuf>,
) -> (PathBuf, Option<PathBuf>, Option<ScriptBuf>) {
    // Resolve the default config location before reading a file-based challenge.
    // A file that selects a custom signet remains at this discoverable path.
    let initial_datadir = datadir_path(base_dir, network, cli_signet_challenge.as_deref());
    let config_file_is_explicit = config_file.is_some();
    let config_file_path = config_file.unwrap_or_else(|| initial_datadir.join("config.toml"));
    let file_signet_challenge = if cli_signet_challenge.is_none() {
        read_signet_challenge(&config_file_path)
    } else {
        None
    };
    let signet_challenge = cli_signet_challenge.or(file_signet_challenge);
    let datadir = datadir_path(base_dir, network, signet_challenge.as_deref());
    let config_file = if config_file_is_explicit || datadir != initial_datadir {
        Some(config_file_path)
    } else {
        None
    };

    (datadir, config_file, signet_challenge)
}

/// Assemble the data directory [`PathBuf`] for the given [`Network`].
///
/// The data directory path is determined in this order:
/// - If `base_dir` is provided, it is used as-is.
/// - If `base_dir` is not provided and the `$HOME` environment
///   variable is set, `$HOME/.floresta` is used.
/// - If `base_dir` is not provided and the `$HOME` environment
///   variable is not set, the current directory is used.
///
/// For non-mainnet networks, the data directory is suffixed with the
/// network name. Custom signets use `signet-<message-start-bytes>`.
///
/// Paths with redundant slashes are automatically normalized.
fn datadir_path(
    base_dir: Option<impl AsRef<Path>>,
    network: Network,
    signet_challenge: Option<&Script>,
) -> PathBuf {
    let base_dir = base_dir
        .map(|p| {
            let s = p.as_ref().to_string_lossy().replace('\\', "/");
            Path::new(&s).components().collect::<PathBuf>()
        })
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".floresta")
        });

    match network {
        Network::Bitcoin => base_dir,
        Network::Signet => base_dir.join(signet_data_dir_name(signet_challenge)),
        Network::Testnet => base_dir.join("testnet3"),
        Network::Testnet4 => base_dir.join("testnet4"),
        Network::Regtest => base_dir.join("regtest"),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    /// Test that `datadir_path` returns the expected path for the given [`Network`].
    fn test_data_dir_path() {
        // (<input path>, <normalized path>)
        let paths: &[(&str, &str)] = &[
            ("path/to/dir", "path/to/dir"),
            ("path/to/dir/", "path/to/dir"),
            ("path///", "path"),
            ("path//to//dir//", "path/to/dir"),
            ("path\\to\\dir", "path/to/dir"),
        ];
        let networks: &[(Network, Option<&str>)] = &[
            (Network::Bitcoin, None),
            (Network::Signet, Some("signet")),
            (Network::Testnet, Some("testnet3")),
            (Network::Testnet4, Some("testnet4")),
            (Network::Regtest, Some("regtest")),
        ];

        for &(input_path, normalized_path) in paths {
            for &(network, suffix) in networks {
                let expected_path = match suffix {
                    Some(s) => PathBuf::from(normalized_path).join(s),
                    None => PathBuf::from(normalized_path),
                };
                assert_eq!(datadir_path(Some(input_path), network, None), expected_path);
            }
        }

        // Default path when none is provided
        let default_expected = dirs::home_dir()
            .unwrap_or(PathBuf::from("."))
            .join(".floresta");

        assert_eq!(
            datadir_path(None::<&str>, Network::Bitcoin, None),
            default_expected
        );

        assert_eq!(
            datadir_path(
                Some("path/to/dir"),
                Network::Signet,
                Some(Script::from_bytes(&[0x51]))
            ),
            PathBuf::from("path/to/dir/signet-54d26fbd")
        );
    }

    #[test]
    fn file_configured_signet_uses_challenge_specific_directory() {
        let tempdir = tempdir().expect("temporary directory");
        let default_signet_dir = tempdir.path().join("signet");
        fs::create_dir(&default_signet_dir).expect("default signet directory");
        let config_path = default_signet_dir.join("config.toml");
        fs::write(&config_path, "signet_challenge = \"51\"").expect("signet configuration");

        let (datadir, config_file, signet_challenge) =
            resolve_data_paths(Some(tempdir.path()), Network::Signet, None, None);

        assert_eq!(datadir, tempdir.path().join("signet-54d26fbd"));
        assert_eq!(config_file, Some(config_path));
        assert_eq!(signet_challenge, Some(ScriptBuf::from_bytes(vec![0x51])));
    }
}

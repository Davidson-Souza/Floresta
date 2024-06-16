// Written in 2022 by Davidson Souza.
// SPDX-License-Identifier: CC0-1.0

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
mod config_file;
mod error;
mod florestad;
#[cfg(feature = "json-rpc")]
mod json_rpc;
mod slip132;
mod wallet_input;
#[cfg(feature = "zmq-server")]
mod zmq;

use std::time::Duration;

use clap::Parser;
use cli::Cli;
use cli::Commands;
use florestad::Config;
use florestad::Florestad;
#[cfg(feature = "zmq-server")]
use zmq::ZMQServer;

#[async_std::main]
async fn main() {
    let params = Cli::parse();

    let config = match params.command {
        #[cfg(feature = "experimental-p2p")]
        Some(Commands::Run {
            data_dir,
            assume_valid,
            wallet_xpub,
            wallet_descriptor,
            rescan,
            proxy,
            zmq_address: _zmq_address,
            cfilters,
            connect,
            rpc_address,
            electrum_address,
        }) => Config {
            data_dir,
            assume_valid,
            wallet_xpub,
            wallet_descriptor,
            rescan,
            proxy,
            config_file: params.config_file,
            network: params.network,
            cfilters,
            #[cfg(feature = "zmq-server")]
            zmq_address: _zmq_address,
            connect,
            #[cfg(feature = "json-rpc")]
            json_rpc_address: rpc_address,
            electrum_address,
            log_to_file: true,
            log_to_stdout: true,
        },

        // We may have more commands here, like setup and dump wallet
        None => Config {
            config_file: params.config_file,
            network: params.network,
            cfilters: true,
            log_to_file: false,
            log_to_stdout: true,
            ..Default::default()
        },
    };

    let florestad = Florestad::from(config);
    florestad.start();

    let stop_signal = florestad.get_stop_signal();
    let _stop_signal = stop_signal.clone();

    ctrlc::set_handler(move || {
        async_std::task::block_on(async {
            *(stop_signal.write().await) = true;
        })
    })
    .expect("Could not setup ctr+c handler");

    loop {
        if *_stop_signal.read().await {
            florestad.wait_shutdown().await;
            break;
        }
        async_std::task::sleep(Duration::from_secs(5)).await;
    }
}

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
#![deny(unused)]

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

use std::sync::mpsc;

use clap::Parser;
use cli::Cli;
use cli::Commands;
use cli::FilterType;
use log::info;

use florestad::Config;
use florestad::Florestad;

fn main() {
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
            cfilter_types,
            connect,
            rpc_address,
            electrum_address,
        }) => {
            // By default, we build filters for WPKH and TR outputs, as they are the newest.
            // We also build the `inputs` filters to find spends
            let cfilter_types = match cfilter_types {
                Some(cfilters) if !cfilters.is_empty() => cfilters,
                _ => {
                    vec![FilterType::SpkWPKH, FilterType::SpkTR, FilterType::Inputs]
                }
            };

            Config {
                data_dir,
                assume_valid,
                wallet_xpub,
                wallet_descriptor,
                rescan,
                proxy,
                config_file: params.config_file,
                network: params.network,
                cfilters,
                cfilter_types,
                #[cfg(feature = "zmq-server")]
                zmq_address: _zmq_address,
                connect,
                #[cfg(feature = "json-rpc")]
                json_rpc_address: rpc_address,
                electrum_address,
                log_to_stdout: true,
            }
        }

        // We may have more commands here, like setup and dump wallet
        None => {
            let cfilter_types = vec![FilterType::SpkWPKH, FilterType::SpkTR, FilterType::Inputs];

            Config {
                config_file: params.config_file,
                network: params.network,
                cfilters: true,
                cfilter_types,
                ..Default::default()
            }
        }
    };

    let mut daemon = Florestad::from(config);
    daemon.start();

    // this channel will be used to signal we should stop
    let (stop_sender, stop_reader) = mpsc::channel();

    // setup a SIGNINT handler
    ctrlc::set_handler(move || {
        info!("got a request to shutdown");
        stop_sender.send(()).expect("main loop died?");
    })
    .expect("Error setting Ctrl-C handler");

    // this channel will only have something if we get a SIGINT or similar
    stop_reader.recv().map(|_| daemon.stop()).expect("this channel is valid");
}

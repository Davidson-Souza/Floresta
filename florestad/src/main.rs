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

use std::time::Duration;

use clap::Parser;
use cli::Cli;
use florestad::Config;
use florestad::Florestad;
use futures::executor::block_on;
use tokio::time::sleep;

#[tokio::main]
async fn main() {
    let params = Cli::parse();

    let config = Config {
        network: params.network.into(),
        debug: params.debug,
        data_dir: params.data_dir,
        cfilters: params.cfilters,
        proxy: params.proxy,
        rescan: params.rescan,
        assume_utreexo: params.assume_utreexo,
        connect: params.connect,
        wallet_xpub: params.wallet_xpub,
        config_file: params.config_file,
        log_to_file: params.log_file,
        assume_valid: params.assume_valid,
        log_to_stdout: true,
        json_rpc_address: params.rpc_address,
        electrum_address: params.electrum_address,
        wallet_descriptor: params.wallet_descriptor,
        filters_start_height: params.filters_start_height,
    };

    let florestad = Florestad::from(config);
    florestad.start();

    let stop_signal = florestad.get_stop_signal();
    let _stop_signal = stop_signal.clone();

    ctrlc::set_handler(move || {
        block_on(async {
            *(stop_signal.write().await) = true;
        })
    })
    .expect("Could not setup ctr+c handler");

    loop {
        if *_stop_signal.read().await {
            florestad.wait_shutdown().await;
            break;
        }
        sleep(Duration::from_secs(5)).await;
    }
}

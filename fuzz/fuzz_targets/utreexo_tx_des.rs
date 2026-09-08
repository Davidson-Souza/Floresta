// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use bitcoin::consensus::deserialize;
use floresta_wire::tx_relay::UtreexoTx;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = deserialize::<UtreexoTx>(data);
});

// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use std::sync::LazyLock;

use floresta_mempool::mempool::Mempool;
use floresta_wire::fuzz::v2_peer;
use libfuzzer_sys::fuzz_target;
use tokio::runtime::Runtime;

static RUNTIME: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("fuzz runtime should start"));

fuzz_target!(|data: &[u8]| {
    RUNTIME.block_on(v2_peer(data.to_vec(), Mempool::new(0)));
});

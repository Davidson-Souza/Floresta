// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use std::sync::LazyLock;

use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256d;
use floresta_mempool::mempool::Mempool;
use floresta_wire::fuzz::v1_peer;
use libfuzzer_sys::fuzz_mutator;
use libfuzzer_sys::fuzz_target;
use tokio::runtime::Runtime;

const V1_HEADER_SIZE: usize = 24;
const V1_MESSAGE_PREFIX_SIZE: usize = 1;
const MIN_V1_MESSAGE_SIZE: usize = V1_MESSAGE_PREFIX_SIZE + V1_HEADER_SIZE;
const INVALID_CHECKSUM_SEED_DIVISOR: u32 = 32;

static RUNTIME: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("fuzz runtime should start"));

fuzz_target!(|data: &[u8]| {
    RUNTIME.block_on(v1_peer(data.to_vec(), Mempool::new(0)));
});

fuzz_mutator!(|data: &mut [u8], size: usize, max_size: usize, seed: u32| {
    mutate_v1_message(data, size, max_size, seed)
});

fn mutate_v1_message(data: &mut [u8], size: usize, max_size: usize, seed: u32) -> usize {
    let max_size = max_size.min(data.len());
    let size = libfuzzer_sys::fuzzer_mutate(data, size.min(data.len()), max_size).min(data.len());
    if size < MIN_V1_MESSAGE_SIZE {
        return size;
    }
    // Keep the checksum valid most of the time, but occasionally
    // make it invalid so checksum rejection remains reachable.
    if seed % INVALID_CHECKSUM_SEED_DIVISOR != 0 {
        let message = &mut data[V1_MESSAGE_PREFIX_SIZE..size];
        let checksum = sha256d::Hash::hash(&message[V1_HEADER_SIZE..]).to_byte_array();
        message[20..24].copy_from_slice(&checksum[..4]);
    }
    size
}

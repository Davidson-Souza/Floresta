// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use std::sync::LazyLock;

use bitcoin::Network;
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
    let mutated_size =
        libfuzzer_sys::fuzzer_mutate(data, size.min(data.len()), max_size).min(data.len());
    if max_size < MIN_V1_MESSAGE_SIZE {
        return mutated_size;
    }

    let size = mutated_size.max(MIN_V1_MESSAGE_SIZE);
    data[mutated_size..size].fill(0);
    let message = &mut data[V1_MESSAGE_PREFIX_SIZE..size];
    let command = command_for(message.get(V1_HEADER_SIZE).copied().unwrap_or_default());

    message[..4].copy_from_slice(&Network::Regtest.magic().to_bytes());
    message[4..16].fill(0);
    message[4..4 + command.len()].copy_from_slice(command);
    let payload_len = u32::try_from(message.len() - V1_HEADER_SIZE)
        .expect("libFuzzer input length fits in a P2P message");
    message[16..20].copy_from_slice(&payload_len.to_le_bytes());

    let checksum = sha256d::Hash::hash(&message[V1_HEADER_SIZE..]).to_byte_array();
    message[20..24].copy_from_slice(&checksum[..4]);

    // LibFuzzer supplies the seed, keeping checksum rejection reachable in one out of 32 cases.
    if seed % INVALID_CHECKSUM_SEED_DIVISOR == 0 {
        message[20] ^= 1;
    }

    size
}

fn command_for(selector: u8) -> &'static [u8] {
    const COMMANDS: [&[u8]; 5] = [b"verack", b"getaddr", b"ping", b"addr", b"fuzz"];

    COMMANDS[usize::from(selector) % COMMANDS.len()]
}

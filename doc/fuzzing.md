# Fuzzing

This project uses `cargo-fuzz` (libfuzzer) for fuzzing, you can run a fuzz target with:

```bash
CARGO_PROFILE_RELEASE_LTO=false cargo +nightly fuzz run local_address_str
```

You can replace `local_address_str` with the name of any other target you want to run.

Available fuzz targets:

- `addrman`
- `best_chain_decode`
- `best_chain_roundtrip`
- `bitcoin_socket_addr`
- `disk_block_header_decode`
- `disk_block_header_roundtrip`
- `flat_chainstore_header_insertion`
- `local_address_str`
- `merkle_root`
- `onion_address_rtt`
- `onion_address_str_parse`
- `p2p_v1_peer`
- `p2p_v2_message`
- `p2p_v2_peer`
- `utreexo_proof_des`

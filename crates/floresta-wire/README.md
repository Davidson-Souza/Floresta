<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# floresta-wire

`floresta-wire` is the P2P layer for the Floresta project: a lightweight, Utreexo-powered Bitcoin node. It supports Bitcoin P2P transport v1 and v2/BIP324, with peer discovery via DNS seeds, hardcoded addresses, and user-provided peers. It learns about new blocks, selects the best header chain, performs IBD, announces locally-submitted transactions to peers, and stays in sync with the network.

It also supports compact block filters, which are particularly helpful for rescans because Floresta is fully pruned, and an optional SOCKS5 proxy to route all P2P traffic and DNS seed queries.

This crate delegates chain state, validation, and persistence to [`floresta-chain`](https://github.com/getfloresta/Floresta/tree/master/crates/floresta-chain) and is used by the `florestad` binary.

## Usage

Construct a node with `UtreexoNode`:

```rust
let node = UtreexoNode::<_, RunningNode>::new(
    /* UtreexoNodeConfig */,
    /* ChainState */,
    /* Mempool */,
    /* Optional compact block filters */,
    /* Kill signal */,
    /* AddressMan */,
).unwrap();
```

Where:

- `UtreexoNodeConfig` and `AddressMan` are defined in this crate.
- The mempool can be any implementation of `MempoolBase` from `floresta-domain` — for example `Mempool` from `floresta-mempool`.
- `ChainState` is provided by `floresta-chain` (implements the `ChainBackend` trait).
- Optional compact filters use `NetworkFilters` from `floresta-compact-filters`.
- The kill signal is an `Arc<tokio::sync::RwLock<bool>>`. Set it to `true` to stop the node.

`UtreexoNode` can run in three contexts: `RunningNode` (default, high-level), `SyncNode` (IBD), and `ChainSelector` (header chain selection). In most cases you want the default `RunningNode` context, which delegates to `SyncNode`/`ChainSelector` as needed based on the current phase.

> To start the node, regardless of context, call the `run` method. To stop it, set the kill signal you passed to the builder to `true`.

## Peer sources

Just after the node startup, it will attempt to discover and/or load peers in the following order:

- `peers.json`: a file in the node's data directory that can contain user-provided peers. If not present, this file will be created after the first run. Its purpose is to **store all known peers** and their metadata.
- Configured seed nodes: temporary feeler connections that request peer addresses and disconnect after receiving them.
- DNS seeds: active peers fetched via the system DNS resolver, or (if a proxy is configured) via DNS-over-HTTPS routed through SOCKS5. We usually get a few hundred peers with this method. DNS seeds can be disabled with `UtreexoNodeConfig.disable_dns_seeds`.
- Hardcoded addresses (from the `seeds` directory): used only if the node is not connected to any peers within one minute of startup. Acts as a fallback when the previous two methods are unused or fail.

Custom signets never use the default Signet DNS seeds or hardcoded addresses.
Configure seed nodes or fixed peers to bootstrap a custom network.

Once the node has at least one peer, it will be able to discover additional peers through the P2P network's address gossip (all of them saved in `peers.json`).

Example `peers.json`:

```json
[
  {
    "address": {
      "V4": "1.228.21.110"
    },
    "last_connected": 1678986166,
    "state": {
      "Tried": 0
    },
    "services": 50331657,
    "port": 8333
  }
]
```

## Cargo features

- `metrics`: when enabled, `floresta-wire` links in a small Prometheus exporter so your node can expose runtime metrics over HTTP (start with `floresta_metrics::metrics_server(addr)`).

## Minimum Supported Rust Version (MSRV)

This library should compile with any feature combination on **Rust 1.85.0**.

## License

`floresta-wire` is released under the terms of the MIT license. See [LICENSE](https://github.com/getfloresta/Floresta/blob/master/LICENSE) for more information or see https://opensource.org/licenses/MIT.

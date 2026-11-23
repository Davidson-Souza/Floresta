# `getnetworkinfo`

Returns information about the node's P2P networking state, including version, advertised services, connection counts, and per-network reachability.

## Usage

### Synopsis

```bash
floresta-cli getnetworkinfo
```

### Examples

```bash
# Get networking information for the running node
floresta-cli getnetworkinfo
```

## Arguments

This command takes no arguments.

## Returns

### Ok Response

Returns a JSON object with the following fields:

- `version` - (numeric) The server version, encoded as Bitcoin Core's `MMmmpp` numeric scheme (e.g., `1.2.5` is encoded as `10205`).

- `subversion` - (string) The User Agent string the node advertises to peers (e.g., `/Floresta:0.10.0/`).

- `protocolversion` - (numeric) The P2P protocol version the node speaks.

- `localservices` - (string) The services the node advertises to the network, as a hex-encoded 64-bit bitfield.

- `localservicesnames` - (array of strings) Human-readable names of the advertised services (e.g., `NETWORK`, `WITNESS`, `UTREEXO`).

- `localrelay` - (boolean) Whether the node requests transaction relay from peers. Always `false` in Floresta, since Floresta's mempool only holds locally-submitted transactions.

- `timeoffset` - (numeric) The time offset, in seconds. Always `0` in Floresta, since Floresta does not track peer time offsets.

- `connections` - (numeric) The total number of peer connections (inbound + outbound).

- `connections_in` - (numeric) The number of inbound connections. Always `0` in Floresta, since the node does not accept inbound connections.

- `connections_out` - (numeric) The number of outbound connections.

- `networkactive` - (boolean) Whether P2P networking is enabled. Always `true` in Floresta, since networking cannot be toggled at runtime.

- `networks` - (array of objects) Information about each network the node knows of. Each entry contains:
    * `name` - (string) The network name (e.g., `ipv4`, `ipv6`, `onion`, `i2p`, `cjdns`).
    * `limited` - (boolean) Whether the network is unreachable from this node.
    * `reachable` - (boolean) Whether the network is reachable from this node.
    * `proxy` - (string) The proxy used for this network in `host:port` form, or empty if none.
    * `proxy_randomize_credentials` - (boolean) Whether randomized credentials are used for the proxy.

- `relayfee` - (numeric) The minimum relay fee rate, in BTC/kB. Always `0.0` in Floresta, since Floresta's mempool only holds locally-submitted transactions.

- `incrementalfee` - (numeric) The minimum fee rate increment for mempool limiting or replacement, in BTC/kB. Always `0.0` in Floresta, since Floresta's mempool only holds locally-submitted transactions.

- `localaddresses` - (array of objects) Local addresses the node is listening on. Always empty in Floresta, since the node does not accept inbound connections.

- `warnings` - (array of strings) Any network or blockchain warnings. Always empty in Floresta, since Floresta does not emit network or blockchain warnings.

### Error Enum

* `JsonRpcError::Node` - If there is an internal node error preventing retrieval of the connection count (e.g., "Failed to get connection count").

## Notes

- This RPC mirrors Bitcoin Core's `getnetworkinfo` and reuses Bitcoin Core's response schema for compatibility. Several fields are hardcoded because Floresta is a lightweight, outbound-only node: it does not accept inbound connections (`connections_in`, `localaddresses`), has a mempool that only holds locally-submitted transactions with no inbound relay or fee policy (`localrelay`, `relayfee`, `incrementalfee`), does not track peer time offsets (`timeoffset`), cannot toggle networking at runtime (`networkactive`), and does not emit network or blockchain warnings (`warnings`).
- For per-peer details rather than node-wide networking state, see [`getpeerinfo`](getpeerinfo.md). For just the connection count, see [`getconnectioncount`](getconnectioncount.md).

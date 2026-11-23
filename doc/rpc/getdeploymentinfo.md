# `getdeploymentinfo`

Returns information about the deployment of consensus changes (softforks) at a given block.

## Usage

### Synopsis

```bash
floresta-cli getdeploymentinfo [blockhash]
```

### Examples

```bash
# Query deployment info at the current chain tip
floresta-cli getdeploymentinfo

# Query deployment info at a specific block
floresta-cli getdeploymentinfo "0000000000000000000123abc..."
```

## Arguments

- `blockhash` - (string, optional) Block hash to query. Defaults to the current chain tip.

## Returns

### Ok Response

Returns a JSON object describing the activation state of every known consensus deployment at the queried block.

- `hash` - (string) Hash of the block at which deployment state was queried.
- `height` - (numeric) Height of that block.
- `script_flags` : (json array) script verify flags for the block
  - `"str"` - (string) a script verify flag
- `deployments` - (object) Map of deployment name to deployment state. Each entry contains:
  - `type` - (string) `"buried"` for deployments locked-in at fixed heights, `"bip9"` for versionbits deployments.
  - `height` - (numeric) The activation height of the deployment.
  - `active` - (boolean) Whether the deployment is active at the queried block.
  - `bip9` - (object, nullable) BIP9 state information when `type` is `"bip9"`; currently always `null` since Floresta only emits buried deployments.

### Error Enum

* `JsonRpcError::BlockNotFound` - The requested block hash was not found in the blockchain
* `JsonRpcError::Chain` - If there's an error accessing blockchain data

## Notes

- If `blockhash` is omitted, the current chain tip is used.
- Floresta currently reports the five buried deployments tracked by Bitcoin Core: `bip34`, `bip66`, `bip65`, `csv`, and `segwit`. BIP9 deployments (`taproot`, `testdummy`) require the versionbits state machine and are not yet emitted.
- Supported networks: mainnet, testnet, testnet4, signet, and regtest.

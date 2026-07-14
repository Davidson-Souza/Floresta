# `getblock`

Retrieve information about a specific block by its hash. The verbosity parameter determines the format of the returned data.

## Usage

### Synopsis

```bash
floresta-cli getblock <blockhash> [verbosity]
```

### Examples

```bash
# Returns a JSON object with detailed block information (default verbosity = 1)
floresta-cli getblock "000000000000000000007ae6247b184396b8a1a292b8435508f448669ead45a6"

# Returns a serialized, hex-encoded string of the block data (verbosity = 0)
floresta-cli getblock "000000000000000000007ae6247b184396b8a1a292b8435508f448669ead45a6" 0

# Returns a JSON object with detailed block information (verbosity = 1)
floresta-cli getblock "000000000000000000007ae6247b184396b8a1a292b8435508f448669ead45a6" 1
```

## Arguments

- `blockhash` - (string, required) The block hash.
- `verbosity` - (numeric, optional, default=1)
  - `0`: Returns a serialized, hex-encoded string of the block data.
  - `1`: Returns a JSON object with detailed block information.

## Returns

### Ok Response (for verbosity = 0)

- `"hex"` - (string) A serialized, hex-encoded string of the block data.

### Ok Response (for verbosity = 1)

Return Json object

Return JSON object
- `confirmations` - (numeric) The number of confirmations.
- `size` - (numeric) The block size.
- `strippedsize` - (numeric) The block size excluding witness data.
- `weight` - (numeric) The block weight as defined in BIP 141.
- `height` - (numeric) The block height or index.
- `version` - (numeric) The block version.
- `versionHex` - (string) The block version formatted in hexadecimal.
- `merkleroot` - (string) The merkle root.
- `coinbase_tx` - (json object) Coinbase transaction metadata
   - `version` -  (numeric) The coinbase transaction version
   - `locktime` - (numeric) The coinbase transaction's locktime (nLockTime)
   - `sequence` - (numeric) The coinbase input's sequence number (nSequence)
   - `coinbase` - (string) The coinbase input's script
   - `witness` - (string, optional) The coinbase input's first (and only) witness stack element, if present
- `tx` - (json array) Transaction IDs:
   - `"hex"` - (string) A transaction id (hex-encoded).
- `time` - (numeric) The block time expressed in UNIX epoch time.
- `mediantime` - (numeric) The median block time expressed in UNIX epoch time.
- `nonce` - (numeric) The nonce.
- `bits` - (string) Compact representation of the block difficulty target.
- `target` - (string) The difficulty target.
- `difficulty` - (numeric) The difficulty.
- `chainwork` - (string) Expected number of hashes required to produce the chain up to this block (in hex).
- `nTx` - (numeric) The number of transactions in the block.
- `previousblockhash` - (string, optional) The hash of the previous block.
- `nextblockhash` - (string, optional) The hash of the next block.

### Error Enum `CommandError`

* `JsonRpcError::Chain`
* `JsonRpcError::BlockNotFound`
* `JsonRpcError::ChainWorkOverflow`

## Notes

- Floresta doesnt store blocks so this command will delegate the request to the bitcoin network.
- To retrieve block hashes, you can use the `getblockhash` RPC to obtain the hash of a specific block by its height, or the `getbestblockhash` RPC to get the hash of the latest known block. These hashes can then be used with the `getblock` RPC to retrieve detailed block information.
- **In regtest**, the difficulty value may not match real-world conditions due to easier mining.
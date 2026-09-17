# Running Floresta

After building, florestad and floresta-cli will be available in the target directory. You can run the full node with
```bash
./target/release/florestad
# or, if you installed it with cargo install
florestad
```

You may run it as a background process with the `--daemon` flag.

```bash
florestad --daemon
```

This will start the full node, and you can connect to it with an Electrum wallet or with the `floresta-cli` tool.

```bash
floresta-cli getblockchaininfo
```

For more information on how to use the `floresta-cli` tool, you can check the [API documentation](https://github.com/getfloresta/Floresta/blob/master/bin/floresta-cli/README.md).

## TLS

By default, `florestad` will run an Electrum server without encryption, but you can add TLS encryption to Electrum communication:

Manually create the TLS private key and certificate before running. These must be PKCS#8-encoded:

```bash
openssl genpkey -algorithm RSA -out key.pem -pkeyopt rsa_keygen_bits:2048

openssl req -x509 -new -key key.pem -out cert.pem -days 365 -subj "/CN=localhost"

florestad --enable-electrum-tls --tls-key-path=<privkey_path> --tls-cert-path=<cert_path>
```

Or have Floresta generate them for you:

```bash
florestad --enable-electrum-tls --generate-cert
```

> Be aware that self-signed certificates do not inherently protect against
man-in-the-middle (MITM) attacks because they
[lack validation from a trusted Certificate Authority (CA)](https://security.stackexchange.com/questions/264247/man-in-the-middle-attack-only-affects-tls-certs-with-unqualified-subject-names).

## Assume Utreexo

If you want to start your node and get up and running quickly, you can use the Assume Utreexo feature. This is enabled by default, but you can disable it with the `--no-assume-utreexo` flag.

```bash
florestad --no-assume-utreexo
```

## Backfill

After the node starts it will validate everything in the background, downloading blocks from genesis to the assumed height, validating them and compare with the provided value. This way, you can start using the node right away, but still validate everything. This option, however, will take some time to complete, using a lot of CPU and bandwidth.

This is the default behavior of the `florestad` if no flags are provided. You can disable it using:

```bash
florestad --no-backfill
```

## Compact Filters

Floresta synchronizes and persists the compact-filter header chain, then fetches and validates full filters on demand for wallet rescans. Only a bounded cache of recent full filters is retained, so enabling rescans does not require storing every filter.

To disable compact-filter synchronization and historical rescans, start the node with the `--no-cfilters` flag.

```bash
florestad --no-cfilters
```

## Getting Help

You can get a list of all the available commands by running

```bash
floresta-cli help
```

and you can get the cli parameters by running
```bash
floresta-cli help <command>
```

## Wallet

Floresta comes with a watch-only wallet that you can use to track your transactions. You just need to provide the wallet
information, either as a configuration file or as a command line argument. See the [sample configuration file](../contrib/config.toml.sample) for an example config. Floresta supports SLIP-132 extended public keys (xpubs) and output descriptors. You can add new wallets to follow at any time, just
call the `rescanblockchain` rpc after adding the wallet.

You can add new descriptors to the wallet with the `loaddescriptor` rpc.

```bash
floresta-cli loaddescriptor "wpkh(xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/<0;1>/*)"
```

Historical rescans require compact filters to be enabled. Filter headers are synchronized automatically, and full filters are fetched and validated on demand. To scan from block 800,000 to the current tip, use the `rescanblockchain` RPC:

```bash
floresta-cli rescanblockchain 800000
```

Once you have a transaction cached in your watch-only, you can use either the rpc or integrated electrum server to retrieve information about your wallet. You can use wallets like Electrum or Sparrow to connect to your node and retrieve information about your wallet. Just connect with the server running at `127.0.0.1:50001:t`. On electrum you may want to use the `--oneserver` flag to connect to a single server, for better privacy.

## Using `tokio console`

[tokio-console](https://github.com/tokio-rs/console) is an utility from the `tokio` project that let's you inspect all running tasks, with a tool reminiscent of `htop`. You can navigate through each task, see how often it runs, for how long, and which resources are being used. To use it, you must compile `floresta` with the following configuration

```bash
RUSTFLAGS=--cfg tokio_unstable cargo build --release --bin florestad --features tokio-console
```

Then you just need to install and run `tokio-console`, you can use `cargo` for that

```bash
cargo install tokio-console
tokio-console
```

You should see something like this:

![Screenshot_2025-06-23_20-35-03](https://github.com/user-attachments/assets/f89a5898-a531-4aa3-ade4-5feb27c5197b)

The main screen shows all the tasks that were spawned, how many time it woken up, how long did it run, wait or blocked.

![Screenshot_2025-06-23_20-35-25](https://github.com/user-attachments/assets/4697ca9c-5283-4d89-a129-51ff5e2e9b45)

This is the resource tab, it shows all resources that `tokio` is menaging (locks, files, sockets, channels...). You can toggle this view by pressing `r`, and go back to tasks by pressing `t`.

![Screenshot_2025-06-23_20-35-44](https://github.com/user-attachments/assets/e05bf099-87a7-422b-ae3e-d1326941a688)

This shows some info about one specific task, like a histogram of CPU usage (you can toggle this by pressing ENTER while on the main screen)

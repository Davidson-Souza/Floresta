# JSON-RPC authentication

This document covers how florestad authenticates JSON-RPC requests and the
three configurations supported: cookie auth (default), single-user
`-rpcuser`/`-rpcpassword`, and multi-user `-rpcauth`.

## Quick start

Cookie auth is on by default. If you run `floresta-cli` on the same machine
as `florestad` and as the same Unix user, no setup is required:

```bash
$ florestad
# In another terminal:
$ floresta-cli getblockcount
0
```

The daemon writes a randomly-generated credential to a `.cookie` file in the
data directory and `floresta-cli` reads it. The file rotates every restart.

## Choosing an auth mode

- **Cookie auth (default):** start `florestad` with no auth flags. A
  random credential is written to a `.cookie` file in the data directory
  and rotated on every restart.
- **Username/password auth:** start with
  `florestad --rpc-user=alice --rpc-password=hunter2` for a single
  configured operator credential. Setting `--rpc-password` disables
  cookie generation. `--rpc-user` is optional: without it the username is
  empty, and clients send `:<password>` as `bitcoin-cli` does. An empty
  `--rpc-password` counts as unset on both the daemon and the CLI, so
  `-P "$UNSET_VAR"` falls back to cookie auth instead of failing.
- **Multi-user auth:** start with one or more
  `florestad --rpc-auth='alice:<salt>$<hash>'` entries for pre-hashed
  credentials. Cookie auth stays active alongside unless explicitly
  disabled.

These modes can be combined: `-rpcauth` entries are loaded on top of
whichever of cookie or password auth is active first. Cookie auth and
password auth are mutually exclusive (setting `--rpc-password` disables
cookie generation).

## Cookie file

**Default location:** under the network-specific data directory, named
`.cookie`. With the default `$HOME/.floresta` datadir:

- mainnet: `~/.floresta/.cookie`
- signet: `~/.floresta/signet/.cookie`
- testnet3: `~/.floresta/testnet3/.cookie`
- testnet4: `~/.floresta/testnet4/.cookie`
- regtest: `~/.floresta/regtest/.cookie`

Mainnet sits at the datadir root; every other network nests one level
deeper, matching Bitcoin Core's `<datadir>/[<net>/].cookie` layout.

**File format:** a single line with no trailing newline, of the form
`__cookie__:<64-char-hex-token>`. The whole line is the basic-auth
`user:pass` payload.

**File permissions:** 0600 on Unix (owner read/write only). Any process
running as the same Unix user can read it; any other user gets
`Permission denied`.

**Rotation:** every `florestad` restart rolls a fresh token. Any prior
cookie file at the path is replaced.

**Lifetime:** the cookie file belongs to the daemon. Whatever sits at the
configured path is replaced on startup and removed on shutdown, so point
`--rpc-cookie-file` at a path you are willing to hand over. Replacing an
existing file at an operator-chosen path is logged as a warning. A failed
write is never followed by a delete: if the daemon could not create the
cookie, it removes nothing on the way out. This matches Bitcoin Core, which
writes `-rpccookiefile` the same way and deletes it on shutdown when the
running process created it.

**Overriding the path:** use `--rpc-cookie-file=<path>`. Absolute paths are
used as-is; relative paths are joined against the network-specific data
directory.

```bash
$ florestad --rpc-cookie-file="$HOME/.floresta/custom.cookie"
$ floresta-cli --rpc-cookie-file="$HOME/.floresta/custom.cookie" getblockcount
```

**Disabling cookie auth entirely:** use `--no-rpc-cookie-file`. You must
configure either `-rpcuser`/`-rpcpassword` or one or more `-rpcauth`
entries alongside it, or the daemon refuses to start.

```bash
$ florestad --no-rpc-cookie-file --rpc-auth='alice:<salt>$<hash>'
```

## Multi-user auth via -rpcauth

For provisioning multiple users without sharing plaintext passwords with
the daemon. Each entry is a pre-hashed line generated with Bitcoin Core's
reference script.

**Generating an entry:**

```bash
$ curl -sO https://raw.githubusercontent.com/bitcoin/bitcoin/master/share/rpcauth/rpcauth.py
$ python3 rpcauth.py alice
String to be appended to bitcoin.conf:
rpcauth=alice:cf2c6b493e4690de904306d1a82ef1cc$9e0770d953ba59dcd5cf447615717764936d48b01d4428d251628f0156088901
Your password:
Ys-WMXs6znL6q2BH9yjh77k9keytl4JpCshiapNCMyg
```

**Passing it to florestad:**

```bash
$ florestad --rpc-auth='alice:cf2c6b493e4690de904306d1a82ef1cc$9e0770d953ba59dcd5cf447615717764936d48b01d4428d251628f0156088901'
$ floresta-cli --rpc-user=alice --rpc-password='Ys-WMXs6znL6q2BH9yjh77k9keytl4JpCshiapNCMyg' getblockcount
```

Save the password output from `rpcauth.py` and give it to the user; only
the hashed entry lives on the daemon side.

**Multiple users:** repeat `--rpc-auth` for each entry.

```bash
$ florestad --rpc-auth='alice:s1$h1' --rpc-auth='bob:s2$h2'
```

## Username/password auth

For a single configured operator account:

```bash
$ florestad --rpc-user=alice --rpc-password=hunter2
$ floresta-cli --rpc-user=alice --rpc-password=hunter2 getblockcount
```

When `--rpc-password` is set, no cookie file is written (mutual exclusion
with cookie auth). The password is hashed with a fresh random salt at
startup, and requests are checked against `(user, salt_hex, hash_hex)`.
The plaintext is not wiped: it stays in the daemon's config for the life
of the process, as it does in the process arguments or the config file it
came from.

## TOML config file

All four `[rpc]` keys can be set in the floresta config file, read from
`<datadir>/config.toml` (so `~/.floresta/config.toml` on mainnet, and
`~/.floresta/signet/config.toml` on signet) unless `--config-file` points
elsewhere:

```toml
[rpc]
user = "alice"
password = "hunter2"
auth = [
    "bob:<salt>$<hash>",
    "carol:<salt>$<hash>",
]
cookie_file = "/home/alice/.floresta/custom.cookie"
```

CLI flags override config-file values for `user`, `password`, and
`cookie_file`. The `auth` vector merges additively with `--rpc-auth` CLI
entries.

## What clients send

HTTP basic auth header on every request:

```text
Authorization: Basic <base64(user:pass)>
```

A `curl` example using the cookie file directly:

```bash
$ curl --user "$(cat ~/.floresta/signet/.cookie)" \
       --data '{"jsonrpc":"2.0","method":"getblockcount","params":[],"id":1}' \
       -H 'content-type: application/json' \
       http://127.0.0.1:38332/
```

`curl --user` does the base64 encoding and sets the header. The same
construction works for any HTTP client.

## Error responses

**On missing, malformed, or wrong credentials:**

```text
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Basic realm="jsonrpc"
```

The body is empty.

**On any present-but-invalid `Authorization` header** (wrong
credentials, malformed base64, missing `:`, wrong scheme, or non-ASCII
bytes), the daemon sleeps 250 ms before the 401 lands. A request with
no `Authorization` header is rejected immediately, without the delay.

The wrong-credentials case additionally logs a warn line tagged with the
peer address; the other malformed cases are traced at debug level only:

```text
WARN incorrect password attempt from 192.0.2.1:54321
```

A request takes one of 16 auth slots *before* its credentials are
checked, and the slot stays held for the delay on failure even if the
client hangs up, capping guesses at roughly 64/s. Both halves matter:
taking the slot after the check would only cap how fast 401s go out, and
releasing it on hang-up would free it the moment a guess was checked.
Either way an attacker could ignore the 401s, since a correct guess
returns 200 without waiting. Core caps the same rate by blocking one of
its 16 HTTP workers, which a hang-up cannot free either. Past 64 queued
requests the daemon returns 503 instead of growing the backlog, as Core
sheds when `-rpcworkqueue` fills.

Two consequences before exposing the port: the cap is per-process, not
per-IP, so it slows guessing rather than preventing it; and the slots are
shared, so during a flood your own authenticated calls can be queued or
get a 503. Core accepts the same trade.

## Common configuration errors

- `Invalid -rpcauth argument.`: the entry doesn't match
  `<user>:<salt_hex>$<hash_hex>` exactly, contains empty `user`, `salt`,
  or `hash` fields, or has uppercase characters in the hash. The salt's
  case is not checked, since it is used verbatim as the HMAC key.
  Re-generate with `rpcauth.py`.

- `--rpc-cookie-file=<path> conflicts with --no-rpc-cookie-file; pick one`:
  both flags were set on the command line. Drop whichever you didn't
  mean.

- `no authentication method configured: cookie auth disabled, no
  -rpcuser/-rpcpassword or -rpcauth entries`: the daemon refuses to
  start because nobody could authenticate. Add at least one credential
  configuration.

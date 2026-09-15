# Reproducible cross-platform releases

Floresta release archives are cross-built on one `x86_64-linux` machine with the
Nix expressions pinned by `flake.lock`. The workflow separates three operations:

1. independent builders produce bit-for-bit identical unsigned archives;
2. builders GPG-sign identical SHA-256 manifests in an attestation repository;
3. designated code signers create detached Windows and macOS signatures, which
   every builder can apply before attesting to the final archives.

The attestation layout and detached-signature split follow Bitcoin Core's Guix
release process. macOS signatures use
[`signapple`](https://github.com/achow101/signapple), pinned to Bitcoin Core's
revision. Windows signatures use `osslsigncode`. Signing keys never enter a Nix
build or the Floresta repository.

## Supported targets

Run `contrib/reproducible/nix-build` without arguments to build this complete
matrix:

| Build name | Rust target | Archive | Platform signing |
| --- | --- | --- | --- |
| `linux-x86_64` | `x86_64-unknown-linux-musl` | `.tar.gz` | none |
| `linux-aarch64` | `aarch64-unknown-linux-musl` | `.tar.gz` | none |
| `linux-riscv64` | `riscv64gc-unknown-linux-musl` | `.tar.gz` | none |
| `windows-x86_64` | `x86_64-pc-windows-gnu` | `.zip` | Authenticode |
| `windows-aarch64` | `aarch64-pc-windows-gnullvm` | `.zip` | Authenticode |
| `macos-x86_64` | `x86_64-apple-darwin` | `.tar.gz` | Apple Developer ID |
| `macos-aarch64` | `aarch64-apple-darwin` | `.tar.gz` | Apple Developer ID |

Linux archives are statically linked with musl. ARM means 64-bit AArch64 for
current desktop/server releases. RISC-V is currently a Linux target: neither
macOS nor Windows defines a supported RISC-V userspace target, so those two
cross-products do not exist.

## Prerequisites

Use an `x86_64` Linux installation with:

- Git;
- Nix 2.18 or newer with flakes enabled;
- approximately 80 GiB of free disk space for the full matrix.

Install Nix using the [official installer](https://nixos.org/download/). A
multi-user installation is recommended. The host compiler, Rust installation,
and system libraries are not used. Nix obtains the exact Rust toolchains,
C/C++ cross-toolchains, Zig, macOS 11.3 SDK, MinGW, archive tools, and signing
tools from the locked inputs. The SDK is used only for macOS cross-compilation
and remains subject to Apple's Xcode and SDK license terms.

Release builds must use a clean checkout. Fetch tags and check out the exact
signed release tag before building:

```bash
git clone https://github.com/getfloresta/Floresta.git
cd Floresta
git fetch --tags origin
git checkout v0.9.0
```

The tag must match the `florestad` version. On an untagged commit, output is
namespaced by `VERSION-<12-character-commit>` instead. `FORCE_DIRTY_WORKTREE=1`
is only for development; outputs from a dirty tree must not be published or
attested.

## Build

Build every target:

```bash
contrib/reproducible/nix-build
```

Build a subset by naming it:

```bash
contrib/reproducible/nix-build linux-riscv64 windows-x86_64
```

The default output is:

```text
builds/nix-<tag-or-commit>/output/
  linux-x86_64/
    floresta-<version>-linux-x86_64.tar.gz
    floresta-<version>-linux-x86_64-buildinfo.json
    SHA256SUMS.part
  windows-x86_64/
    floresta-<version>-windows-x86_64-unsigned.zip
    floresta-<version>-windows-x86_64-codesigning.tar.gz
    floresta-<version>-windows-x86_64-buildinfo.json
    SHA256SUMS.part
  ...
```

`OUTDIR_BASE=/absolute/path` selects another output directory. Existing target
output is never overwritten. Remove an incomplete generated target directory
before retrying it.

The Nix derivations fix the source tree, dependency graph, toolchains, target
SDKs, build path mapping, locale, timestamps, ownership, permissions, archive
ordering, and compression options. Each `SHA256SUMS.part` is checked as it is
copied out of the Nix store.

## Independently reproduce and attest

Each builder needs a checkout of a shared, otherwise empty attestation Git
repository. It is intentionally separate from the source repository, like
Bitcoin Core's `guix.sigs` repository. Import the public GPG keys of every
builder whose attestation will be trusted.

After building the same clean tag or commit, create the manifest and detached
GPG signature:

```bash
ATTESTATIONS_REPO="$HOME/src/floresta-attestations" \
SIGNER='GPG-FINGERPRINT=builder-name' \
  contrib/reproducible/nix-attest
```

This writes:

```text
<attestations>/<tag-or-commit>/<builder-name>/
  noncodesigned.SHA256SUMS
  noncodesigned.SHA256SUMS.asc
```

Commit those two files and submit them to the shared attestation repository.
`SIGNER=GPG-FINGERPRINT` uses the fingerprint as the directory name. For a dry
run, `NO_SIGN=1` writes only the manifest.

After pulling at least one other builder's attestation, verify the signatures
and byte-for-byte agreement:

```bash
ATTESTATIONS_REPO="$HOME/src/floresta-attestations" \
MIN_ATTESTATIONS=2 \
  contrib/reproducible/nix-verify
```

`MIN_ATTESTATIONS` is a release-policy threshold, not a substitute for checking
who controls each imported key. Maintainers should choose and publish the
required number of independent builders before the first release. The verifier
counts distinct valid signing-key fingerprints and fails on a missing/invalid
GPG signature, a differing manifest, or too few attestations.

## Detached Windows and macOS signing

Code signing is deliberately a second phase. Authenticode timestamping and
Apple notarization contact external services and cannot themselves be
reproduced. A designated signer creates one detached signature payload. All
builders then apply that identical payload with pinned tools, making the final
signed archives reproducible.

### Create Windows detached signatures

Enter the pinned signing environment from the Floresta checkout:

```bash
nix develop .#codesign
```

Extract the target's codesigning archive and run its included script:

```bash
mkdir -p /tmp/floresta-sign
cd /tmp/floresta-sign
tar -xf /path/to/floresta-0.9.0-windows-x86_64-codesigning.tar.gz
cd floresta-0.9.0-windows-x86_64-codesigning
./detached-sig-create /secure/code-signing-certificate.pem \
  /secure/code-signing-private-key.pem
```

The script prompts for the private-key password and uses Bitcoin Core's
`osslsigncode` detached-signature workflow. For unattended signing,
`CODESIGN_PASSWORD` may be supplied by a secret runner. `TIMESTAMP_SERVER`
overrides the Authenticode server.

### Create macOS detached signatures

In the same pinned `nix develop .#codesign` environment:

```bash
mkdir -p /tmp/floresta-sign
cd /tmp/floresta-sign
tar -xf /path/to/floresta-0.9.0-macos-aarch64-codesigning.tar.gz
cd floresta-0.9.0-macos-aarch64-codesigning
./detached-sig-create /secure/developer-id.p12 \
  /secure/AuthKey_ID.p8 \
  APP-STORE-CONNECT-ISSUER-ID
```

This uses Bitcoin Core's pinned `signapple` revision to sign with hardened
runtime, emit detached signatures, and submit the signed binaries for
notarization. It prompts for both key passwords. Secret runners may set
`CODESIGN_PASSWORD` and `NOTARIZATION_PASSWORD`.

### Publish and apply detached signatures

Each command above creates `signature-<target>.tar.gz`. Extract every approved
payload at the root of a dedicated detached-signature Git repository, commit
it, and create a signed tag identifying the Floresta release:

```bash
tar -C "$HOME/src/floresta-detached-sigs" \
  -xf signature-windows-x86_64.tar.gz
git -C "$HOME/src/floresta-detached-sigs" add windows-x86_64
git -C "$HOME/src/floresta-detached-sigs" commit \
  -m '0.9.0: Windows x86_64 signatures'
git -C "$HOME/src/floresta-detached-sigs" tag -s v0.9.0
```

After checking out the approved detached-signature commit or tag, every builder
applies the signatures to its own unsigned output:

```bash
DETACHED_SIGS_REPO="$HOME/src/floresta-detached-sigs" \
  contrib/reproducible/nix-codesign
```

The command automatically enters the pinned Nix signing environment. A target
subset can be passed in the same way as `nix-build`. Final archives appear in
`output/<target>-codesigned/`.
Set `CODESIGN_CA_FILE=/path/to/ca-chain.pem` when the Windows certificate's
issuer is not in the pinned public CA bundle.

Run `nix-attest` again. It preserves `noncodesigned.SHA256SUMS` and adds
`all.SHA256SUMS` plus `all.SHA256SUMS.asc`, covering both unsigned inputs and
final signed archives. Submit those files to the attestation repository and run
`nix-verify` again. Publish one agreed `all.SHA256SUMS` and the detached GPG
signatures from the builders that meet the release threshold.

## Docker alternative

Docker is useful for testing this workflow but is not part of the release trust
model. Build the pinned helper image from a clean checkout:

```bash
docker build -f contrib/reproducible/Dockerfile -t floresta-reproducible .
```

Run all targets while preserving outputs and the Nix store cache:

```bash
mkdir -p builds
docker volume create floresta-nix-store
docker run --rm \
  -e OUTPUT_UID="$(id -u)" \
  -e OUTPUT_GID="$(id -g)" \
  -v floresta-nix-store:/nix \
  -v "$PWD/builds:/src/builds" \
  floresta-reproducible
```

Append target names to build a subset:

```bash
docker run --rm \
  -e OUTPUT_UID="$(id -u)" \
  -e OUTPUT_GID="$(id -g)" \
  -v floresta-nix-store:/nix \
  -v "$PWD/builds:/src/builds" \
  floresta-reproducible linux-x86_64 linux-riscv64
```

The container is only a launcher. `flake.lock`, rather than the Docker host,
still defines every build input.

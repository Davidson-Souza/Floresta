# Using Nix

The repository flake pins the development and reproducible-release toolchains.
Install Nix with flakes enabled, then run commands from the repository root.

## Development shell

```bash
nix develop
```

The default shell provides Rust tooling and the native dependencies used by
normal development builds and tests.

## Reproducible release packages

Release derivations are exposed on an `x86_64-linux` builder:

```bash
nix build .#release-linux-x86_64
nix build .#release-linux-aarch64
nix build .#release-linux-riscv64
nix build .#release-windows-x86_64
nix build .#release-windows-aarch64
nix build .#release-macos-x86_64
nix build .#release-macos-aarch64
```

Prefer `contrib/reproducible/nix-build` for releases. It builds the requested
matrix, copies artifacts out of the Nix store, and verifies each generated
checksum manifest. The complete independent-builder, attestation, and detached
code-signing process is documented in
[Reproducible cross-platform releases](reproducible-builds.md).

## Platform-signing shell

```bash
nix develop .#codesign
```

This shell contains the pinned `signapple`, `osslsigncode`, GPG, and deterministic
archive tools used by the release signing workflow.

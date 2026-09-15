#!/usr/bin/env bash
# Copyright (c) 2026 The Floresta developers
# Distributed under the MIT software license; see LICENSE-MIT.

export LC_ALL=C
export TZ=UTC
umask 0022

readonly DEFAULT_TARGETS="linux-x86_64 linux-aarch64 linux-riscv64 windows-x86_64 windows-aarch64 macos-x86_64 macos-aarch64"
readonly SIGNABLE_TARGETS="windows-x86_64 windows-aarch64 macos-x86_64 macos-aarch64"

print_error() {
    printf 'error: %s\n' "$*" >&2
}

die() {
    print_error "$@"
    exit 1
}

require_tools() {
    local tool
    for tool in "$@"; do
        command -v "$tool" >/dev/null 2>&1 || die "required command not found: $tool"
    done
}

check_top_directory() {
    [[ -f Cargo.toml && -f flake.nix && -x contrib/reproducible/nix-build ]] \
        || die 'run this command from the top of the Floresta repository'
}

release_version() {
    local version
    version="$(sed -n 's/^version = "\([^"]*\)".*/\1/p' bin/florestad/Cargo.toml)"
    [[ -n "$version" && "$version" != *$'\n'* ]] || die 'could not read florestad version'
    printf '%s\n' "$version"
}

build_id() {
    local version tag revision
    version="$(release_version)"
    if tag="$(git describe --exact-match --tags HEAD 2>/dev/null)"; then
        [[ "$tag" == "v$version" || "$tag" == "v$version"-* ]] \
            || die "tag $tag does not match florestad version $version"
        printf '%s\n' "${tag#v}"
    else
        revision="$(git rev-parse --short=12 HEAD)"
        printf '%s-%s\n' "$version" "$revision"
    fi
}

source_date_epoch() {
    git -c log.showSignature=false log -1 --format=%ct
}

check_clean_worktree() {
    if [[ -z "${FORCE_DIRTY_WORKTREE:-}" && -n "$(git status --porcelain --untracked-files=normal)" ]]; then
        die 'the Git worktree is dirty; commit/stash changes or set FORCE_DIRTY_WORKTREE=1'
    fi
}

validate_target() {
    local requested="$1" supported
    for supported in $DEFAULT_TARGETS; do
        [[ "$requested" == "$supported" ]] && return 0
    done
    die "unsupported target '$requested' (supported: $DEFAULT_TARGETS)"
}

is_signable_target() {
    local requested="$1" signable
    for signable in $SIGNABLE_TARGETS; do
        [[ "$requested" == "$signable" ]] && return 0
    done
    return 1
}

output_base() {
    printf '%s\n' "${OUTDIR_BASE:-$PWD/builds/nix-$(build_id)/output}"
}

attestation_version() {
    printf '%s\n' "${VERSION:-$(build_id)}"
}

#!/usr/bin/env bash

# SPDX-License-Identifier: MIT OR Apache-2.0

set -e

# Parse CLI flags:
# --build    : force rebuilding utreexod/bitcoind even if present
# --release  : build florestad in release mode (default: debug)
FORCE_BUILD=0
BUILD_RELEASE=0
for ARG in "$@"; do
    case "$ARG" in
        --build) FORCE_BUILD=1 ;;
        --release) BUILD_RELEASE=1 ;;
        *) ;;
    esac
done

BITCOIN_REVISION="${BITCOIN_REVISION:-31.0}"
# We need for the current dir to be the root dir of the project.
FLORESTA_PROJ_DIR=$(git rev-parse --show-toplevel)
TEMP_DIR="${FLORESTA_TEMP_DIR:-/tmp/floresta-func-tests}"
BINARIES_DIR="$TEMP_DIR/binaries"
# Dont use mktemp so we can have deterministic results for each version of floresta.
mkdir -p "$BINARIES_DIR"

BINARIES_DIR="$(cd "$BINARIES_DIR" && pwd)"

# Create a temporary disposable directory, switch to it, and ensure it is removed on function exit
create_disposable_dir() {
    DISPOSABLE_DIR=$(mktemp -d)
    trap 'rm -rf -- "$DISPOSABLE_DIR"' RETURN
    echo "$DISPOSABLE_DIR"
    # Change to the disposable directory
    pushd "$DISPOSABLE_DIR" >/dev/null
}

check_installed() {
    if ! command -v "$1" &>/dev/null; then
        echo "You must have $1 installed to run those tests!"
        exit 1
    fi
}

# Check for a C++ compiler
check_installed_compiler() {
    if command -v gcc &>/dev/null; then
        echo "GCC is installed."
        return 0
    elif command -v clang &>/dev/null; then
        echo "Clang is installed."
        return 0
    else
        echo "You must have either GCC or Clang installed to run these tests!"
        exit 1
    fi
}

# If the user provided a bitcoind binary via BITCOIND_EXE, use it.
try_use_provided_bitcoind() {
    if [ -n "${BITCOIND_EXE:-}" ]; then
        if [ ! -f "$BITCOIND_EXE" ] || [ ! -x "$BITCOIND_EXE" ]; then
            echo "BITCOIND_EXE is set but does not point to an executable: $BITCOIND_EXE" >&2
            # Fail hard because user explicitly requested a custom binary.
            exit 1
        fi
        cp "$BITCOIND_EXE" "$BINARIES_DIR/bitcoind"
        chmod +x "$BINARIES_DIR/bitcoind"
        echo "Using user-provided bitcoind: $BINARIES_DIR/bitcoind"
        return 0
    fi
    return 1
}

download_prebuilt_bitcoind() {
    HASH_FILE="${FLORESTA_PROJ_DIR}/tests/bitcoin_hashes/${BITCOIN_REVISION}"
    # If the SHA256SUMS file for this revision does not exist, that revision isn't supported.
    if [ ! -f "$HASH_FILE" ]; then
        echo "No SHA256SUMS found for Bitcoin Core revision '${BITCOIN_REVISION}' at: $HASH_FILE"
        return 1
    fi

    UNAME_S="$(uname -s)"
    UNAME_M="$(uname -m)"

    # Map uname -> platform name used by Bitcoin Core distribution filenames
    case "$UNAME_S" in
        Linux)
            case "$UNAME_M" in
                x86_64) PLATFORM="x86_64-linux-gnu" ;;
                aarch64|arm64) PLATFORM="aarch64-linux-gnu" ;;
                armv7l) PLATFORM="arm-linux-gnueabihf" ;;
                *) echo "Unsupported architecture for prebuilt bitcoind: $UNAME_M"; return 1 ;;
            esac
            FILE_EXT="tar.gz"
            ;;
        Darwin)
            case "$UNAME_M" in
                x86_64) PLATFORM="x86_64-apple-darwin" ;;
                aarch64|arm64) PLATFORM="arm64-apple-darwin" ;;
                *) echo "Unsupported architecture for prebuilt bitcoind on macOS: $UNAME_M"; return 1 ;;
            esac
            FILE_EXT="tar.gz"
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            PLATFORM="win64"
            FILE_EXT="zip"
            ;;
        *)
            echo "Unsupported OS for prebuilt bitcoind: $UNAME_S"
            return 1
            ;;
    esac

    FILE_NAME="bitcoin-${BITCOIN_REVISION}-${PLATFORM}.${FILE_EXT}"

    HASH=$(awk -v f="$FILE_NAME" '$2==f {print $1; exit}' "$HASH_FILE" || true)
    if [ -z "$HASH" ]; then
        echo "No prebuilt hash for $FILE_NAME in $HASH_FILE"
        return 1
    fi

    DL_URL="https://bitcoincore.org/bin/bitcoin-core-${BITCOIN_REVISION}/${FILE_NAME}"

    # Change to a disposable directory for download and extraction
    DISPOSABLE_DIR=$(create_disposable_dir)

    echo "Downloading $DL_URL"
    if ! curl -L -o "$FILE_NAME" "$DL_URL"; then
        echo "Failed to download $DL_URL"
        return 1
    fi

    # Normalize hash before verify
    DOWNLOADED_SHA256=$({ sha256sum "$FILE_NAME" 2>/dev/null || shasum -a 256 "$FILE_NAME"; } | awk '{print $1}' | tr -d '\r')
    EXPECTED_SHA256=${HASH%%$'\r'}

    if [ "$DOWNLOADED_SHA256" != "$EXPECTED_SHA256" ]; then
        printf 'SHA256 mismatch for %s\nExpected: %s\nActual:   %s\n' "$FILE_NAME" "$EXPECTED_SHA256" "$DOWNLOADED_SHA256"
        exit 1
    fi

    if ! tar xzf "$FILE_NAME"; then
        echo "Failed to extract $FILE_NAME"
        return 1
    fi

    cp "bitcoin-${BITCOIN_REVISION}/bin/bitcoind" "$BINARIES_DIR/bitcoind"
    chmod +x "$BINARIES_DIR/bitcoind"

    echo "bitcoind downloaded to $BINARIES_DIR/bitcoind"
    return 0
}

build_bitcoind_from_source() {
    check_installed_compiler
    check_installed make
    check_installed cmake

    # Change to a disposable directory for download and build
    DISPOSABLE_DIR=$(create_disposable_dir)

    echo "Downloading and Building Bitcoin Core..."
    git clone https://github.com/bitcoin/bitcoin "$DISPOSABLE_DIR/bitcoin"
    cd "$DISPOSABLE_DIR/bitcoin" || exit 1

    # Determine current ref: prefer branch name, fall back to exact tag when detached.
    current_ref="$(git symbolic-ref -q --short HEAD 2>/dev/null || true)"
    if [ -z "$current_ref" ]; then
        # If HEAD is detached but exactly at a tag, use that tag name.
        current_ref="$(git describe --tags --exact-match 2>/dev/null || true)"
    fi

    # If already on requested ref (with or without leading "v"), skip checkout.
    if [ "$current_ref" = "$BITCOIN_REVISION" ] || [ "$current_ref" = "v$BITCOIN_REVISION" ]; then
        echo "Already on '$current_ref', skipping checkout"
    else
        # Try tag "v<revision>"
        if git show-ref --verify --quiet "refs/tags/v$BITCOIN_REVISION"; then
            git checkout "v$BITCOIN_REVISION" || return 1
        # Try local branch "<revision>"
        elif git show-ref --verify --quiet "refs/heads/$BITCOIN_REVISION"; then
            git checkout "$BITCOIN_REVISION" || return 1
        # Try remote branch "origin/<revision>"
        elif git ls-remote --heads origin "$BITCOIN_REVISION" | grep -q .; then
            git checkout -b "$BITCOIN_REVISION" "origin/$BITCOIN_REVISION" || return 1
        else
            echo "bitcoin '$BITCOIN_REVISION' is not a valid tag or branch."
            return 1
        fi
    fi

    # Check compatibility with cmake arguments with those used with make
    # See https://gist.github.com/hebasto/2ef97d3a726bfce08ded9df07f7dab5e and
    # https://github.com/bitcoin-core/bitcoin-devwiki/wiki/Autotools-to-CMake-Options-Mapping
    rev="${BITCOIN_REVISION#v}"
    # Normalize revision: remove leading 'v' if present, then extract leading digits.
    if [[ "$rev" =~ ^([0-9]+) ]]; then
        major_version="${BASH_REMATCH[1]}"
    else
        # Non-numeric branch names (e.g. "master") => treat as modern (use cmake path).
        major_version=999
    fi
    if [ "$major_version" -ge 29 ]; then
        cmake -S . -B build \
            -DBUILD_CLI=OFF \
            -DBUILD_TESTS=OFF \
            -DCMAKE_BUILD_TYPE=MinSizeRel \
            -DENABLE_EXTERNAL_SIGNER=OFF \
            -DENABLE_IPC=OFF \
            -DINSTALL_MAN=OFF
        cmake_nprocs="${BUILD_BITCOIND_NPROCS:-4}" || exit 1
        cmake --build build --target bitcoind -j"${cmake_nprocs}" || exit 1
        mv "$DISPOSABLE_DIR/bitcoin/build/bin/bitcoind" "$BINARIES_DIR/bitcoind" || exit 1
    else
        ./autogen.sh
        ./configure \
            --without-gui \
            --disable-tests \
            --disable-bench \
        make_nprocs="${BUILD_BITCOIND_NPROCS:-4}" || exit 1
        make -j"$(make_nprocs)" || exit 1
        mv "$DISPOSABLE_DIR/bitcoin/src/bitcoind" "$BINARIES_DIR/bitcoind" || exit 1
    fi

    return 0
}

ensure_bitcoind() {
    # 1) user-provided binary
    if try_use_provided_bitcoind; then
        return 0
    fi

    # 2) try download prebuilt tarball for supported versions/platforms
    if download_prebuilt_bitcoind; then
        return 0
    fi

    # 3) try to build from source
    if build_bitcoind_from_source; then
        return 0
    fi

    echo "Failed to obtain bitcoind (tried BITCOIND_EXE and prebuilt tarball)"
    return 1
}

build_utreexod() {
    # Change to a disposable directory for download and build
    DISPOSABLE_DIR=$(create_disposable_dir)

    echo "Downloading and Building utreexod..."
    git clone https://github.com/utreexo/utreexod "$DISPOSABLE_DIR/utreexod" || exit 1

    cd "$DISPOSABLE_DIR/utreexod"

    echo "Building utreexod..."
    go build -o utreexod . || exit 1

    echo "Copying utreexod to binaries directory..."
    cp -f utreexod "$BINARIES_DIR/utreexod" || exit 1
    chmod +x "$BINARIES_DIR/utreexod"

    echo "Utreexod built successfully."
}

build_floresta() {
    # We dont check if floresta already exist because a floresta binary could be already be installed on PATH
    # causing collisions with the tests.
    echo "Building florestad..."

    cd "$FLORESTA_PROJ_DIR"

    if [ "$BUILD_RELEASE" -eq 1 ]; then
        echo "Building florestad (release)..."
        cargo build --bin florestad --release
        PROFILE="release"
    else
        echo "Building florestad (debug)..."
        cargo build --bin florestad
        PROFILE="debug"
    fi

    ln -fs "$(pwd)/target/${PROFILE}/florestad" "$BINARIES_DIR/florestad"
}

check_installed git
check_installed cargo
check_installed go

build_floresta

# Check if utreexod is already built or if --build is passed
if [ ! -f "$BINARIES_DIR/utreexod" ] || [ "$FORCE_BUILD" -eq 1 ]; then
    build_utreexod
else
    echo "Utreexod already built, skipping..."
fi

# Ensure bitcoind is obtained (downloaded, built, or reused) if --build is passed or not already present
if [ ! -f "$BINARIES_DIR/bitcoind" ] || [ "$FORCE_BUILD" -eq 1 ]; then
    ensure_bitcoind
else
    echo "Bitcoind already built/downloaded, skipping..."
fi

echo "All done!"

echo "Temporary Directory at $TEMP_DIR"

exit 0

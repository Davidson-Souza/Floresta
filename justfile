_default:
    @just --list

# Checks whether a command is available.
check-command cmd recipe="check-command" link_to_package="":
    @if ! command -v "{{ cmd }}" >/dev/null; then \
        echo "Command '{{ cmd }}' is not available, but 'just {{ recipe }}' requires it." >&2; \
        if [ -n "{{ link_to_package }}" ]; then \
            echo "This might help you: {{ link_to_package }}" >&2; \
        fi; \
        exit 1; \
    fi

# Run Florestad locally
run:
    cargo run --bin florestad

# Runs Florestad locally with release options
run-release:
    cargo run --release --bin florestad

# Compile project with debug options
build:
    cargo build

# Compile project with release options
build-release:
    cargo build --release

# Clean project build directory
clean:
    cargo clean

# Execute all tests
test crate="":
    @just test-doc {{ crate }}
    @just test-unit {{ crate }}
    @just test-wkspc

# Execute doc tests
test-doc crate="": build
    cargo test {{ crate }} --doc

# Execute unit tests
test-unit crate="": build
    cargo test --lib {{ crate }} -- --nocapture

# Execute workspace-related tests
test-wkspc: build
    cargo test --workspace -- --nocapture

# Execute tests/prepare.sh.
test-functional-prepare arg="":
    bash tests/prepare.sh {{ arg }}

# Execute tests/run.sh
test-functional-run arg="":
    bash tests/run.sh {{ arg }}

# Execute tests/run.sh -n 1 --run-expensive ./tests/expensive
test-expensive-functional-run:
    bash tests/run.sh -n 1 --run-expensive ./tests/expensive

# Format and lint functional tests
test-functional-uv-fmt:
    @just check-command uv test-functional-uv-fmt "https://docs.astral.sh/uv/getting-started/installation/"
    uv run black --verbose ./tests
    uv run pylint --verbose $(find ./tests -name "*.py")

# Run the functional tests
test-functional:
    @just test-functional-prepare
    @just test-functional-run
    @just test-expensive-functional-run

# Run the benchmarks
bench:
    cargo bench -p floresta-chain --no-default-features --features test-utils,flat-chainstore

# Generate the public documentation for all crates
doc:
    RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --workspace --no-deps --lib --all-features

# Generate and open the public documentation for all crates
open-doc:
    RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --workspace --no-deps --all-features --lib --open

# Generate the documentation for all crates, including private items, and fail on warnings
doc-check:
    RUSTDOCFLAGS="--cfg docsrs -D warnings" \
    cargo +nightly doc --workspace --no-deps --all-features --lib --document-private-items

# Format code and run configured linters
lint:
    @just fmt
    @just doc-check

    # 1) Run with no features
    cargo +nightly clippy --workspace --all-targets --no-default-features

    # 2) Run with all features
    cargo +nightly clippy --workspace --all-targets --all-features

    @just spell-check

    # Lint the functional tests
    @just test-functional-uv-fmt

# Format code
fmt:
    cargo +nightly fmt --all

# Checks the formatting
format:
    cargo +nightly fmt --all --check

# Test all feature combinations in each crate (arg: optional, e.g., --quiet or --verbose)
test-features arg="":
    @just check-command "cargo-hack" "test-features" "cargo install cargo-hack --locked --version 0.6.34"
    ./contrib/feature_matrix.sh test {{ arg }}

# Format code and run clippy for all feature combinations in each crate (arg: optional, e.g., '-- -D warnings')
lint-features arg="":
    @just check-command "cargo-hack" "lint-features" "cargo install cargo-hack --locked --version 0.6.34"

    @just fmt
    @just doc-check
    @just spell-check

    ./contrib/feature_matrix.sh clippy '{{ arg }}'

    @just test-functional-uv-fmt

# Remove test-generated data
clean-data:
    ./contrib/clean_data.sh

# Run all needed checks before contributing code (pre-commit check)
pcc:
    @just check-command "cargo-hack" "pcc" "cargo install cargo-hack --locked --version 0.6.34"

    just lint-features '-- -D warnings'
    just test-features
    just test-functional
    just check-flake

# Run nix flake check if Nix configuration changed and nix is available
check-flake:
    @if command -v nix >/dev/null && git diff --name-only HEAD | grep -qE '^(flake\.(nix|lock)|nix/)'; then \
        echo "Nix configuration changed, running nix flake check..."; \
        nix flake check; \
    fi

# Must have pandoc installed
# Needs sudo to overwrite existing man pages

# Convert all markdown files on /doc/rpc/ to man pages on /doc/rpc_man/
gen-manpages path="":
    @just check-command pandoc gen-manpages "https://pandoc.org/installing.html"
    ./contrib/dist/gen_manpages.sh {{ path }}

# Run typos
spell-check:
    @just check-command typos spell-check "cargo +nightly install typos-cli --locked"
    typos --config _typos.toml

# Usage:
#   just install                   # installs both florestad and floresta-cli
#   just install florestad         # installs only florestad
#   just install floresta-cli      # installs only floresta-cli
#

# Floresta recipe to help installing the binaries without versioning problems.
install bin="all":
    if [ "{{ bin }}" = "all" ]; then \
        cargo install --path bin/florestad --locked && \
        cargo install --path bin/floresta-cli --locked; \
    elif [ "{{ bin }}" = "florestad" ]; then \
        cargo install --path bin/florestad --locked; \
    elif [ "{{ bin }}" = "floresta-cli" ]; then \
        cargo install --path bin/floresta-cli --locked; \
    else \
        printf "Unknown binary: %s\n" "{{ bin }}" >&2; \
        exit 1; \
    fi

# Tests

This document is a guide for the different testing options available in Floresta. We have an extensive suite of Rust tests, as well as a functional tests Python framework, found in the [tests directory](../tests). For fuzzing tests refer to [this document](fuzzing.md).

## Requirements

The tests in `floresta-cli` depend on the compiled `florestad` binary. Make sure to build the entire project first by running:

```bash
# Build florestad and floresta-cli separately
cargo build -p florestad
cargo build -p floresta-cli
```

The functional tests also need some dependencies, we use python for writing them and `uv` to manage its dependencies.

### Dependencies requirements to run functional tests

The functional tests will build Bitcoin Core, Utreexo and Floresta in order to make integration testing. To do so it will use some dependencies.

The following guide is a compilation taken from [Bitcoin](https://github.com/bitcoin/bitcoin/tree/master/doc) and [Utreexo](https://github.com/utreexo/utreexod/). It considers the user running the tests already has the required dependencies for building [Floresta](https://github.com/getfloresta/Floresta/tree/master/doc).

#### Ubuntu & Debian

```bash
sudo apt-get install build-essential cmake pkgconf python3 libevent-dev libboost-dev golang
```

#### Fedora

```bash
sudo dnf install gcc-c++ cmake make python3 libevent-devel boost-devel golang
```

#### MacOS

```bash
brew install cmake boost pkgconf libevent coreutils go
```

#### Installing UV

UV is an extremely fast Python package and project manager, written in Rust.

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Testing Options
There's a set of unit and integration Rust tests that you can run with:

```bash
cargo test
```

For the full test suite, including long-running tests, use:

```bash
cargo test --release
```

Next sections will cover the Python functional tests.

### Setting Functional Tests Binaries

We provide three way for running functional tests:
* from `just` tool that abstracts what is necessary to run the tests before doing a commit;
* from helper scripts — [prepare.sh](https://github.com/getfloresta/Floresta/blob/master/tests/prepare.sh) and [run.sh](https://github.com/getfloresta/Floresta/blob/master/tests/run.sh) — to automatically build and run the tests;
* from python utility directly: the most laborious, but you can run a specific test suite.

#### From `just` tool
It abstracts all things that will be explained in the next sections, and for that
reason, we recommend to use it before doing a commit when changes only the functional tests.

```bash
just test-functional
```

Furthermore, you can only specific tests, rather than all at once.

```bash
# runs all tests in 'floresta-cli' suite
just test-functional-run "--test-suite floresta-cli"

# same as above
just test-functional-run "-t floresta-cli"

# run the stop and ping tests in the floresta-cli suite
just test-functional-run "--test-suite floresta-cli --test-name stop --test-name ping"

# same as above
just test-functional-run "-t floresta-cli -k stop -k ping"

# run many tests that start with the word `getblock` (getblockhash, getblockheader, etc...)
just test-functional-run "-t floresta-cli -k getblock"
```

#### From helper scripts

We provide two helper scripts to support our functional tests in this process and guarantee isolation and reproducibility.

* [prepare.sh](https://github.com/getfloresta/Floresta/blob/master/tests/prepare.sh) checks for build dependencies for both `utreexod` and `florestad`, builds them, and sets the `$FLORESTA_TEMP_DIR` environment variable. This variable points to where our functional tests will look for the binaries — specifically at `$FLORESTA_TEMP_DIR/binaries`.

* [run.sh](https://github.com/getfloresta/Floresta/blob/master/tests/run.sh) adds the binaries found at `$FLORESTA_TEMP_DIR/binaries` to your `$PATH` and runs the tests in that environment.

So a basic usage would be:

```bash
./tests/prepare.sh && ./tests/run.sh
```

##### Utreexod

By default, the tool will build `utreexod` on its [latest release](https://github.com/utreexo/utreexod/releases/latest).
If you want to build a specific release, you can set the `UTREEXO_REVISION` environment variable before running the script.
It must be a [valid tag](https://github.com/utreexo/utreexod/tags) without the `v` prefix. For example:

```bash
UTREEXO_REVISION=0.1.0 ./tests/prepare.sh && ./tests/run.sh
```

##### Bitcoin-core
By default, the `prepare.sh` script will obtain a runnable `bitcoind` binary in one of three exclusive ways. The default Bitcoin Core version is `31.0`, but you can override this by setting the `BITCOIN_REVISION` environment variable. The three methods are:

1. **Using a user-provided binary**: If the `BITCOIND_EXE` environment variable is set and points to an executable, that exact binary is used. No download or build is attempted, and any `BITCOIN_REVISION` or build-parallelism settings are ignored.
```bash
    BITCOIND_EXE=/path/to/bitcoind ./tests/prepare.sh
```

2. **Downloading a prebuilt binary**: If `BITCOIND_EXE` is not set, the script will try to download a prebuilt Bitcoin Core tarball for the specified `BITCOIN_REVISION`. Prebuilt binaries are available for all platforms and operating systems supported by Bitcoin Core. The supported versions are `31.0`, `30.2`, `29.2`, `28.3` and `27.2`.
```bash
    BITCOIN_REVISION=28.3 ./tests/prepare.sh
```

3. **Building from source**: If no prebuilt binary is available for the platform, operating system, or specified version, the script will clone the Bitcoin Core repository and build `bitcoind` from the specified `BITCOIN_REVISION`. This can be a version tag (e.g., `29.1`) or a branch(e.g., `master`) from the remote repository.
```bash
    BITCOIN_REVISION=master ./tests/prepare.sh
```

Additionally, you can use some arguments in those scripts:

```bash
./tests/prepare.sh --build --release && ./tests/run.sh --preserve-data-dir
```

The `--build` argument will force the script to fetch the `bitcoind` binary again and to build the `utreexod`.

The `--release` argument will build the `florestad` binary in release mode, which is optimized for production use. If this flag is not provided, the binary will be built in debug mode by default.

The `--preserve-data-dir` argument will keep the data and logs directories after running the tests
(this is useful if you want to keep the data for debugging purposes).

Furthermore, you can run a set of specific tests, rather than all at once.

```bash
# runs all tests in 'floresta-cli' suite
./tests/run.sh --test-suite floresta-cli

# same as above
./tests/run.sh -t floresta-cli

# run the stop and ping tests in the floresta-cli suite
./tests/run.sh --test-suite floresta-cli --test-name stop --test-name ping

# same as above
./tests/run.sh -t floresta-cli -k stop -k ping

# run many tests that start with the word `getblock` (getblockhash, getblockheader, etc...)
./tests/run.sh -t floresta-cli -k getblock
```

#### From python utility directly
Additional functional tests are available (minimum python version: 3.12).
It's not recommended to run them directly, since you will need to manually
build the binaries yourself and place them at `$FLORESTA_TEMP_DIR/binaries`.
The advantage is that you can run a specific test suite. For this you'll need to:

* Setup `floresta`/`utreexod` environment;
* Setup python utility;
* Run tests from python utility directly;
* Clean up the environment.

##### Setup `floresta`/`utreexod` environment

After build the `floresta` and `utreexod` binaries, you'll need to define
a `FLORESTA_TEMP_DIR` environment variable. This variable points to where
our functional tests will look for the binaries.

##### Setup python utility
* Recommended: install [uv: a rust-based python package and project manager](https://docs.astral.sh/uv/).

* Configure an isolated environment:

```bash
# create a virtual environment
# (it's good to not mess up with your os)
uv venv

# Alternatively, you can specify a python version (e.g, 3.12),
uv venv --python 3.12

# activate the python virtual environment
source .venv/bin/activate

# check if the python path was modified
which python
```

* Install module dependencies:

```bash
# Install Python test dependencies declared
# on `pyproject.toml` on a virtual environment
uv pip install -r pyproject.toml

# Alternatively, synchronize dependencies from `pyproject.toml` and `uv.lock`
uv sync
```

* Format code
```bash
uv run black ./tests

# if you want to just check
uv run black --check --verbose ./tests
```


* Lint code
```bash
uv run pylint ./tests
```

##### Run tests from python utility directly

Our tests are separated by "test suites". Suites are folders located in `./tests/<suite>` and the tests are the `./tests/<suite>/*-test.py` files. To run all suites, type:

```bash
FLORESTA_TEMP_DIR=<your_bin_dir> uv run tests/test_runner.py
```

You can list all suites with:

```bash
FLORESTA_TEMP_DIR=<your_bin_dir> uv run tests/test_runner.py --list-suites
```

To run a specific suite:

```bash
FLORESTA_TEMP_DIR=<your_bin_dir> uv run tests/test_runner.py --test-suite <suite>
```

You can even add more:

```bash
FLORESTA_TEMP_DIR=<your_bin_dir> uv run tests/test_runner.py --test-suite <suite_A> --test-suite <suite_B>
```

##### Clean up the environment

If you tests fails it will be necessary to cleanup the `data`
folder created by the tests (some tests use it to retain
information about tested nodes, like the `addnode` command).

You can do this by running:

```bash
rm -rf FLORESTA_TEMP_DIR/data
```

### Running/Developing Functional Tests with Nix

If you have nix, you can run the tests following the instructions [here](nix.md).

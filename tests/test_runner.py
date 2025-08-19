"""
test_runner.py

Command Line Interface to run an individual test or multiple tests in a suite.

New test suites should be added as a subdirectory of `./tests`.

Running tests using the `uv` Python package manager is recommended, although you can run them
using your system's Python directly (the use of a virtual environment is highly advised).

Test running is parallelized: each test will spawn a new thread as a subprocess of the main test
thread, and their logs will be written to a temporary directory.

For more information about how to run the tests, see
[doc/running_tests.md](doc/running_tests.md).

For more information about how to define a test, see
[tests/example](./tests/example) files.
"""

import argparse
import os
import subprocess
from collections import defaultdict
from threading import Thread
from queue import Queue
from time import time

from test_framework import FlorestaTestFramework

INFO_EMOJI = "ℹ️"
SUCCESS_EMOJI = "✅"
FAILURE_EMOJI = "❌"
ALLDONE_EMOJI = "🎉"

# Scripts that are run by default.
# Longest test should go first,
# to favor running tests in parallel.
# We use this like those in  the
# Bitcoin Core tests/functional/test_runner.py:89

# Tests that are ran by default. The longest running tests should be ran first,
# so parallelization is used most effectively. This structure is copied from
# Bitcoin Core's functional tests:
# https://github.com/bitcoin/bitcoin/blob/master/test/functional/test_runner.py#L89
BASE_TEST_SUITE = [
    ("floresta-cli", "addnode-v2"),
    ("floresta-cli", "addnode-v1"),
    ("florestad", "reorg-chain"),
    ("floresta-cli", "getbestblockhash"),
    ("floresta-cli", "getblockcount"),
    ("floresta-cli", "uptime"),
    ("florestad", "restart"),
    ("florestad", "connect"),
    ("floresta-cli", "stop"),
    ("floresta-cli", "ping"),
    ("floresta-cli", "getrpcinfo"),
    ("example", "integration"),
    ("example", "functional"),
    ("example", "electrum"),
    ("floresta-cli", "getblockhash"),
    ("florestad", "tls"),
    ("floresta-cli", "getroots"),
    ("floresta-cli", "getblock"),
    ("floresta-cli", "getmemoryinfo"),
    ("floresta-cli", "getblockheader"),
    ("floresta-cli", "getpeerinfo"),
    ("florestad", "tls-fail"),
    ("floresta-cli", "getblockchaininfo"),
    ("example", "bitcoin"),
    ("example", "utreexod"),
    ("example", "wallet"),
]

# Before running the tests, we check if the number of tests
# in the base test suite matches the number of tests in the
# `example`, `floresta-cli`, and `florestad` directories.
COUNT = 0
for testdir in ("example", "floresta-cli", "florestad"):
    dirname = os.path.abspath(os.path.dirname(__file__))
    tests_path = os.path.join(dirname, testdir)
    for test_name in os.listdir(tests_path):
        if test_name.endswith(".py"):
            COUNT += 1

# If the number of tests in the base test suite is not equal
# to these found in these directories, we raise an error because
# we forgot to add a test to the base suite.
if COUNT != len(BASE_TEST_SUITE):
    raise RuntimeError(
        f"Number of tests in the base test suite ({len(BASE_TEST_SUITE)})"
        f" does not match the number of tests found ({COUNT})."
        " Please update the BASE_TEST_SUITE variable in tests/test_runner.py."
    )


def list_test_suites(test_dir: str):
    """List all test suites inside ./tests"""
    print(f"{INFO_EMOJI} Available test suites:")
    print()

    organized_suites: dict[str, list[str]] = defaultdict(list)

    for folder, name in BASE_TEST_SUITE:
        fullpath = os.path.join(test_dir, folder, f"{name}.py")
        if os.path.exists(fullpath):
            organized_suites[folder].append(name)

    for folder in sorted(organized_suites.keys()):
        print(f"{folder}:")
        print(f"{'-' * len(folder)}")
        for name in sorted(organized_suites[folder]):
            print(f"    {name}")
        print()


def setup_test_suite(args: argparse.Namespace, test_dir: str) -> Queue:
    """
    Setup the test suite directories based on the provided arguments.
    If no test suite is specified, it defaults to the BASE_TEST_SUITE.
    """
    task_queue = Queue()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    if not args.test_suite:
        for folder, name in BASE_TEST_SUITE:
            full_path = os.path.join(test_dir, folder)
            if not os.path.exists(full_path):
                raise argparse.ArgumentError(
                    None, f"Test suite '{full_path}' does not exist"
                )

            if args.test_name and not any(name.startswith(n) for n in args.test_name):
                continue

            task_queue.put((full_path, name))

    else:
        for folder in args.test_suite:
            full_path = os.path.join(test_dir, folder)
            if not os.path.exists(full_path):
                raise argparse.ArgumentError(
                    None, f"Test suite '{full_path}' not found"
                )

            for file in os.listdir(full_path):
                if file.endswith(".py"):
                    if args.test_name and not any(
                        file.startswith(name) for name in args.test_name
                    ):
                        continue

                    name = file.split(".py")[0]
                    task_queue.put((full_path, name))

    return task_queue


def run_test_workers(task_queue: Queue, args: argparse.Namespace) -> list:
    """Run the tests in parallel"""
    results_queue = Queue()
    workers = []

    for _ in range(args.threads):
        worker = Thread(target=run_test_worker, args=(task_queue, results_queue, args))
        worker.start()
        workers.append(worker)

    for _ in workers:
        task_queue.put(None)

    for worker in workers:
        worker.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    return results


def run_test_worker(task_queue: Queue, results_queue: Queue, args: argparse.Namespace):
    """
    Worker function to run tests pulled from the task queue.
    Each test is run in a subprocess and logs output to a file.
    """
    while True:
        task = task_queue.get()
        if task is None:
            break

        test_suite_dir, name = task
        os.makedirs(args.log_dir, exist_ok=True)

        test_filename = os.path.normpath(os.path.join(test_suite_dir, f"{name}.py"))
        test_log_name = os.path.normpath(os.path.join(args.log_dir, f"{name}.log"))

        cli = ["python", test_filename]
        cli_msg = " ".join(cli)
        print(f"\n{INFO_EMOJI} Running '{cli_msg}'")

        start = time()
        with open(
            test_log_name, "wt", encoding="utf-8", buffering=args.log_buffer
        ) as log_file:

            # Avoid using 'with' for `subprocess.Popen` here, as we need the
            # process to start and stream output immediately for port detection
            # to work correctly. Using 'with' might delay output flushing,
            # which breaks log-based detection of random ports in tests.
            # pylint: disable=consider-using-with
            test = subprocess.Popen(cli, stdout=log_file, stderr=log_file)
            test.wait()

        results_queue.put(
            (test_filename, test.returncode == 0, test_log_name, start, time())
        )
        task_queue.task_done()


# pylint: disable=too-many-branches,line-too-long
def main():
    """
    Create a CLI called `test_runner` with calling arguments

    usage: test_runner [-h,-l] [-L DATA_DIR] [-t TEST_SUITE] [-k TEST_NAME] [-T THREADS] [-b BUFFER_SIZE]

    Functional test runner for Floresta.

    Options:
        -h, --help                    Show this help message and exit.
        -L, --log-dir DATA_DIR        Data directory for functional test logs.
        -t, --test-suite TEST_NAME    Test-suite directory to be tested. May be used more than once.
        -k, --test-name TEST_NAME     Test name to be tested in a suite. May be used more than once.
        -l, --list-suites             List all available test-suite directories.
        -T, --threads THREADS         Number of threads to run tests in parallel (default: 4).
        -b, --log-buffer BUFFER_SIZE  Changes the `io.DEFAULT_BUFFER_SIZE` for log files (default: 1024).
                                      Small values may cause issues with large logs.
    """
    # Define a global variable for the base log directory
    # so it can be used in the test framework. But if we just
    # want to list suites or want see the help, we don't need it.
    base_log_dir = os.path.normpath(
        os.path.join(FlorestaTestFramework.get_integration_test_dir(), "logs")
    )

    # Structure the CLI
    parser = argparse.ArgumentParser(prog="run_tests")
    parser.add_argument(
        "-L",
        "--log-dir",
        default=base_log_dir,
        help="Data directory for functional test logs.",
    )
    parser.add_argument(
        "-t",
        "--test-suite",
        action="append",
        default=None,
        help="Test-suite directory to be tested. May be used more than once.",
    )
    parser.add_argument(
        "-k",
        "--test-name",
        action="append",
        default=[],
        help="Test name to be tested in a suite. May be used more than once.",
    )
    parser.add_argument(
        "-l",
        "--list-suites",
        action="store_true",
        default=False,
        help="List all available test-suite directories.",
    )
    parser.add_argument(
        "-T",
        "--threads",
        type=int,
        default=4,
        help="Number of threads to run tests in parallel (default: 4).",
    )

    # See these links for more information:
    # https://docs.python.org/3/library/io.html#io.DEFAULT_BUFFER_SIZE
    # https://stackoverflow.com/questions/29712445/what-is-the-use-of-buffering-in-pythons-built-in-open-function
    parser.add_argument(
        "-b",
        "--log-buffer",
        type=int,
        default=1024,
        help="changes the `io.DEFAULT_BUFFER_SIZE` for log files (default: 1024). Small values may cause issues with large logs.",
    )

    args = parser.parse_args()

    args.log_dir = os.path.abspath(base_log_dir)

    test_dir = os.path.abspath(os.path.dirname(__file__))

    if args.list_suites:
        list_test_suites(test_dir)
        return

    task_queue = setup_test_suite(args, test_dir)
    results = run_test_workers(task_queue, args)

    passed = [(name, log, start, end) for (name, ok, log, start, end) in results if ok]
    failed = [
        (name, log, start, end) for (name, ok, log, start, end) in results if not ok
    ]

    print("\nTest Summary:")
    print(f"\n{len(passed)} test(s) passed:")
    for name, log, start, end in passed:
        print(f"\n  {SUCCESS_EMOJI} {name}: {log} (took {end - start:.2f}s)")

    if failed:
        print(f"\n{len(failed)} test(s) failed:")
        for name, log, start, end in failed:
            print(f"\n  {FAILURE_EMOJI} {name} failed: {log} (took {end - start:.2f}s)")
        raise SystemExit(
            f"\n{FAILURE_EMOJI} Some tests failed. Check the logs in {args.log_dir}."
        )
    print(f"\n{ALLDONE_EMOJI} ALL TESTS PASSED! GOOD JOB!")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT OR Apache-2.0

"""Utility helpers used by the test framework (paths, ports, TLS helpers)."""

import inspect
import itertools
import math
import os
import re
import socket
import subprocess
import time

from test_framework.constants import FLORESTA_TEMP_DIR
from test_framework.crypto.pkcs8 import (
    create_pkcs8_private_key,
    create_pkcs8_self_signed_certificate,
)

# Ports are handed out per xdist worker, see `get_worker_port_block`.
# The range stays below the ephemeral ports the OS hands out on its own
# (32768 on Linux, 49152 on macOS), so it can't clash with them either.
PORT_RANGE_START = 10000
PORTS_PER_WORKER = 500
_PORT_COUNTER = itertools.count()

SERVICE_FLAGS_BY_NAME = {
    "NETWORK": 1 << 0,
    "GETUTXO": 1 << 1,
    "BLOOM": 1 << 2,
    "WITNESS": 1 << 3,
    "COMPACT_FILTERS": 1 << 6,
    "NETWORK_LIMITED": 1 << 10,
    "P2P_V2": 1 << 11,
    "UTREEXO": 1 << 12,
    "UTREEXO_ARCHIVE": 1 << 13,
}

BITCOIND_TEST_FRAMEWORK_SERVICES = "0000000000000c09"
BITCOIND_TEST_FRAMEWORK_SERVICESNAMES = {
    "NETWORK",
    "WITNESS",
    "NETWORK_LIMITED",
    "P2P_V2",
}


class Utility:
    """
    A utility class for common functions used in the test framework.
    """

    @staticmethod
    def get_integration_test_dir():
        """
        Get path for florestad used in integration tests, generally set on
        $FLORESTA_TEMP_DIR/binaries
        """
        if FLORESTA_TEMP_DIR is None:
            raise RuntimeError(
                "FLORESTA_TEMP_DIR not set. "
                + " Please set it to the path of the integration test directory."
            )
        return FLORESTA_TEMP_DIR

    @staticmethod
    def get_git_describe():
        """
        Get the output of 'git describe --tags --always' command.
        """
        try:
            git_describe = subprocess.check_output(
                ["git", "describe", "--tags", "--always"], text=True
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to run 'git describe'. Run this at the Floresta directory."
            ) from exc

        return git_describe

    @staticmethod
    def get_log_path():
        """
        Get the path for the test log file,
        """
        log_dir = os.path.join(
            Utility.get_integration_test_dir(), "logs", Utility.get_git_describe()
        )
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @staticmethod
    def get_worker_port_block() -> int:
        """
        First port of the block reserved for this xdist worker.

        Two workers picking ports at random eventually pick the same one, and
        the loser gets a daemon that dies on startup, or worse, a `stop` meant
        for someone else's node. Giving each worker its own block makes that
        impossible, the same way Bitcoin Core's `PortSeed` does.
        """
        worker = os.getenv("PYTEST_XDIST_WORKER", "gw0")
        index = int(re.sub(r"\D", "", worker) or 0)

        return PORT_RANGE_START + index * PORTS_PER_WORKER

    @staticmethod
    def get_random_port():
        """
        Get a free port from this worker's block, in order.

        Ports are handed out sequentially so a port is only reused after the
        whole block has been, by which time the daemon that had it is long gone.
        """
        block = Utility.get_worker_port_block()

        for _ in range(PORTS_PER_WORKER):
            port = block + next(_PORT_COUNTER) % PORTS_PER_WORKER
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Something outside the test suite may still hold it
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port

        raise RuntimeError(f"No free port in [{block}, {block + PORTS_PER_WORKER})")

    @staticmethod
    def create_tls_key_cert() -> tuple[str, str]:
        """
        Create a PKCS#8 formatted private key and a self-signed certificate.
        These keys are intended to be used with florestad's --tls-key-path and --tls-cert-path
        options.
        """
        # If we're in CI, we need to use the
        # path to the integration test dir
        # tempfile will be used to get the proper
        # temp dir for the OS
        tls_rel_path = os.path.join(Utility.get_integration_test_dir(), "data", "tls")
        tls_path = os.path.normpath(os.path.abspath(tls_rel_path))

        # Create the folder if not exists
        os.makedirs(tls_path, exist_ok=True)

        # Create certificates
        pk_path, private_key = create_pkcs8_private_key(tls_path)

        cert_path = create_pkcs8_self_signed_certificate(
            tls_path, private_key, common_name="florestad", validity_days=365
        )

        return (pk_path, cert_path)


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check, in a single attempt, whether `host` accepts connections on `port`.

    Returns True when the connection succeeds, False otherwise. Pair it with
    `wait_until` when the port is expected to change state over time.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def florestad_has_feature(flag: str) -> bool:
    """Check whether the `florestad` under test exposes `flag`.

    Feature-gated flags are only compiled into the CLI when their cargo feature is
    enabled, so asking `--help` for one tells us whether this build has the feature.
    Returns False when the binary cannot be run, so callers skip instead of erroring.
    """
    try:
        daemon = os.path.join(
            Utility.get_integration_test_dir(), "binaries", "florestad"
        )
        result = subprocess.run(
            [daemon, "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return False

    return flag in (result.stdout + result.stderr)


def wait_until_helper_internal(
    predicate, *, timeout=60, lock=None, timeout_factor=1.0, check_interval=0.05
):
    """Sleep until the predicate resolves to be True.

    Warning: Note that this method is not recommended to be used in tests as it is
    not aware of the context of the test framework. Using the `wait_until()` members
    from `BitcoinTestFramework` or `P2PInterface` class ensures the timeout is
    properly scaled. Furthermore, `wait_until()` from `P2PInterface` class in
    `p2p.py` has a preset lock.
    """
    timeout = timeout * timeout_factor
    time_end = time.time() + timeout

    while time.time() < time_end:
        if lock:
            with lock:
                if predicate():
                    return
        else:
            if predicate():
                return
        time.sleep(check_interval)

    # Print the cause of the timeout
    predicate_source = "''''\n" + inspect.getsource(predicate) + "'''"
    print(f"wait_until() failed. Predicate: {predicate_source}")
    raise AssertionError(
        f"Predicate {predicate_source} not true after {timeout} seconds"
    )


def wait_until(predicate, timeout=30, interval=0.5, error_msg="Condition not met"):
    """
    Wait until a predicate returns True or timeout is reached.
    """
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return True
        time.sleep(interval)

    raise TimeoutError(f"{error_msg} after {timeout} seconds")


def assert_service_fields_consistent(peer_info):
    """
    Assert that getpeerinfo's services and servicesnames describe the same flags.
    """
    services = peer_info["services"]
    assert isinstance(services, str)
    assert len(services) == 16
    services_bits = int(services, 16)

    services_names = peer_info["servicesnames"]
    assert isinstance(services_names, list)
    assert all(isinstance(name, str) for name in services_names)

    services_names = set(services_names)
    assert services_names <= set(SERVICE_FLAGS_BY_NAME)

    for name, flag in SERVICE_FLAGS_BY_NAME.items():
        assert bool(services_bits & flag) == (name in services_names), name


def assert_bitcoind_service_fields(peer_info):
    """
    Assert getpeerinfo service fields for the Bitcoin Core test framework node.
    """
    assert_service_fields_consistent(peer_info)
    assert peer_info["services"] == BITCOIND_TEST_FRAMEWORK_SERVICES
    assert set(peer_info["servicesnames"]) == BITCOIND_TEST_FRAMEWORK_SERVICESNAMES


def compare_fields(
    candidate, reference, ignore_fields=None, float_tol=1e-8, unordered_fields=None
):
    """
    Recursively compare two data structures (dicts, lists, or scalars),
    ignoring specified fields.

    Note:
        The comparison is asymmetric. `reference` defines the required fields.
        For Floresta RPC tests, use Floresta as `candidate` and the reference
        node as `reference`.

        Lists are compared element-by-element in order (strict, positional),
        since most RPC lists are semantically ordered (e.g. `tx`, `vin`,
        `vout`). Fields whose lists have no meaningful order (e.g. sets of
        names) can be listed in `unordered_fields`; those lists are then
        matched one-to-one regardless of position, so duplicates, missing and
        extra items are still detected.
    """
    if ignore_fields is None:
        ignore_fields = set()
    elif isinstance(ignore_fields, list):
        ignore_fields = set(ignore_fields)

    if unordered_fields is None:
        unordered_fields = set()
    elif isinstance(unordered_fields, list):
        unordered_fields = set(unordered_fields)

    _compare_fields(
        candidate,
        reference,
        ignore_fields=ignore_fields,
        float_tol=float_tol,
        unordered_fields=unordered_fields,
        key=None,
    )


# pylint: disable=too-many-arguments too-many-positional-arguments
def _compare_fields(
    candidate, reference, ignore_fields, float_tol, unordered_fields, key
):
    """
    Recursive worker for `compare_fields`.

    `key` is the dict key this pair was reached through (None at the top level
    or for elements of a list); it decides whether a list value is compared
    unordered.
    """
    # float tolerance
    if isinstance(candidate, float) or isinstance(reference, float):
        assert math.isclose(candidate, reference, rel_tol=0.0, abs_tol=float_tol), (
            f"Float mismatch: candidate={candidate}, reference={reference}, "
            f"tolerance={float_tol}"
        )
        return

    # dict
    if isinstance(candidate, dict) and isinstance(reference, dict):
        for dict_key, ref_value in reference.items():
            if dict_key in ignore_fields:
                continue
            assert dict_key in candidate, f"Missing key in candidate: {dict_key}"
            _compare_fields(
                candidate[dict_key],
                ref_value,
                ignore_fields=ignore_fields,
                float_tol=float_tol,
                unordered_fields=unordered_fields,
                key=dict_key,
            )
        return

    # list
    if isinstance(candidate, list) and isinstance(reference, list):
        assert len(candidate) == len(
            reference
        ), f"List length mismatch: expected {len(reference)}, got {len(candidate)}"

        if key in unordered_fields:
            # Order-free matching: every reference item must match exactly one
            # candidate item, regardless of position, so duplicates, missing
            # and extra items are all detected.
            remaining = list(range(len(candidate)))
            for ref_item in reference:
                match_index = None
                for i in remaining:
                    try:
                        _compare_fields(
                            candidate[i],
                            ref_item,
                            ignore_fields=ignore_fields,
                            float_tol=float_tol,
                            unordered_fields=unordered_fields,
                            key=None,
                        )
                        match_index = i
                        break
                    except AssertionError:
                        continue
                assert (
                    match_index is not None
                ), f"Item {ref_item!r} not found in candidate list {candidate!r}"
                remaining.remove(match_index)
            return

        for cand_item, ref_item in zip(candidate, reference):
            _compare_fields(
                cand_item,
                ref_item,
                ignore_fields=ignore_fields,
                float_tol=float_tol,
                unordered_fields=unordered_fields,
                key=None,
            )
        return

    # scalar
    assert candidate == reference

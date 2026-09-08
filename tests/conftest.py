# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Pytest configuration and fixtures for node testing.

This module provides fixtures for creating and managing test nodes
(florestad, bitcoind, utreexod) in various configurations.
"""

# pylint: disable=redefined-outer-name

import logging
import os
import time
from typing import Callable, List

import pytest
from test_framework import FlorestaTestFramework
from test_framework.constants import (
    FLORESTA_TEMP_DIR,
    WALLET_ADDRESS,
    WALLET_DESCRIPTOR_EXTERNAL,
    WALLET_DESCRIPTOR_INTERNAL,
)
from test_framework.node import Node, NodeType
from test_framework.util import Utility


def pytest_addoption(parser):
    """Register custom pytest command-line options used by this test suite."""
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Run tests marked with the expensive marker",
    )


def pytest_collection_modifyitems(config, items):
    """Skip expensive-marked tests unless the dedicated opt-in flag is enabled."""
    if config.getoption("--run-expensive"):
        return

    skip_expensive = pytest.mark.skip(
        reason="need --run-expensive to run expensive tests"
    )
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)


@pytest.fixture(scope="session", autouse=True)
def validate_and_check_environment():
    """Validate environment and check for required binaries before running tests."""
    temp_dir = FLORESTA_TEMP_DIR
    if not temp_dir:
        pytest.fail("FLORESTA_TEMP_DIR environment variable not set")

    if not os.path.exists(temp_dir):
        pytest.fail(f"FLORESTA_TEMP_DIR directory does not exist: {temp_dir}")

    # Create necessary subdirectories
    os.makedirs(os.path.join(temp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "data"), exist_ok=True)

    # Check for required binaries
    binaries_dir = os.path.join(temp_dir, "binaries")
    binaries = {
        "florestad": os.path.join(binaries_dir, "florestad"),
        "utreexod": os.path.join(binaries_dir, "utreexod"),
        "bitcoind": os.path.join(binaries_dir, "bitcoind"),
    }

    for binary_name, binary_path in binaries.items():
        if not os.path.exists(binary_path):
            pytest.fail(f"{binary_name} binary not found at {binary_path}")


# pylint: disable=unused-argument
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook that captures the test result for use in fixtures.
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def _create_logger(test_name):
    """Create a logger with a file handler for the given test name.

    Shared helper used by both function-scoped and class-scoped logging
    fixtures to avoid duplicating the setup logic.
    """
    logger = logging.getLogger(test_name)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )

    log_path = Utility.get_log_path()
    log_file = os.path.join(log_path, f"{test_name}", f"{test_name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger, log_file


@pytest.fixture(scope="function")
def setup_logging(request):
    """
    Configure logging for the test, including the file and line number where the log was called.
    """
    test_name = request.node.name
    logger, log_file = _create_logger(test_name)

    yield logger

    # Capture test result and log it
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        logger.error("=" * 80)
        logger.error("TEST FAILED: %s", test_name)
        logger.error("=" * 80)
        logger.error("%s", request.node.rep_call.longrepr)
        logger.error("=" * 80)

        print(f"📋 Log file: {log_file}\n")

    # Clear handlers after the test
    logger.handlers.clear()


@pytest.fixture(scope="function")
def node_manager(setup_logging, request):
    """Provides a FlorestaTestFramework instance that automatically cleans up after each test"""
    manager = FlorestaTestFramework(logger=setup_logging, test_name=request.node.name)

    yield manager

    # Cleanup happens automatically after yield
    manager.stop()


@pytest.fixture
def florestad_node(node_manager) -> Node:
    """Single `florestad` node with default configurations, started and ready for testing"""
    node = node_manager.add_node_default_args(variant=NodeType.FLORESTAD)
    node_manager.run_node(node)
    return node


@pytest.fixture
def bitcoind_node(node_manager) -> Node:
    """Single `bitcoind` node with default configurations, started and ready for testing"""
    node = node_manager.add_node_default_args(variant=NodeType.BITCOIND)
    node_manager.run_node(node)
    return node


@pytest.fixture
def utreexod_node(node_manager) -> Node:
    """Single `utreexod` node with default configurations, started and ready for testing"""
    node = node_manager.add_node_extra_args(
        variant=NodeType.UTREEXOD,
        extra_args=[
            f"--miningaddr={WALLET_ADDRESS}",
            "--utreexoproofindex",
            "--prune=0",
        ],
    )
    node_manager.run_node(node)
    return node


@pytest.fixture
def florestad_utreexod(
    florestad_node, utreexod_node, node_manager
) -> tuple[Node, Node]:
    """
    Creates and starts a `florestad` node and a `utreexod` node.
    The nodes are automatically connected to each other and are ready for testing.
    """
    node_manager.connect_nodes(florestad_node, utreexod_node)

    return florestad_node, utreexod_node


@pytest.fixture
def florestad_bitcoind(
    florestad_node, bitcoind_node, node_manager
) -> tuple[Node, Node]:
    """
    Creates and starts a `florestad` node and a `bitcoind` node.
    The nodes are automatically connected to each other and are ready for testing.
    """
    node_manager.connect_nodes(florestad_node, bitcoind_node)

    return florestad_node, bitcoind_node


@pytest.fixture
def florestad_bitcoind_utreexod_with_chain(
    florestad_node, bitcoind_node, utreexod_node, node_manager
) -> Callable[..., tuple[Node, Node, Node]]:
    """
    Factory fixture that initializes a three-node network with a populated blockchain.

    Instantiates florestad, bitcoind, and utreexod nodes with pre-generated blocks and
    establishes mesh connectivity. Florestad loads wallet descriptors before chain sync,
    allowing it to track transactions during synchronization.
    """

    def _create_nodes_with_chain(
        blocks: int = 100,
        floresta_descriptors: List[str] | None = None,
        addr_coinbase: str | None = None,
    ) -> tuple[Node, Node, Node]:
        if floresta_descriptors is None:
            floresta_descriptors = [
                WALLET_DESCRIPTOR_EXTERNAL,
                WALLET_DESCRIPTOR_INTERNAL,
            ]

        for descriptor in floresta_descriptors:
            florestad_node.rpc.load_descriptor(descriptor)

        if addr_coinbase:
            bitcoind_node.rpc.generate_block_to_address(blocks, addr_coinbase)
        else:
            utreexod_node.rpc.generate(blocks)

        node_manager.connect_nodes(florestad_node, utreexod_node)
        time.sleep(3)
        node_manager.connect_nodes(bitcoind_node, utreexod_node)
        time.sleep(1)
        node_manager.connect_nodes(florestad_node, bitcoind_node)

        return florestad_node, bitcoind_node, utreexod_node

    return _create_nodes_with_chain


@pytest.fixture(scope="class")
def shared_florestad_bitcoind_utreexod_with_chain(
    shared_florestad_node,
    shared_bitcoind_node,
    shared_utreexod_node,
    shared_node_manager,
) -> Callable[..., tuple[Node, Node, Node]]:
    """
    Class-scoped variant of ``florestad_bitcoind_utreexod_with_chain``.

    Returns a factory that initializes a three-node network shared across
    every method in a test class.
    """

    def _create_nodes_with_chain(
        blocks: int = 100,
        floresta_descriptors: List[str] | None = None,
    ) -> tuple[Node, Node, Node]:
        if floresta_descriptors is None:
            floresta_descriptors = [
                WALLET_DESCRIPTOR_EXTERNAL,
                WALLET_DESCRIPTOR_INTERNAL,
            ]

        for descriptor in floresta_descriptors:
            shared_florestad_node.rpc.load_descriptor(descriptor)

        shared_utreexod_node.rpc.generate(blocks)

        shared_node_manager.connect_nodes(shared_florestad_node, shared_utreexod_node)
        time.sleep(3)
        shared_node_manager.connect_nodes(shared_bitcoind_node, shared_utreexod_node)
        time.sleep(1)
        shared_node_manager.connect_nodes(shared_florestad_node, shared_bitcoind_node)

        shared_node_manager.wait_for_sync_nodes(is_finished_ibd=False)

        return shared_florestad_node, shared_bitcoind_node, shared_utreexod_node

    return _create_nodes_with_chain


@pytest.fixture
def add_node_with_tls(node_manager):
    """Creates and starts a node with TLS enabled, based on the specified variant."""

    def _create_node(variant: NodeType) -> Node:
        if variant == NodeType.BITCOIND:
            raise ValueError("BITCOIND does not support TLS")

        node = node_manager.add_node_with_tls(
            variant=variant,
        )
        node_manager.run_node(node)
        return node

    return _create_node


@pytest.fixture(scope="class")
def shared_setup_logging(request):
    """Class-scoped logging fixture for tests that share a single node."""
    test_name = request.node.name
    logger, _log_file = _create_logger(test_name)

    yield logger

    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        logger.error("=" * 80)
        logger.error("TEST FAILED: %s", test_name)
        logger.error("=" * 80)

    logger.handlers.clear()


@pytest.fixture(scope="class")
def shared_node_manager(shared_setup_logging, request):
    """Class-scoped node manager that lives for the entire test class."""
    manager = FlorestaTestFramework(
        logger=shared_setup_logging, test_name=request.node.name
    )
    yield manager
    manager.stop()


@pytest.fixture(scope="class")
def shared_florestad_node(shared_node_manager) -> Node:
    """Single florestad node shared across all methods in a test class."""
    node = shared_node_manager.add_node_default_args(variant=NodeType.FLORESTAD)
    shared_node_manager.run_node(node)
    return node


@pytest.fixture(scope="class")
def shared_bitcoind_node(shared_node_manager) -> Node:
    """Single bitcoind node shared across all methods in a test class."""
    node = shared_node_manager.add_node_default_args(variant=NodeType.BITCOIND)
    shared_node_manager.run_node(node)
    return node


@pytest.fixture(scope="class")
def shared_utreexod_node(shared_node_manager) -> Node:
    """Single utreexod node shared across all methods in a test class."""
    node = shared_node_manager.add_node_extra_args(
        variant=NodeType.UTREEXOD,
        extra_args=[
            f"--miningaddr={WALLET_ADDRESS}",
            "--utreexoproofindex",
            "--prune=0",
        ],
    )
    shared_node_manager.run_node(node)
    return node


@pytest.fixture
def add_node_with_extra_args(node_manager):
    """
    Creates and starts a node with extra command-line arguments, based on the
    specified variant.
    """

    def _create_node(variant: NodeType, extra_args: list) -> Node:
        node = node_manager.add_node_extra_args(
            variant=variant,
            extra_args=extra_args,
        )
        node_manager.run_node(node)
        return node

    return _create_node

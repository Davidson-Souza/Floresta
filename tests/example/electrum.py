# SPDX-License-Identifier: MIT OR Apache-2.0

"""
electrum.py

This example demonstrates how to use the Floresta test framework to start a
Florestad node with an integrated Electrum server and exercise it from tests.

It shows:
- Using pytest fixtures provided in tests/conftest.py (for example `florestad_node`)
  to create, configure and teardown a node instance.
- How to call Electrum RPC methods via `node.electrum` and inspect the response.
"""

import pytest

EXPECTED_VERSION = ["Floresta 0.6.0", "1.4"]


@pytest.mark.example
def test_electrum(florestad_node):
    """
    This test demonstrates how to set up and run an Electrum client,
    and verifies that the Electrum server responds with the expected version.
    """
    rpc_response = florestad_node.electrum.get_version()

    assert rpc_response[0] == EXPECTED_VERSION[0]
    assert rpc_response[1] == EXPECTED_VERSION[1]

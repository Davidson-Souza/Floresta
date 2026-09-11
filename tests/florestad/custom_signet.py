# SPDX-License-Identifier: MIT OR Apache-2.0

"""Verify that Floresta follows Bitcoin Core on a custom signet."""

import pytest

from test_framework.node import NodeType
from test_framework.util import wait_until

CUSTOM_SIGNET_CHALLENGE = "51"  # OP_TRUE permits unsigned test blocks.
MINING_ADDRESS = "tb1q9d4zjf92nvd3zhg6cvyckzaqumk4zre26x02q9"
BLOCK_BATCHES = (1, 4, 8)


@pytest.mark.florestad
@pytest.mark.p2p
def test_custom_signet_consensus(node_manager):
    """Mine custom-signet blocks with Core and compare both validated chains."""
    bitcoind = node_manager.add_node_extra_args(
        variant=NodeType.BITCOIND,
        extra_args=[
            "-chain=signet",
            f"-signetchallenge={CUSTOM_SIGNET_CHALLENGE}",
            "-dnsseed=0",
            "-fixedseeds=0",
        ],
    )
    node_manager.run_node(bitcoind)

    florestad = node_manager.add_node_extra_args(
        variant=NodeType.FLORESTAD,
        extra_args=[
            "--network=signet",
            f"--signet-challenge={CUSTOM_SIGNET_CHALLENGE}",
            f"--connect={bitcoind.p2p_url}",
            "--no-assume-utreexo",
            "--no-backfill",
            "--no-cfilters",
        ],
    )
    node_manager.run_node(florestad)
    node_manager.wait_for_peers_connections(florestad, bitcoind)

    for block_count in BLOCK_BATCHES:
        bitcoind.rpc.generate_block_to_address(block_count, MINING_ADDRESS)
        core_height = bitcoind.rpc.get_block_count()
        wait_until(
            predicate=lambda expected_height=core_height: florestad.rpc.get_blockchain_info()[
                "blocks"
            ]
            == expected_height,
            timeout=120,
            error_msg=f"Floresta did not validate custom signet height {core_height}",
        )

        floresta_info = florestad.rpc.get_blockchain_info()

        assert floresta_info["blocks"] == core_height
        assert floresta_info["headers"] == core_height
        assert florestad.rpc.get_bestblockhash() == bitcoind.rpc.get_bestblockhash()

    for height in range(bitcoind.rpc.get_block_count() + 1):
        assert florestad.rpc.get_blockhash(height) == bitcoind.rpc.get_blockhash(height)

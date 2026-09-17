# SPDX-License-Identifier: MIT OR Apache-2.0

"""Compact-filter rescan integration test using utreexod."""

import pytest
from requests.exceptions import HTTPError

from test_framework.constants import WALLET_ADDRESS
from test_framework.node import NodeType
from test_framework.rpc.exceptions import JSONRPCError
from test_framework.util import wait_until

MINE_BLOCKS = 20
HISTORICAL_HEIGHT = 1


@pytest.mark.florestad
def test_compact_filter_rescan(add_node_with_extra_args, node_manager, setup_logging):
    """Find a historical output through filters served by utreexod."""
    log = setup_logging
    utreexod = add_node_with_extra_args(
        variant=NodeType.UTREEXOD,
        extra_args=[
            f"--miningaddr={WALLET_ADDRESS}",
            "--utreexoproofindex",
            "--prune=0",
            "--cfilters",
        ],
    )

    log.info("Mining blocks before Floresta starts")
    utreexod.rpc.generate(MINE_BLOCKS)
    block_hash = utreexod.rpc.get_blockhash(HISTORICAL_HEIGHT)
    coinbase_txid = utreexod.rpc.get_block(block_hash)["tx"][0]
    coinbase_output = utreexod.rpc.get_txout(coinbase_txid, 0, False)
    script = coinbase_output["scriptPubKey"]["hex"]

    florestad = add_node_with_extra_args(
        variant=NodeType.FLORESTAD,
        extra_args=[],
    )
    node_manager.connect_nodes(florestad, utreexod)
    node_manager.wait_for_sync_nodes()

    wait_until(
        lambda: any(
            "COMPACT_FILTERS" in peer["servicesnames"]
            for peer in florestad.rpc.get_peerinfo()
        ),
        error_msg="Floresta did not connect to a compact-filter peer",
    )

    with pytest.raises((HTTPError, JSONRPCError)):
        florestad.rpc.get_raw_transaction(coinbase_txid, 1)

    found_output = {}

    def rescan_finds_output():
        try:
            found_output["value"] = florestad.rpc.find_tx_out(
                coinbase_txid,
                0,
                script,
                HISTORICAL_HEIGHT,
            )
        except (HTTPError, JSONRPCError):
            return False
        return bool(found_output["value"])

    log.info("Rescanning historical blocks through BIP157/158 filters")
    wait_until(
        rescan_finds_output,
        timeout=60,
        error_msg="Floresta did not find the historical coinbase output",
    )

    transaction = florestad.rpc.get_raw_transaction(coinbase_txid, 1)
    assert transaction["txid"] == coinbase_txid

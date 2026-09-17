# SPDX-License-Identifier: MIT OR Apache-2.0

"""Historical rescans triggered by Electrum scriptpubkey endpoints."""

import pytest

from test_framework.node import NodeType
from test_framework.util import wait_until

MAX_MINE_BLOCKS = 100
MINE_BATCH_SIZE = 10
MINING_ADDRESSES = [
    "bcrt1q427ze5mrzqupzyfmqsx9gxh7xav538yk2j4cft",
    "bcrt1q4gfcga7jfjmm02zpvrh4ttc5k7lmnq2re52z2y",
    "bcrt1q3ml87jemlfvk7lq8gfs7pthvj5678ndnxnw9ch",
]
SCRIPT_ENDPOINTS = ["balance", "history", "subscribe"]


def _mine_distinct_scripts(utreexod):
    outputs = {}
    while len(outputs) < len(SCRIPT_ENDPOINTS):
        if utreexod.rpc.get_block_count() >= MAX_MINE_BLOCKS:
            raise AssertionError("utreexod did not mine to every configured address")

        for block_hash in utreexod.rpc.generate(MINE_BATCH_SIZE):
            coinbase_txid = utreexod.rpc.get_block(block_hash)["tx"][0]
            coinbase_output = utreexod.rpc.get_txout(coinbase_txid, 0, False)
            script = coinbase_output["scriptPubKey"]["hex"]
            outputs.setdefault(script, coinbase_txid)

    return list(outputs.items())


def _start_nodes_with_history(add_node_with_extra_args, node_manager):
    utreexod = add_node_with_extra_args(
        variant=NodeType.UTREEXOD,
        extra_args=[
            *(f"--miningaddr={address}" for address in MINING_ADDRESSES),
            "--utreexoproofindex",
            "--prune=0",
            "--cfilters",
        ],
    )
    historical_outputs = _mine_distinct_scripts(utreexod)

    florestad = add_node_with_extra_args(
        variant=NodeType.FLORESTAD,
        extra_args=[],
    )
    node_manager.connect_nodes(florestad, utreexod)
    node_manager.wait_for_sync_nodes()
    return florestad, historical_outputs


def _wait_for_rescan(electrum, script, endpoint, log):
    history = {}

    def rescan_finished():
        history["value"] = electrum.get_scriptpubkey_history(script)
        return bool(history["value"])

    log.info("Waiting for %s to trigger a historical rescan", endpoint)
    wait_until(
        rescan_finished,
        timeout=90,
        interval=2,
        error_msg=f"{endpoint} did not trigger an Electrum rescan",
    )
    return history["value"]


@pytest.mark.electrum
def test_scriptpubkey_endpoints_rescan(
    add_node_with_extra_args, node_manager, setup_logging
):
    """Each scriptpubkey endpoint discovers a historical transaction."""
    log = setup_logging
    florestad, historical_outputs = _start_nodes_with_history(
        add_node_with_extra_args, node_manager
    )
    wait_until(
        lambda: any(
            "COMPACT_FILTERS" in peer["servicesnames"]
            for peer in florestad.rpc.get_peerinfo()
        ),
        error_msg="Floresta did not connect to a compact-filter peer",
    )

    for endpoint, (script, coinbase_txid) in zip(
        SCRIPT_ENDPOINTS, historical_outputs, strict=True
    ):
        with pytest.raises(ValueError):
            florestad.electrum.get_transaction(coinbase_txid)

        trigger = {
            "balance": florestad.electrum.get_scriptpubkey_balance,
            "history": florestad.electrum.get_scriptpubkey_history,
            "subscribe": florestad.electrum.subscribe_scriptpubkey,
        }[endpoint]
        initial_result = trigger(script)
        if endpoint == "balance":
            assert initial_result == {"confirmed": 0, "unconfirmed": 0}
        else:
            assert initial_result is None

        history = _wait_for_rescan(florestad.electrum, script, endpoint, log)
        assert history[0]["tx_hash"] == coinbase_txid
        balance = florestad.electrum.get_scriptpubkey_balance(script)
        assert balance["confirmed"] > 0
        assert florestad.electrum.subscribe_scriptpubkey(script) is not None

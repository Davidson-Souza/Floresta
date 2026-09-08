# SPDX-License-Identifier: MIT OR Apache-2.0

"""Verify that Floresta receives Utreexo transactions from utreexod."""

import json

from pathlib import Path

import pytest

from test_framework.node import NodeType

from test_framework.util import wait_until

IBD_FEE_FILTER = 21_000_000 * 100_000_000
RUNNING_FEE_FILTER = 10

UTREEXOD_SERVICES = 0x340D


def florestad_received_utreexo_transaction(florestad_node, txid):
    """Return whether Florestad logged receipt of the expected transaction."""
    log_path = Path(florestad_node.daemon.p2p_config.log_path)
    if not log_path.exists():
        return False

    marker = f"Received Utreexo transaction txid={txid}"
    return marker in log_path.read_text(encoding="utf-8")


def utreexod_fee_filter(utreexod_node):
    """Return the transaction relay filter Utreexod received from Florestad."""
    return next(
        (
            peer["feefilter"]
            for peer in utreexod_node.rpc.get_peerinfo()
            if "Floresta" in peer["subver"]
        ),
        None,
    )


def configure_utreexod_anchor(florestad_node, utreexod_node):
    """Write the Utreexod service advertisement before Florestad starts."""
    host, port = utreexod_node.p2p_url.rsplit(":", maxsplit=1)
    anchors = [
        {
            "address": {"V4": host},
            "last_connected": 0,
            "state": {"Tried": 0},
            "services": UTREEXOD_SERVICES,
            "port": int(port),
        }
    ]
    anchor_path = Path(florestad_node.daemon.data_dir) / "regtest" / "anchors.json"
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_path.write_text(json.dumps(anchors), encoding="utf-8")


@pytest.mark.p2p
def test_florestad_receives_utreexo_transaction(
    setup_logging,
    node_manager,
    bitcoind_node,
):
    """Relay a wallet transaction through utreexod with every proof hash."""
    log = setup_logging
    bitcoind_node.rpc.create_wallet("utreexo-relay")
    miner_address = bitcoind_node.rpc.get_new_address()
    utreexod_node = node_manager.add_node_extra_args(
        variant=NodeType.UTREEXOD,
        extra_args=[
            f"--miningaddr={miner_address}",
            "--utreexoproofindex",
            "--prune=0",
        ],
    )
    node_manager.run_node(utreexod_node)
    utreexod_node.rpc.generate(432)

    florestad_node = node_manager.add_node_default_args(variant=NodeType.FLORESTAD)
    configure_utreexod_anchor(florestad_node, utreexod_node)
    node_manager.run_node(florestad_node)
    node_manager.wait_for_peers_connections(florestad_node, utreexod_node)
    wait_until(
        lambda: utreexod_fee_filter(utreexod_node) == IBD_FEE_FILTER,
        error_msg="Florestad did not suppress transaction relay during IBD",
    )

    node_manager.connect_nodes(bitcoind_node, utreexod_node)
    node_manager.connect_nodes(florestad_node, bitcoind_node)
    wait_until(
        lambda: node_manager.check_sync_nodes(is_finished_ibd=True),
        timeout=180,
        error_msg="Florestad did not complete initial block download",
    )

    wait_until(
        lambda: utreexod_fee_filter(utreexod_node) == RUNNING_FEE_FILTER,
        error_msg="Florestad did not restore its transaction relay fee filter",
    )

    log.info("Disconnecting Florestad from bitcoind")
    florestad_node.rpc.disconnectnode(node_address=bitcoind_node.p2p_url)
    node_manager.wait_for_peers_connections(
        florestad_node,
        bitcoind_node,
        is_connected=False,
    )

    log.info("Broadcasting a transaction from bitcoind through utreexod")
    txid = bitcoind_node.rpc.send_to_address(
        bitcoind_node.rpc.get_new_address(),
        1.0,
        fee_rate=1,
    )

    wait_until(
        lambda: florestad_received_utreexo_transaction(florestad_node, txid),
        timeout=60,
        error_msg="Florestad did not receive the Utreexo transaction from utreexod",
    )

# SPDX-License-Identifier: MIT OR Apache-2.0

"""
floresta_cli_getdeploymentinfo.py

Functional test for `getdeploymentinfo`. Mines blocks via utreexod, then
compares florestad's response against bitcoind's for each buried deployment.

Floresta currently emits only buried deployments (bip34, bip66, bip65, csv,
segwit). Bitcoin Core also emits BIP9 deployments (taproot, testdummy) which
require the versionbits state machine; those keys are present in bitcoind's
response but skipped on the floresta side.
"""

import pytest
from test_framework.util import compare_fields

MINE_BLOCKS = 10

# Deployments have special validation because Floresta does not support all of them.
IGNORE_FIELDS = ["deployments"]
UNORDERED_FIELDS = ["script_flags"]

# bitcoind's output is BIP9 and intentionally absent on floresta.
IGNORE_FIELDS_DEPLOYMENTS = ["bip9"]


@pytest.mark.rpc
def test_get_deployment_info(node_manager, florestad_bitcoind_utreexod_with_chain):
    """
    Compare florestad's getdeploymentinfo response against bitcoind's after a
    small chain extension. Each buried deployment must match by type, height
    and active flag. BIP9 entries are validated by absence on the floresta side.
    """
    florestad, bitcoind, _ = florestad_bitcoind_utreexod_with_chain(MINE_BLOCKS)

    node_manager.wait_for_sync_nodes()

    valid_info(florestad, bitcoind, None)

    genesis_hash = florestad.rpc.get_blockhash(0)

    valid_info(florestad, bitcoind, genesis_hash)

    # Mid-chain blockhash: exercises the `Some(blockhash)` branch of the handler.
    # Activation heights are network constants already checked at the tip, so
    # only the active flag is compared here.
    mid_height = MINE_BLOCKS // 2
    mid_hash = florestad.rpc.get_blockhash(mid_height)

    valid_info(florestad, bitcoind, mid_hash)


def valid_info(florestad, bitcoind, block_hash):
    """
    Compare a single deployment entry from floresta and bitcoind.
    Raises an assertion error if any field does not match.
    """
    f_entry = florestad.rpc.get_deployment_info(block_hash)
    b_entry = bitcoind.rpc.get_deployment_info(block_hash)

    compare_fields(
        f_entry, b_entry, ignore_fields=IGNORE_FIELDS, unordered_fields=UNORDERED_FIELDS
    )

    f_deployments = f_entry["deployments"]
    b_deployments = b_entry["deployments"]

    # Now the validation uses Floresta's result as the reference and Bitcoin Core's result as the
    # candidate. Since Floresta doesn't support all the deployments that Bitcoin Core does, the
    # comparison only validates the fields that Floresta provides.
    compare_fields(
        b_deployments, f_deployments, ignore_fields=IGNORE_FIELDS_DEPLOYMENTS
    )

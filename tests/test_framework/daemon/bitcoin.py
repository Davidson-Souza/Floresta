# SPDX-License-Identifier: MIT OR Apache-2.0

"""
test_framework.daemon.bitcoin.py

A test framework for testing bitcoind daemon in regtest mode.
"""

from typing import List

from test_framework.daemon.base import BaseDaemon
from test_framework.rpc import ConfigRPC
from test_framework.daemon import ConfigP2P
from test_framework.electrum import ConfigElectrum


class BitcoinDaemon(BaseDaemon):
    """
    Spawn a new Bitcoind process on background and run it on
    regtest mode for tests.
    """

    def get_cmd_network(self) -> List[str]:
        """
        Return the network configuration flags for the node.
        """
        if any(
            argument == "-chain"
            or argument.startswith("-chain=")
            or argument in ("-regtest", "-signet", "-testnet", "-testnet4")
            for argument in self.extra_args
        ):
            return []

        return [
            "-chain=regtest",
        ]

    def get_cmd_data_dir(self, data_dir: str) -> List[str]:
        """
        Return the data directory configuration flags for the node.
        """
        return [f"-datadir={data_dir}"]

    def get_cmd_rpc(self, config: ConfigRPC) -> List[str]:
        """
        Return the RPC configuration flags for the node.
        """
        if config.user is None or config.password is None:
            raise ValueError("RPC user and password must be set for bitcoind")
        return [
            f"-rpcuser={config.user}",
            f"-rpcpassword={config.password}",
            f"-rpcport={config.port}",
            f"-rpcbind={config.host}",
            "-rpcthreads=1",
        ]

    def get_cmd_p2p(self, config: ConfigP2P) -> List[str]:
        """
        Return the P2P configuration flags for the node.
        """
        return [f"-port={config.port}", f"-bind={config.host}"]

    def get_cmd_electrum(self, config: ConfigElectrum) -> List[str]:
        """
        Return empty list, because bitcoind doesn't support Electrum.
        """
        return []

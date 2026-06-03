# SPDX-License-Identifier: MIT OR Apache-2.0

"""
test_framework.daemon.floresta.py

A test framework for testing florestad daemon in regtest mode.
"""

from typing import List

from test_framework.daemon.base import BaseDaemon
from test_framework.rpc import ConfigRPC
from test_framework.daemon import ConfigP2P
from test_framework.electrum import ConfigElectrum


class FlorestaDaemon(BaseDaemon):
    """
    Spawn a new Florestad process on background and run it on
    regtest mode for tests.
    """

    def get_cmd_network(self) -> List[str]:
        """
        Return the network configuration flags for the node.
        """
        return [
            "--network=regtest",
        ]

    def get_cmd_data_dir(self, data_dir: str) -> List[str]:
        """
        Return the data directory configuration flags for the node.
        """
        return [f"--data-dir={data_dir}"]

    def get_cmd_rpc(self, config: ConfigRPC) -> List[str]:
        """
        Return the RPC configuration flags for the node.

        Credentials are emitted only when set so tests can opt out (e.g. to
        exercise cookie auth in `tests/florestad/rpc_auth.py`).
        """
        address = f"{config.host}:{config.port}"
        flags = [f"--rpc-address={address}"]
        if config.user is not None:
            flags.append(f"--rpc-user={config.user}")
        if config.password is not None:
            flags.append(f"--rpc-password={config.password}")
        return flags

    def get_cmd_p2p(self, config: ConfigP2P) -> List[str]:
        """
        Return empty list, because floresta doesn't listen for connections
        """
        return []

    def get_cmd_electrum(self, config: ConfigElectrum) -> List[str]:
        """
        Return the Electrum configuration flags for the node.
        """
        electrum_settings = []
        electrum_settings.append(f"--electrum-address={config.host}:{config.port}")
        if config.tls:
            electrum_settings.extend(
                [
                    "--enable-electrum-tls",
                    f"--tls-key-path={config.tls.key_file}",
                    f"--tls-cert-path={config.tls.cert_file}",
                    f"--electrum-address-tls={config.host}:{config.tls.port}",
                ]
            )

        return electrum_settings

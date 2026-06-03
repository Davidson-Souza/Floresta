# SPDX-License-Identifier: MIT OR Apache-2.0

"""
node.py

This file defines the `Node` class, which provides control and management functionality
for different types of nodes in the test framework. It encapsulates the behavior of nodes,
including their daemon processes, RPC interfaces, and configurations.
"""

from enum import Enum
from typing import List, Tuple, Optional

from test_framework.daemon import ConfigP2P
from test_framework.daemon.bitcoin import BitcoinDaemon
from test_framework.daemon.floresta import FlorestaDaemon
from test_framework.daemon.utreexo import UtreexoDaemon
from test_framework.rpc import ConfigRPC
from test_framework.rpc.bitcoin import BitcoinRPC
from test_framework.rpc.floresta import FlorestaRPC
from test_framework.rpc.utreexo import UtreexoRPC
from test_framework.electrum import ConfigElectrum, ConfigTls
from test_framework.electrum.client import ElectrumClient
from test_framework.util import Utility, wait_until


class NodeType(Enum):
    """
    Enum for different node types.
    """

    BITCOIND = "bitcoind"
    FLORESTAD = "florestad"
    UTREEXOD = "utreexod"


# pylint: disable=too-many-instance-attributes
class Node:
    """
    Represents a node in the test framework.

    This class encapsulates the behavior of a node, including its daemon process,
    RPC interface, and configuration.
    """

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        variant: NodeType,
        rpc_config: ConfigRPC,
        p2p_config: ConfigP2P,
        extra_args: List[str],
        electrum_config: ConfigElectrum,
        targetdir: str,
        data_dir: str,
        tls: bool,
        log,
    ):
        match variant:
            case NodeType.FLORESTAD:
                rpc = FlorestaRPC(config=rpc_config, log=log)
                daemon = FlorestaDaemon(
                    name=variant.value,
                    rpc_config=rpc_config,
                    p2p_config=p2p_config,
                    extra_args=extra_args,
                    electrum_config=electrum_config,
                    target=targetdir,
                    data_dir=data_dir,
                    log=log,
                )
            case NodeType.UTREEXOD:
                rpc = UtreexoRPC(config=rpc_config, log=log)
                daemon = UtreexoDaemon(
                    name=variant.value,
                    rpc_config=rpc_config,
                    p2p_config=p2p_config,
                    extra_args=extra_args,
                    electrum_config=electrum_config,
                    target=targetdir,
                    data_dir=data_dir,
                    log=log,
                )
            case NodeType.BITCOIND:
                rpc = BitcoinRPC(config=rpc_config, log=log)
                daemon = BitcoinDaemon(
                    name=variant.value,
                    rpc_config=rpc_config,
                    p2p_config=p2p_config,
                    electrum_config=electrum_config,
                    extra_args=extra_args,
                    target=targetdir,
                    data_dir=data_dir,
                    log=log,
                )
            case _:
                raise ValueError(
                    f"Unsupported variant: {variant}. Use 'florestad', 'utreexod' or 'bitcoind'."
                )

        if variant == NodeType.BITCOIND:
            electrum = None
        else:
            electrum = ElectrumClient(electrum_config, log=log)

        self.daemon = daemon
        self.rpc = rpc
        self.electrum = electrum
        self._tls = tls
        self._variant = variant
        self._static_values = True
        self._log = log
        self._log_path = p2p_config.log_path

    @classmethod
    def create_node_default_config(
        cls,
        variant: NodeType,
        extra_args: List[str],
        data_dir: str,
        targetdir: str,
        tls: bool,
        log,
        log_path: str,
    ) -> "Node":
        """
        Create a node with default arguments. this argument

        During initialization, the `static_values` attribute is set to False,
        allowing the node's arguments to be modified after creation.
        """
        config_rpc = cls.create_config_rpc_default()
        config_p2p = cls.create_config_p2p_default(log_path=log_path)
        config_electrum = cls.create_config_electrum_default(tls=tls)

        node = cls(
            variant=variant,
            p2p_config=config_p2p,
            rpc_config=config_rpc,
            extra_args=extra_args,
            electrum_config=config_electrum,
            data_dir=data_dir,
            targetdir=targetdir,
            tls=tls,
            log=log,
        )

        node.static_values = False

        return node

    @property
    def variant(self) -> NodeType:
        """
        Get the node variant.
        """
        return self._variant

    @property
    def p2p_url(self) -> str:
        """
        Get the P2P URL to connect to the node.
        """
        return self.daemon.p2p_url

    @property
    def static_values(self) -> bool:
        """
        Get the static values flag.
        """
        return self._static_values

    @static_values.setter
    def static_values(self, value: bool):
        """Setter for `static_values` property"""
        self._static_values = value

    def set_config_electrum(self, value: ConfigElectrum):
        """Setter for `config_electrum` property"""
        if self.static_values:
            raise ValueError("Cannot modify static config_electrum")

        self.daemon.set_electrum_config(value)
        if self.electrum is not None:
            self.electrum.set_config(value)

    def set_p2p_config(self, value: ConfigP2P):
        """Setter for `p2p_config` property"""
        if self.static_values:
            raise ValueError("Cannot modify static p2p_config")

        self.daemon.set_p2p_config(value)
        self._log_path = value.log_path

    def set_rpc_config(self, value: ConfigRPC):
        """Setter for `rpc_config` property"""
        if self.static_values:
            raise ValueError("Cannot modify static rpc_config")

        self.daemon.set_rpc_config(value)
        self.rpc.set_config(value)

    def set_extra_args(self, value: List[str]):
        """Setter for `extra_args` property"""
        if self.static_values:
            raise ValueError("Cannot modify static extra_args")

        self.daemon.set_extra_args(value)

    @staticmethod
    def create_config_rpc_default() -> ConfigRPC:
        """
        Create a default RPC configuration for a node.

        Generates a random port and sets default credentials.
        """
        return ConfigRPC(
            host="127.0.0.1",
            port=Utility.get_random_port(),
            user="test",
            password="test",
        )

    @staticmethod
    def create_config_p2p_default(log_path: str) -> ConfigP2P:
        """
        Create a default P2P configuration for nodes.
        The port is random.
        """
        return ConfigP2P(
            host="127.0.0.1", port=Utility.get_random_port(), log_path=log_path
        )

    @staticmethod
    def create_config_electrum_default(tls: bool) -> ConfigElectrum:
        """
        Create a default Electrum configuration for nodes.
        The port is random.
        """
        if tls:
            key, cert = Utility.create_tls_key_cert()
            config_tls = ConfigTls(
                cert_file=cert,
                key_file=key,
                port=Utility.get_random_port(),
            )
        else:
            config_tls = None

        return ConfigElectrum(
            host="127.0.0.1",
            port=Utility.get_random_port(),
            tls=config_tls,
        )

    def update_configs(self):
        """
        Update the node's configurations for RPC, P2P, and Electrum.

        This function sets new configurations for the node by using the default
        configuration creation methods
        """
        new_rpc_config = self.create_config_rpc_default()
        new_p2p_config = self.create_config_p2p_default(self._log_path)
        new_electrum_config = self.create_config_electrum_default(self._tls)

        # Apply the new configurations
        self.set_rpc_config(new_rpc_config)
        self.set_p2p_config(new_p2p_config)
        self.set_config_electrum(new_electrum_config)

    def start(self):
        """
        Start the node.
        """
        if self.daemon.is_running:
            raise RuntimeError(f"Node '{self.variant}' is already running.")

        self.daemon.start()
        self.rpc.wait_on_socket(opened=True)

        # An open port is not readiness: the daemon only answers RPC calls once
        # it is done loading, so poll it until it does.
        wait_until(
            self.rpc.is_responsive,
            timeout=self.rpc.TIMEOUT,
            interval=self.rpc.POLL_INTERVAL,
            error_msg=f"Node '{self.variant}' did not answer RPC calls",
        )

        # When starting Floresta for the first time, it is ideal to check
        # if the Electrum server is ready to receive requests.
        if self.variant == NodeType.FLORESTAD and self.static_values is not True:
            self.electrum.ping()

    def stop(self):
        """
        Stop the node.
        """
        response = None
        if self.daemon.is_running:
            try:
                response = self.rpc.stop()
            # pylint: disable=broad-exception-caught
            except Exception:
                self.daemon.process.terminate()

            self.daemon.process.wait()
            self.rpc.wait_on_socket(opened=False)

        return response

    def connect_node(
        self, node: "Node", method: str = "add", v2transport: bool = False
    ):
        """
        Connect to another node.
        """
        if not self.daemon.is_running or not node.daemon.is_running:
            raise ValueError(
                f"Peers must be running, {self.variant} is running: {self.daemon.is_running}, "
                f"{node.variant} is running: {node.daemon.is_running}"
            )

        if node.variant == NodeType.FLORESTAD:
            raise ValueError("The p2p port is not configurable in floresta")

        self.rpc.addnode(node.p2p_url, method, v2transport=v2transport)

    def connect_node_by_url(self, url, method: str = "add", v2transport: bool = True):
        """
        Connect to a node by URL.
        """
        if not self.daemon.is_running:
            raise ValueError(
                f"Node '{self.variant}' must be running to connect to a node."
            )

        self.rpc.addnode(url, method, v2transport=v2transport)

    def get_connection_info(self) -> Tuple[str, Optional[str]]:
        """
        Get the user agent and host for the current node.
        """
        address = (
            self.p2p_url
            if self.variant != NodeType.FLORESTAD
            else None  # The p2p port is not configurable in floresta
        )
        variants = {
            NodeType.FLORESTAD: ("Floresta", address),
            NodeType.UTREEXOD: ("utreexod", address),
            NodeType.BITCOIND: ("Satoshi", address),
        }

        if self.variant not in variants:
            raise ValueError(f"Unknown peer variant: {self.variant}")

        return variants[self.variant]

    def is_peer_connected(self, peer: "Node") -> bool:
        """
        Check if the given peer is connected to this node via RPC.
        """
        keys = {
            NodeType.FLORESTAD: ("user_agent", "address"),
            NodeType.BITCOIND: ("subver", "addr"),
            NodeType.UTREEXOD: ("subver", "addr"),
        }

        if self.variant not in keys:
            raise ValueError(f"Unknown peer variant: {self.variant}")

        user_agent_key, address_key = keys[self.variant]
        peers_info = self.rpc.get_peerinfo()
        user_agent, address = peer.get_connection_info()

        return any(
            user_agent in peer_info.get(user_agent_key)
            and (
                address == peer_info.get(address_key)
                # The p2p port is not configurable in floresta
                or address is None
                # Utreexo nodes use `addrlocal` instead of `addr` to show connection information.
                or self.variant == NodeType.UTREEXOD
            )
            for peer_info in peers_info
        )

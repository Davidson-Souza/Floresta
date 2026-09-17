# SPDX-License-Identifier: MIT OR Apache-2.0

"""
floresta_rpc.py

A test framework for testing JsonRPC calls to a floresta node.
"""

from test_framework.rpc.base import BaseRPC


class FlorestaRPC(BaseRPC):
    """
    A class for making RPC calls to a floresta node.
    """

    def get_jsonrpc_version(self) -> str:
        """
        Get the JSON-RPC version of the node
        """
        return "2.0"

    def get_roots(self):
        """
        Returns the roots of our current floresta state performing
        """
        return self.perform_request("getroots")

    def get_memoryinfo(self, mode: str):
        """
        Returns stats about our memory usage performing
        """
        if mode not in ("stats", "mallocinfo"):
            raise ValueError(f"Invalid getmemoryinfo mode: '{mode}'")

        return self.perform_request("getmemoryinfo", params=[mode])

    def load_descriptor(self, descriptor: str):
        """
        Load a script descriptor into the wallet.
        """
        return self.perform_request("loaddescriptor", params=[descriptor])

    def find_tx_out(self, txid: str, vout: int, script: str, height: int):
        """
        Find an output by rescanning compact filters from the given height.
        """
        return self.perform_request("findtxout", params=[txid, vout, script, height])

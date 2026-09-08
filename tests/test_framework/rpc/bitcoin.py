# SPDX-License-Identifier: MIT OR Apache-2.0

"""
tests.test_framework.rpc.bitcoin.py

A test framework for testing JsonRPC calls to a bitocoin node.
"""

from test_framework.rpc.base import BaseRPC


class BitcoinRPC(BaseRPC):
    """
    A class for making RPC calls to a bitcoin-core node.
    """

    def get_jsonrpc_version(self) -> str:
        """
        Get the JSON-RPC version of the node
        """
        return "1.0"

    def generate_block_to_address(self, nblocks: int, address: str) -> list:
        """
        Mine blocks immediately to a specific address by using

        Args:
            nblocks: The number of blocks to mine
            address: The address to mine the blocks to

        Returns:
            A list of block hashes of the newly mined blocks
        """
        return self.perform_request("generatetoaddress", params=[nblocks, address])

    def verify_txout_proof(self, proof: str) -> list:
        """
        Verify a Merkle proof returned by `gettxoutproof`.

        Args:
            proof: The hex-encoded proof string

        Returns:
            A list of transaction ids that the proof commits to
        """
        return self.perform_request("verifytxoutproof", params=[proof])

    def generate_block(self, nblocks: int) -> list:
        """
        Mine blocks immediately to a address(bcrt1q3ml87jemlfvk7lq8gfs7pthvj5678ndnxnw9ch) using
        `generate_block_to_address(nblocks, address)`

        Args:
            nblocks: The number of blocks to mine

        Returns:
            A list of block hashes of the newly mined blocks
        """
        address = "bcrt1q3ml87jemlfvk7lq8gfs7pthvj5678ndnxnw9ch"
        return self.generate_block_to_address(nblocks, address)

    def get_new_address(self) -> str:
        """
        Get a new address from the wallet.
        """
        return self.perform_request("getnewaddress", params=[])

    def generate_block_to_wallet(self, nblocks: int) -> list:
        """
        Mine blocks immediately, with coinbase reward sent to a new wallet address
        """
        address = self.get_new_address()
        return self.generate_block_to_address(nblocks, address)

    def create_wallet(self, wallet_name: str):
        """
        Create a new wallet with the given name.
        """
        return self.perform_request("createwallet", params=[wallet_name])

    def send_to_address(
        self,
        address: str,
        amount: float,
        fee_rate: float | None = None,
    ) -> str:
        """
        Send a specified amount to a given address.

        A fixed fee rate is expressed in sat/vB and avoids depending on fee
        estimation in short regtest chains.
        """
        params = [address, amount]
        if fee_rate is not None:
            params.extend([None, None, False, True, None, "unset", None, fee_rate])

        return self.perform_request("sendtoaddress", params=params)

# SPDX-License-Identifier: MIT OR Apache-2.0

"""
electrum_client.py

A small Electrum client that should be used to test our Electrum functionality.

This should be conformant to the ElectrumX specs.
More here: https://electrumx.readthedocs.io/en/latest/protocol-methods.html
"""

from test_framework.electrum.base import BaseClient


# pylint: disable=too-many-public-methods
class ElectrumClient(BaseClient):
    """
    Client to electrum protocol
    """

    def block_header(self, block_hash: str):
        """
        Return a block header, given its hash.
        """
        return self.request("blockchain.block.header", [block_hash])

    def block_headers(self, start_height: int, count: int):
        """
        Return a chunk of block headers from the main chain, starting at `start_height` and
        returning at most `count` headers.
        """
        return self.request("blockchain.block.headers", [start_height, count])

    def estimate_fee(self, target: int):
        """
        Return an estimation of fees to get your transaction confirmed in `target` blocks.
        """
        return self.request("blockchain.estimatefee", [target])

    def headers_subscribe(self, script_hash: str):
        """
        Subscribe for receiving new headers as blocks are mined.
        """
        return self.request("blockchain.headers.subscribe", [script_hash])

    def relay_fee(self):
        """
        Return the minimum fee a low-priority transaction must pay in order to be
        accepted to the daemon's memory pool.
        """
        return self.request("blockchain.relayfee", [])

    def get_balance(self, script_hash):
        """
        Return the confirmed and unconfirmed balances of a script hash.
        """
        return self.request("blockchain.scripthash.get_balance", [script_hash])

    def get_history(self, script_hash: str):
        """
        Return the confirmed and unconfirmed history of a script hash.
        """
        return self.request("blockchain.scripthash.get_history", [script_hash])

    def get_mempool(self, script_hash: str):
        """
        Return the unconfirmed transactions of a script hash.
        """
        return self.request("blockchain.scripthash.get_mempool", [script_hash])

    def list_unspent(self, script_hash: str):
        """
        Return an ordered list of UTXOs sent to a script hash.
        """
        return self.request("blockchain.scripthash.listunspent", [script_hash])

    def subscribe(self, script_hash: str):
        """
        Subscribe to a script hash to receive new transactions as notification.
        """
        return self.request("blockchain.scripthash.subscribe", [script_hash])

    def unsubscribe(self, script_hash: str):
        """
        Unsubscribe from a script hash, preventing future notifications if its status changes.
        """
        return self.request("blockchain.scripthash.unsubscribe", [script_hash])

    def get_scriptpubkey_balance(self, script: str):
        """Return the confirmed and unconfirmed balance of a script."""
        return self.request("blockchain.scriptpubkey.get_balance", [script])

    def get_scriptpubkey_history(self, script: str):
        """Return the confirmed and unconfirmed history of a script."""
        return self.request("blockchain.scriptpubkey.get_history", [script])

    def subscribe_scriptpubkey(self, script: str):
        """Subscribe to a script and trigger a historical rescan."""
        return self.request("blockchain.scriptpubkey.subscribe", [script])

    def broadcast(self, tx: str):
        """
        Broadcast a transaction to the network.
        """
        return self.request("blockchain.transaction.broadcast", [tx])

    def get_transaction(self, tx_id: str):
        """
        Return a raw transaction.
        """
        return self.request("blockchain.transaction.get", [tx_id])

    def get_merkle(self, tx_id: str, height: int):
        """
        Return the merkle branch to a confirmed transaction given its hash and height.
        """
        return self.request("blockchain.transaction.get_merkle", [tx_id, height])

    # blockchain.transaction.get_tsc_merkle not really useful

    def tx_id_from_pos(self, height: int, pos: int, merkle=True):
        """
        Return a transaction hash and optionally a merkle proof, given a block height
        and a position in the block.
        """
        return self.request("blockchain.transaction.id_from_pos", [height, pos, merkle])

    def get_fee_histogram(self) -> str:
        """
        Return a histogram of the fee rates paid by transactions in the memory pool,
        weighted by transaction size.
        """
        return self.request("mempool.get_fee_histogram", [])

    def add_peer(self, features: str) -> str:
        """
        A newly-started server uses this call to get itself into other servers'
        peers lists. It should not be used by wallet clients.
        """
        return self.request("server.add_peer", [features])

    def add_banner(self) -> str:
        """
        Return a banner to be shown in the Electrum console.
        """
        return self.request("server.banner", [])

    def get_donation_address(self) -> str:
        """
        Return a server donation address.
        """
        return self.request("server.donation_address", [])

    def get_server_features(self) -> str:
        """
        Return a list of features and services supported by the server.
        """
        return self.request("server.features", [])

    def peers_subscribe(self) -> str:
        """
        Return a list of peer servers. Despite the name this is not a subscription
        and the server must send no notifications.
        """
        return self.request("server.peers.subscribe", [])

    def ping(self) -> str:
        """
        Ping the server to ensure it is responding, and to keep the session alive.
        The server may disconnect clients that have sent no requests for roughly 10
        minutes.
        """
        return self.request("server.ping", [])

    def get_version(self) -> str:
        """
        Identify the client to the server and negotiate the protocol version.
        Only the first server.version() message is accepted.
        """
        return self.request("server.version", ["test-client", "1.2"])

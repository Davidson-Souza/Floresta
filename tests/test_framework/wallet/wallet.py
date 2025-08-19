import random
import hashlib

from test_framework import FlorestaRPC
from test_framework.key import generate_privkey, ECKey
from test_framework.primitives.address import Bech32Address
from test_framework.primitives.transaction import (
    Script,
    Transaction,
    TxOutput,
    TxInput,
    OutPoint,
)


class MiniWallet:
    def __init__(self):
        self.balance = 0
        self.transactions = []
        self.utxo = []
        self.scripts = set()
        self.keypool = dict()

    def hash160(self, data: bytes) -> bytes:
        """Returns the RIPEMD160(SHA256(data)) hash."""
        sha256 = hashlib.sha256(data).digest()
        ripemd160 = hashlib.new("ripemd160")
        ripemd160.update(sha256)
        return ripemd160.digest()

    def create_address(self) -> Bech32Address:
        key = generate_privkey()
        pubkey = ECKey()
        pubkey.set(key, compressed=True)
        pubkey_hash = self.hash160(pubkey.get_pubkey().get_bytes())
        script = Script(bytes([0x00, 0x14]) + pubkey_hash)  # P2PKH script
        address = Bech32Address(script)

        self.keypool[address] = key
        self.scripts.add(script)

        return address

    def rescan_for_funds(self, rpc: FlorestaRPC):
        """
        Uses floresta-cli to ask for blocks since the genesis block,
        then finds transactions that match the wallet's addresses.
        """
        blocks = rpc.get_block_count()
        for block in range(blocks + 1):
            block_hash = rpc.get_blockhash(block)
            block_data = rpc.get_block(block_hash)
            for tx in block_data["tx"]:
                self.filter_transaction(tx)

    def filter_transaction(self, tx: dict):
        """
        Filters transactions to only include those that match the wallet's addresses.
        """
        for output in tx["vout"]:
            script = Script(output["scriptPubKey"]["hex"])
            if script in self.scripts:
                amount = output["value"]
                tx_output = TxOutput(script, amount)
                self.add_transaction(tx_output)

        for input in tx["vin"]:
            if "txid" in input and "vout" in input:
                outpoint = OutPoint(input["txid"], input["vout"])
                tx_input = TxInput(outpoint, Script(input["scriptSig"]["hex"]))
                self.process_spend(tx_input)

    def process_spend(self, tx):
        """
        Processes a transaction that spends UTXOs from the wallet.
        """
        for input in tx.inputs:
            for utxo in self.utxo:
                if (
                    utxo["txid"] == input.outpoint.txid
                    and utxo["index"] == input.outpoint.index
                ):
                    self.utxo.remove(utxo)
                    self.balance -= utxo["amount"]
                    break

    def add_transaction(self, tx):
        self.transactions.append(tx)
        for output in tx.outputs:
            if output.script.script in self.addresses:
                self.add_utxo(tx, output)

    def add_utxo(self, tx, output):
        utxo = {
            "txid": tx.txid,
            "index": tx.outputs.index(output),
            "amount": output.amount.value,
            "script": output.script.script,
        }
        self.utxo.append(utxo)
        self.balance += output.amount.value

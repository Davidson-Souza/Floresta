from test_framework import (
    FlorestaTestFramework,
    FlorestaRPC,
    FlorestaDaemon,
    UtreexoDaemon,
    UtreexoRPC,
)
from test_framework.wallet.wallet import MiniWallet


class MiniWalletTest(FlorestaTestFramework):
    """
    Tests for the MiniWallet class.
    """

    def set_test_params(self):
        """
        Here we define setup for test adding a node definition
        """
        self.wallet = MiniWallet()
        mining_addr = self.wallet.create_address()
        self.log(f"Mining address: {mining_addr}")
        self.utreexod = self.add_node(
            variant="utreexod", extra_args=[f"--miningaddr={mining_addr}"]
        )
        self.florestad = self.add_node(variant="florestad")

    def run_test(self):
        """
        Here we define the test itself:
        - create a new address
        - rescan for funds
        - check if the address is in the keypool
        """
        self.run_node(self.florestad)
        self.run_node(self.utreexod, timeout=10)

        address = self.wallet.create_address()
        self.assertIn(address, self.wallet.keypool)

        # Simulate rescanning for funds
        self.wallet.rescan_for_funds(self.florestad.rpc)

        # Check if the wallet's scripts contain the created address script
        script = bytes([0x00, 0x14]) + self.wallet.hash160(address.encode())
        self.assertIn(script, self.wallet.scripts)

        self.stop()


if __name__ == "__main__":
    MiniWalletTest().main()

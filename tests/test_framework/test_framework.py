import subprocess

from threading import Thread

from .mock_rpc import MockUtreexod
from .floresta_rpc import FlorestaRPC

class TestFramework:
    tests = []
    nodes: list() = []
    rpc: MockUtreexod = None

    def run_node(self, datadir: str, net: str):
        node = subprocess.Popen([
            "cargo",
            "run",
            "--features",
            "json-rpc",
            "--bin",
            "florestad",
            "--",
            "--network",
            net
        ])
        self.nodes.append(FlorestaRPC(node))

    def wait_for_rpc_connection(self):
        for node in self.nodes:
            node.wait_for_rpc_connection()

    def run_rpc(self):
        # Run as a thread
        self.rpc = Thread(target=MockUtreexod().run)
        self.rpc.daemon = True
        self.rpc.start()

    def stop_node(self, idx: int):
        self.nodes[idx].kill()
        self.nodes[idx].wait_to_stop()

    # Should be overrided by individual tests
    def run_test(self):
        raise NotImplemented

    def stop(self):
        for node in self.nodes:
            node.kill()
            node.wait_to_stop()

    def main(self):
        self.run_test()
        self.stop()

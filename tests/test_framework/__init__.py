"""
tests/test_framework/__init__.py

Adapted from
https://github.com/bitcoin/bitcoin/blob/master/test/functional/test_framework/test_framework.py

Bitcoin Core's functional tests define a metaclass that checks whether the required
methods are defined or not. Floresta's functional tests will follow this battle tested structure.
The difference is that `florestad` will run under a `cargo run` subprocess, which is defined at
`add_node_settings`.
"""

import os
import re
import sys
import copy
import time
import random
import socket
import signal
import contextlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Pattern, TextIO

from test_framework.crypto.pkcs8 import (
    create_pkcs8_private_key,
    create_pkcs8_self_signed_certificate,
)
from test_framework.daemon.bitcoin import BitcoinDaemon
from test_framework.daemon.floresta import FlorestaDaemon
from test_framework.daemon.utreexo import UtreexoDaemon
from test_framework.rpc.bitcoin import BitcoinRPC
from test_framework.rpc.floresta import FlorestaRPC
from test_framework.rpc.utreexo import UtreexoRPC
from test_framework.rpc.bitcoin import REGTEST_RPC_SERVER as bitcoind_rpc_server
from test_framework.rpc.floresta import REGTEST_RPC_SERVER as florestad_rpc_server
from test_framework.rpc.utreexo import REGTEST_RPC_SERVER as utreexod_rpc_server
from test_framework.primitives.transaction import (
    TxInput,
    TxOutput,
    OutPoint,
    Script,
    Amount,
    Transaction,
)
from test_framework.primitives.address import Bech32Address


class Node:
    """
    A node object to be used in the test framework.
    It contains the `daemon`, `rpc` and `rpc_config` objects.
    """

    def __init__(self, daemon, rpc, rpc_config, variant):
        self.daemon = daemon
        self.rpc = rpc
        self.rpc_config = rpc_config
        self.variant = variant

    def start(self):
        """
        Start the node.
        """
        if self.daemon.is_running:
            raise RuntimeError(f"Node '{self.variant}' is already running.")
        self.daemon.start()
        self.rpc.wait_for_connections(opened=True)

    def stop(self):
        """
        Stop the node.
        """
        if self.daemon.is_running:
            response = self.rpc.stop()
            self.rpc.wait_for_connections(opened=False)
            self.daemon.process.wait()
            return response
        return None

    def get_host(self) -> str:
        """
        Get the host address of the node.
        """
        return self.rpc_config["host"]

    def get_ports(self) -> int:
        """Get all ports of the node."""
        return self.rpc_config["ports"]

    def get_port(self, port_type: str) -> int:
        """
        Get the port of the node based on the port type.
        This is a convenience method for `get_ports`.
        """
        if port_type not in self.rpc_config["ports"]:
            raise ValueError(
                f"Port type '{port_type}' not found in node ports: {self.rpc_config['ports']}"
            )
        return self.rpc_config["ports"][port_type]

    def send_kill_signal(self, sigcode="SIGTERM"):
        """Send a signal to kill the daemon process."""
        with contextlib.suppress(ProcessLookupError):
            pid = self.daemon.process.pid
            os.kill(pid, getattr(signal, sigcode, signal.SIGTERM))


class FlorestaTestMetaClass(type):
    """
    Metaclass for FlorestaTestFramework.

    This metaclass ensures that any subclass of `FlorestaTestFramework`
    adheres to a standard whereby the subclass overrides `set_test_params` and
    `run_test, but DOES NOT override `__init__` or `main`. If those standards
    are violated, a `TypeError` is raised.
    """

    def __new__(mcs, clsname, bases, dct):
        if not clsname == "FlorestaTestFramework":
            if not ("run_test" in dct and "set_test_params" in dct):
                raise TypeError(
                    "FlorestaTestFramework subclasses must override 'run_test'"
                    "and 'set_test_params'"
                )

            if "__init__" in dct or "main" in dct:
                raise TypeError(
                    "FlorestaTestFramework subclasses may not override "
                    "'__init__' or 'main'"
                )

        return super().__new__(mcs, clsname, bases, dct)


# pylint: disable=too-many-public-methods
class FlorestaTestFramework(metaclass=FlorestaTestMetaClass):
    """
    Base class for a floresta test script. Individual floresta
    test scripts should:

    - subclass FlorestaTestFramework;
    - not override the __init__() method;
    - not override the main() method;
    - implement set_test_params();
    - implement run_test();


    The `set_test_params` method is called before the test starts
    and aims to configure the node variant, its daemon parameters
    or whatever you think should be defined. It is a good practice
    to set the number of nodes and their configuration in this method
    with `self.add_node`.

    The `run_test` method is the test itself, where one (or more) node(s)
    are started with the `self.run_node` method. This method will return
    a index integer for a `Node` object stored in a `self.nodes` property,
    each node containing the initialized `daemon` process, a `rpc` and
    `rpc_config` objects. The `rpc` object can be a `FlorestaRPC` or
    `UtreexoRPC` object, depending on the node variant defined.

    When a node start, it will wait for ALL node's socket ports to be opened.
    Inversely, the method `self.stop_node` will wait for ALL node's ports to
    be closed (you could also use `self.stop` to stop all nodes). Internally,
    it uses `node.rpc.wait_for_connections(opened=True)` to wait for all ports
    to be opened, or `node.rpc.wait_for_connections(opened=False)` to wait for
    all ports to be closed. You could use them if you want more control.

    Also, the `self.run_test` method is where you should call for assertions
    like `self.assertIsNone`, `self.assertIsSome`, `self.assertEqual`,
    `self.assertIn`, `self.assertMatch`, `self.assertTrue` and
    `self.assertRaises`. If the assertion passes, the test will continue.
    If it fails, the test will stop all nodes and raise an `AssertionError`.

    The `self.assertRaises` method is a special case. It should be used in a
    context manager, i.e., the `with self.assertRaises(<SomeException>)`
    clause. The context will expect for some exception to be raised and,
    if it raises, the script will continue. If it does not raise, it will stop
    all nodes and raise an `AssertionError`.

    In both methods, you can use `self.log` to log messages.

    At the end of file, you should execute `MyTest().main()` method.

    For more details, see the tests/example/*.py file to see how
    the Floresta team thought the test framework should be used and
    test/test_framework/{crypto,daemon,rpc,electrum}/*.py to see
    how the test framework was structured.
    """

    class _AssertRaisesContext:
        """
        Context manager for testing that an exception is raised.

        This keeps the assertRaises functionality neatly contained within our test framework
        """

        def __init__(self, test_framework, expected_exception):
            """Initialize the context manager with the expected exception type."""
            self.test_framework = test_framework
            self.expected_exception = expected_exception
            self.exception = None

        def __enter__(self):
            """Enter the context manager."""
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            """Exit the context manager and check if the expected exception was raised."""
            if exc_type is None:
                self.test_framework.stop_all_nodes()
                trace = traceback.format_exc()
                message = f"{self.expected_exception} was not raised"
                raise AssertionError(f"{message}: {trace}")

            if not issubclass(exc_type, self.expected_exception):
                trace = traceback.format_exc()
                message = f"Expected {self.expected_exception} but got {exc_type}"
                raise AssertionError(f"{message}: {trace}")

            self.exception = exc_value
            return True

    def __init__(self):
        """
        Sets test framework defaults.

        Do not override this method. Instead, override the set_test_params() method
        """
        self._nodes = []

    # pylint: disable=R0801
    def log(self, msg: str):
        """Log a message with the class caller"""

        now = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
        print(f"[{self.__class__.__name__} {now}] {msg}")

    def main(self):
        """
        Main function.

        This should not be overridden by the subclass test scripts.
        """
        try:
            self.set_test_params()
            self.run_test()
        except Exception as err:
            processes = []
            for node in self._nodes:

                # If the node has an RPC server, stop it gracefully
                # otherwise (maybe the error occurred before the RPC server
                # is started), try to kill the process with SIGTERM. If that
                # fails, try to force kill it with SIGKILL.
                processes.append(str(node.daemon.process.pid))
                if getattr(node, "rpc", None):
                    node.rpc.stop()
                    node.rpc.wait_for_connections(opened=False)
                else:
                    # pylint: disable=broad-exception-caught
                    try:
                        node.send_kill_signal("SIGTERM")
                    except Exception:
                        node.send_kill_signal("SIGKILL")

            raise RuntimeError(
                f"Process with pids {', '.join(processes)} failed to start: {err}"
            ) from err

    # Should be overridden by individual tests
    def set_test_params(self):
        """
        Tests must override this method to change default values for number of nodes, topology, etc
        """
        raise NotImplementedError

    def run_test(self):
        """
        Tests must override this method to run nodes, etc.
        """
        raise NotImplementedError

    @staticmethod
    def get_integration_test_dir():
        """
        Get path for florestad used in integration tests, generally set on
        $FLORESTA_TEMP_DIR/binaries
        """
        if os.getenv("FLORESTA_TEMP_DIR") is None:
            raise RuntimeError(
                "FLORESTA_TEMP_DIR not set. "
                + " Please set it to the path of the integration test directory."
            )
        return os.getenv("FLORESTA_TEMP_DIR")

    @staticmethod
    def create_data_dirs(data_dir: str, base_name: str, nodes: int) -> list[str]:
        """
        Create the data directories for any nodes to be used in the test.
        """
        paths = []
        for i in range(nodes):
            p = os.path.join(data_dir, "data", base_name, f"node-{i}")
            os.makedirs(p, exist_ok=True)
            paths.append(p)

        return paths

    @staticmethod
    def get_available_random_port(start: int, end: int = 65535):
        """Get an available random port in the range [start, end]"""
        while True:
            port = random.randint(start, end)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Check if the port is available
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port

    def get_test_log_path(self) -> str:
        """
        Get the path for the test name log file, which is the class name in lowercase.
        This is used to create a log file for the test.
        """
        tempdir = str(FlorestaTestFramework.get_integration_test_dir())

        # Get the class's base filename
        filename = sys.modules[self.__class__.__module__].__file__
        filename = os.path.basename(filename)
        filename = filename.replace(".py", "")

        return os.path.join(tempdir, "logs", f"{filename}.log")

    def create_tls_key_cert(self) -> tuple[str, str]:
        """
        Create a PKCS#8 formatted private key and a self-signed certificate.
        These keys are intended to be used with florestad's --tls-key-path and --tls-cert-path
        options.
        """
        # If we're in CI, we need to use the
        # path to the integration test dir
        # tempfile will be used to get the proper
        # temp dir for the OS
        tls_rel_path = os.path.join(
            FlorestaTestFramework.get_integration_test_dir(), "data", "tls"
        )
        tls_path = os.path.normpath(os.path.abspath(tls_rel_path))

        # Create the folder if not exists
        os.makedirs(tls_path, exist_ok=True)

        # Create certificates
        pk_path, private_key = create_pkcs8_private_key(tls_path)
        self.log(f"Created PKCS#8 key at {pk_path}")

        cert_path = create_pkcs8_self_signed_certificate(
            tls_path, private_key, common_name="florestad", validity_days=365
        )
        self.log(f"Created self-signed certificate at {cert_path}")

        return (pk_path, cert_path)

    def is_option_set(self, extra_args: list[str], option: str) -> bool:
        """
        Check if an option is set in extra_args
        """
        for arg in extra_args:
            if arg.startswith(option):
                return True
        return False

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_data_dir_for_daemon(
        self,
        data_dir_arg: str,
        default_args: list[str],
        extra_args: list[str],
        tempdir: str,
        testname: str,
    ):
        """
        Create a data directory for the daemon to be run.
        """
        # Add a default data-dir if not set
        if not self.is_option_set(extra_args, data_dir_arg):
            datadir = os.path.normpath(os.path.join(tempdir, "data", testname))
            default_args.append(f"{data_dir_arg}={datadir}")

        else:
            data_dir_arg = next(
                (arg for arg in extra_args if arg.startswith(f"{data_dir_arg}="))
            )
            datadir = data_dir_arg.split("=", 1)[1]

        if not os.path.exists(datadir):
            self.log(f"Creating data directory for {data_dir_arg} in {datadir}")
            os.makedirs(datadir, exist_ok=True)

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    def setup_florestad_daemon(
        self,
        targetdir: str,
        tempdir: str,
        testname: str,
        extra_args: List[str],
        tls: bool,
    ) -> FlorestaDaemon:
        """Add default args to a florestad node settings to be run."""
        daemon = FlorestaDaemon()
        daemon.create(target=targetdir)
        default_args = []

        # Add a default data-dir if not set
        self.create_data_dir_for_daemon(
            "--data-dir", default_args, extra_args, tempdir, testname
        )

        # Add a random rpc address if not set
        if not self.is_option_set(extra_args, "--rpc-address"):
            port = FlorestaTestFramework.get_available_random_port(18443, 19443)
            default_args.append(f"--rpc-address=127.0.0.1:{port}")

        # Add a random electrum address if not set
        if not self.is_option_set(extra_args, "--electrum-address"):
            electrum_port = FlorestaTestFramework.get_available_random_port(
                20001, 21001
            )
            default_args.append(f"--electrum-address=127.0.0.1:{electrum_port}")

        # configure (or not) the ssl keys
        if tls:
            key, cert = self.create_tls_key_cert()
            default_args.append("--enable-electrum-tls")
            default_args.append(f"--tls-key-path={key}")
            default_args.append(f"--tls-cert-path={cert}")

            # Add a random tls electrum address if not set
            if not self.is_option_set(extra_args, "--electrum-address-tls"):
                tls_electrum_port = FlorestaTestFramework.get_available_random_port(
                    20002, 21002
                )
                default_args.append(
                    f"--electrum-address-tls=127.0.0.1:{tls_electrum_port}"
                )

        daemon.add_daemon_settings(default_args)
        daemon.add_daemon_settings(extra_args)
        return daemon

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def setup_utreexod_daemon(
        self,
        targetdir: str,
        tempdir: str,
        testname: str,
        extra_args: List[str],
        tls: bool,
    ):
        """Add default args to a utreexod node settings to be run."""
        daemon = UtreexoDaemon()
        daemon.create(target=targetdir)
        default_args = []

        # Add a default data-dir if not set
        self.create_data_dir_for_daemon(
            "--datadir", default_args, extra_args, tempdir, testname
        )

        # Add a default p2p listen address if not set
        if not self.is_option_set(extra_args, "--listen"):
            port = FlorestaTestFramework.get_available_random_port(18444, 19444)
            default_args.append(f"--listen=127.0.0.1:{port}")

        # Add a default rpc listen address if not set
        if not self.is_option_set(extra_args, "--rpclisten"):
            port = FlorestaTestFramework.get_available_random_port(18443, 19443)
            default_args.append(f"--rpclisten=127.0.0.1:{port}")

        if not self.is_option_set(extra_args, "--electrumlisteners"):
            # Add a default electrum address if not set
            electrum_port = FlorestaTestFramework.get_available_random_port(
                20001, 21001
            )
            default_args.append(f"--electrumlisteners=127.0.0.1:{electrum_port}")

        # configure (or not) the ssl keys
        if not tls:
            default_args.append("--notls")
        else:
            key, cert = self.create_tls_key_cert()
            default_args.append(f"--rpckey={key}")
            default_args.append(f"--rpccert={cert}")

            # Add a random tls electrum address if not set
            if not self.is_option_set(extra_args, "--tlselectrumlisteners"):
                tls_electrum_port = FlorestaTestFramework.get_available_random_port(
                    20002, 21002
                )
                default_args.append(f"--tlselectrumlisteners={tls_electrum_port}")

        daemon.add_daemon_settings(default_args)
        daemon.add_daemon_settings(extra_args)
        return daemon

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def setup_bitcoind_daemon(
        self,
        targetdir: str,
        tempdir: str,
        testname: str,
        extra_args: List[str],
    ) -> BitcoinDaemon:
        """Add default args to a bitcoind node settings to be run."""
        daemon = BitcoinDaemon()
        daemon.create(target=targetdir)
        default_args = []

        # Add a default data-dir if not set
        self.create_data_dir_for_daemon(
            "-datadir", default_args, extra_args, tempdir, testname
        )

        if not self.is_option_set(extra_args, "-bind"):
            # Add a default rpc bind address if not set
            port = FlorestaTestFramework.get_available_random_port(18445, 19445)
            default_args.append(f"-bind=127.0.0.1:{port}")

        if not self.is_option_set(extra_args, "-rpcbind"):
            # Add a default rpc bind address if not set
            port = FlorestaTestFramework.get_available_random_port(20443, 21443)

            # option -rpcbind is ignored if -rpcallowip isnt specified,
            # refusing to allow everyone to connect
            default_args.append("-rpcallowip=127.0.0.1")
            default_args.append(f"-rpcbind=127.0.0.1:{port}")

        daemon.add_daemon_settings(default_args)
        daemon.add_daemon_settings(extra_args)
        return daemon

    # pylint: disable=dangerous-default-value
    def add_node(
        self,
        extra_args: List[str] = [],
        variant: str = "florestad",
        tls: bool = False,
    ) -> Node:
        """
        Add a node settings to be run. Use this on set_test_params method
        many times you want. Extra_args should be a list of string in the
        --key=value strings (see florestad --help for a list of available
        commands)
        """
        # PR #331 introduced a preparatory environment at
        # /tmp/floresta-integration-tests.$(git rev-parse HEAD).
        # So, check for it first before define the florestad path.
        tempdir = str(FlorestaTestFramework.get_integration_test_dir())
        targetdir = os.path.normpath(os.path.join(tempdir, "binaries"))

        # Daemon can be a variant of Floresta, Utreexo or Bitcoin Core
        testname = self.__class__.__name__.lower()

        # If the variant is florestad or utreexod, maybe we need to
        # create a TLS key and certificate. Bitcoind does not need to
        # be testsd with TLS, so it does not need to create it.
        # Also, Setup the RPC server configuration based on the variant
        if variant == "florestad":
            setup_daemon = getattr(self, "setup_florestad_daemon")
            daemon = setup_daemon(targetdir, tempdir, testname, extra_args, tls)
            rpcserver = copy.deepcopy(florestad_rpc_server)
        elif variant == "utreexod":
            setup_daemon = getattr(self, "setup_utreexod_daemon")
            daemon = setup_daemon(targetdir, tempdir, testname, extra_args, tls)
            rpcserver = copy.deepcopy(utreexod_rpc_server)
        elif variant == "bitcoind":
            setup_daemon = getattr(self, "setup_bitcoind_daemon")
            daemon = setup_daemon(targetdir, tempdir, testname, extra_args)
            rpcserver = copy.deepcopy(bitcoind_rpc_server)
        else:
            raise ValueError(
                f"Unsupported variant: {variant}. Use 'florestad', 'utreexod' or 'bitcoind'."
            )

        # Node has been setup, now we can create the Node object
        node = Node(daemon, rpc=None, rpc_config=rpcserver, variant=variant)
        self._nodes.append(node)
        return node

    def get_node(self, index: int) -> Node:
        """
        Given an index, return a node configuration.
        If the node not exists, raise a IndexError exception.
        """
        if index < 0 or index >= len(self._nodes):
            raise IndexError(
                f"Node {index} not found. Please run it with add_node_settings"
            )
        return self._nodes[index]

    # pylint: disable=too-many-branches
    def detect_ports(
        self, mode: str, log_file: TextIO, timeout: int = 180
    ) -> Dict[str, int]:
        """Generic port detector for florestad, utreexod, and bitcoind logs."""
        required_patterns: Dict[str, re.Pattern]
        optional_patterns: Dict[str, re.Pattern] = {}

        # Rpc and electrum ports are required for florestad while the
        # tls electrum port is optional.
        if mode == "florestad":
            required_patterns = {
                "rpc": re.compile(r"RPC server is running at [0-9.]+:(\d+)"),
                "electrum-server": re.compile(
                    r"Electrum Server is running at [0-9.]+:(\d+)"
                ),
            }
            optional_patterns = {
                "electrum-server-tls": re.compile(
                    r"Electrum TLS Server is running at [0-9.]+:(\d+)"
                )
            }

        # Rpc and p2p ports are required for utreexod while the
        # tls electrum port is optional (TODO: add it).
        elif mode == "utreexod":
            required_patterns = {
                "rpc": re.compile(r".*RPCS: RPC server listening on [\d.]+:(\d+)"),
                "p2p": re.compile(r".*CMGR: Server listening on [\d.]+:(\d+)"),
            }

        # The rpc and p2p ports are required for bitcoind
        elif mode == "bitcoind":
            required_patterns = {
                "rpc": re.compile(r"Binding RPC on address [0-9.]+ port (\d+)"),
                "p2p": re.compile(r"Bound to [0-9.]+:(\d+)"),
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Initialize the ports dictionary with None
        # for each required and optional pattern
        ports: Dict[str, int] = {}

        # Read the log file until we find the required ports
        log_file.seek(0, 2)
        start_time = time.time()
        time_tls = None
        tls_period = 0.5

        # Read the log file line by line until we find all required ports
        while time.time() - start_time <= timeout:
            line = log_file.readline()
            if not line:
                time.sleep(0.1)
                continue

            for name, pattern in required_patterns.items():
                if name not in ports:
                    match = pattern.search(line)
                    if match:
                        ports[name] = int(match.group(1))
                        self.log(f"Detected {mode} {name} port: {ports[name]}")

            for name, pattern in optional_patterns.items():
                if name not in ports:
                    match = pattern.search(line)
                    if match:
                        ports[name] = int(match.group(1))
                        self.log(f"Detected {mode} optional {name} port: {ports[name]}")

            # If we find all required ports, we need to wait a little
            # bit to see if there's any TLS port that could have not
            # found yet
            if all(name in ports for name in required_patterns):
                if not optional_patterns:
                    return ports
                if time_tls is None:
                    time_tls = time.time()
                elif (time.time() - time_tls) >= tls_period:
                    return ports

        raise TimeoutError(
            f"Timeout waiting for {mode} ports: {list(required_patterns)}"
        )

    def run_node(self, node: Node, timeout: int = 180):
        """
        Run a node given an index on self._tests.

        If the node not exists raise a IndexError. At the time
        the tests will only run nodes configured to run on regtest.

        This will start with open a file in read mode that was created
        in the parent process in write mode (something similar to the
        `tail` command behavior) and will allow capturing the RPC,
        P2P port and Electrum ports from the log file.

        It will read the log file until it finds a line with the
        "RPC server running on:" pattern and return the port.
        """
        node.daemon.start()

        # Open the log file for reading and detect the RPC port
        log_path = self.get_test_log_path()

        # This could use resource-allocating operations.
        # But since the log file is created by the parent process
        # with `open(log_path, "w")` and closed in the parent process,
        # and we read the file while it is in writing mode,
        # do not ever call close here
        #
        # pylint: disable=R1732
        log_file = open(log_path, "r", encoding="utf-8")

        # Capture the RPC port from the log file
        # This is a workaround for multiple nodes running on
        # multithreaded mode, where the same rpc ports could
        # not be shared.
        node.rpc_config["ports"] = self.detect_ports(node.variant, log_file)
        self.log(node.rpc_config)

        if node.variant == "florestad":
            node.rpc = FlorestaRPC(node.daemon.process, node.rpc_config)

        if node.variant == "utreexod":
            node.rpc = UtreexoRPC(node.daemon.process, node.rpc_config)

        if node.variant == "bitcoind":
            node.rpc = BitcoinRPC(node.daemon.process, node.rpc_config)

        node.rpc.wait_for_connections(opened=True, timeout=timeout)
        self.log(f"Node '{node.variant}' started")

    def stop_node(self, index: int):
        """
        Stop a node given an index on self._tests.
        """
        node = self.get_node(index)
        return node.stop()

    def stop(self):
        """
        Stop all nodes.
        """
        for i in range(len(self._nodes)):
            self.stop_node(i)

    # pylint: disable=invalid-name
    def assertTrue(self, condition: bool):
        """
        Assert if the condition is True, otherwise
        all nodes will be stopped and an AssertionError will
        be raised.
        """
        if not condition:
            self.stop()
            raise AssertionError(f"Actual: {condition}\nExpected: True")

    def assertFalse(self, condition: bool):
        """
        Assert if the condition is False, otherwise
        all nodes will be stopped and an AssertionError will
        be raised.
        """
        if condition:
            self.stop()
            raise AssertionError(f"Actual: {condition}\nExpected: False")

    # pylint: disable=invalid-name
    def assertIsNone(self, thing: Any):
        """
        Assert if the condition is None, otherwise
        all nodes will be stopped and an AssertionError will
        be raised.
        """
        if thing is not None:
            self.stop()
            raise AssertionError(f"Actual: {thing}\nExpected: None")

    # pylint: disable=invalid-name
    def assertIsSome(self, thing: Any):
        """
        Assert if the condition is not None, otherwise
        all nodes will be stopped and an AssertionError will
        be raised.
        """
        if thing is None:
            self.stop()
            raise AssertionError(f"Actual: {thing}\nExpected: not None")

    # pylint: disable=invalid-name
    def assertEqual(self, condition: Any, expected: Any):
        """
        Assert if the condition is True, otherwise
        all nodes will be stopped and an AssertionError will
        be raised.
        """

        if not condition == expected:
            self.stop()
            raise AssertionError(f"Actual: {condition}\nExpected: {expected}")

    # pylint: disable=invalid-name
    def assertNotEqual(self, condition: Any, expected: Any):
        """
        Assert if the condition is True, otherwise
        all nodes will be stopped and an AssertionError will
        be raised.
        """

        if condition == expected:
            self.stop()
            raise AssertionError(f"Actual: {condition}\nExpected: !{expected}")

    # pylint: disable=invalid-name
    def assertIn(self, element: Any, listany: List[Any]):
        """
        Assert if the element is in listany , otherwise
        all nodes will be stopped and an AssertionError will
        be raised.
        """

        if element not in listany:
            self.stop()
            raise AssertionError(
                f"Actual: {element} not in {listany}\nExpected: {element} in {listany}"
            )

    # pylint: disable=invalid-name
    def assertMatch(self, actual: Any, pattern: Pattern):
        """
        Assert if the element fully matches a pattern, otherwise
        all nodes will be stopped and an AssertionError will
        be raised
        """

        if not re.fullmatch(pattern, actual):
            self.stop()
            raise AssertionError(
                f"Actual: {actual} !~ {pattern} \nExpected: {actual} ~ {set}"
            )

    def assertRaises(self, expected_exception):
        """Assert that the expected exception is raised."""
        return self._AssertRaisesContext(self, expected_exception)

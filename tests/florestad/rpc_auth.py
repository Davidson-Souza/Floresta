# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Functional tests for florestad's JSON-RPC authentication layer.

Covers the four configurations the daemon supports:
  - default cookie auth: anonymous request rejected, cookie request accepted
  - configured username/password: wrong password rejected
  - password without a username: accepted with an empty user, no cookie written
  - multi-user `-rpcauth`: pre-hashed entry accepted with the matching plaintext

Plus the brute-force cap, which only an end-to-end test can measure: the Rust
unit test can check that a slot is held, not that the check needs one.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

from requests import post
from requests.exceptions import ReadTimeout
from requests.models import HTTPBasicAuth

from test_framework.node import NodeType

# Mirrors `AUTH_SLOTS` and `reject`'s delay in
# crates/floresta-node/src/json_rpc/auth.rs.
AUTH_SLOTS = 16
REJECT_DELAY = 0.250

# Pre-computed rpcauth triple, generated with Bitcoin Core's rpcauth.py algorithm
# (HMAC-SHA256 keyed by the salt's ASCII bytes over the plaintext password).
# Pinned in-tree so the test stays deterministic without invoking rpcauth.py at
# test time.
RPCAUTH_USER = "alice"
RPCAUTH_PASSWORD = "AlicePassword42"
RPCAUTH_SALT = "cf2c6b493e4690de904306d1a82ef1cc"
RPCAUTH_HASH = "1fa75db549cdb6b598c791f5db104d10919b6b0e6f2ebca11f66a348fe204727"
RPCAUTH_ENTRY = f"{RPCAUTH_USER}:{RPCAUTH_SALT}${RPCAUTH_HASH}"


def _rpc_url(node):
    cfg = node.rpc.config
    return f"http://{cfg.host}:{cfg.port}"


def _post_rpc(node, *, auth=None, headers=None, timeout=15):
    """Send a getblockcount POST and return the raw `requests.Response`."""
    body = b'{"jsonrpc":"2.0","id":"auth-test","method":"getblockcount","params":[]}'
    request_headers = {"content-type": "application/json"}
    if headers:
        request_headers.update(headers)
    return post(
        _rpc_url(node),
        data=body,
        headers=request_headers,
        auth=auth,
        timeout=timeout,
    )


def _start_florestad(node_manager, extra_args, load_cookie):
    """
    Register and launch a florestad node bypassing `Node.start()`. Default RPC
    credentials are cleared so `get_cmd_rpc` omits `--rpc-user`/`--rpc-password`,
    letting tests drive cookie auth or supply custom creds via `extra_args`.
    """
    node = node_manager.add_node_extra_args(
        variant=NodeType.FLORESTAD, extra_args=extra_args
    )
    node.rpc.config.user = None
    node.rpc.config.password = None
    node.daemon.start()
    node.rpc.wait_on_socket(opened=True)
    if load_cookie:
        cookie_path = os.path.join(node.daemon.data_dir, "regtest", ".cookie")
        with open(cookie_path, "r", encoding="utf-8") as fh:
            user, password = fh.read().split(":", 1)
        node.rpc.config.user = user
        node.rpc.config.password = password
    node.static_values = True
    return node


def _set_rpc_credentials(node, user, password):
    """Populate the RPC client's basic-auth credentials so node teardown's
    graceful `stop` call authenticates instead of relying on SIGTERM."""
    node.rpc.config.user = user
    node.rpc.config.password = password


class TestRpcAuth:
    """End-to-end checks of the JSON-RPC basic-auth gate."""

    def test_anonymous_request_rejected(self, node_manager):
        """A request without an Authorization header gets a 401 + WWW-Authenticate."""
        node = _start_florestad(node_manager, extra_args=[], load_cookie=True)

        resp = _post_rpc(node)

        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == 'Basic realm="jsonrpc"'
        assert resp.content == b""

    def test_cookie_request_accepted(self, node_manager):
        """The cookie file's `__cookie__:<token>` line authenticates a request."""
        node = _start_florestad(node_manager, extra_args=[], load_cookie=False)

        cookie_path = os.path.join(node.daemon.data_dir, "regtest", ".cookie")
        with open(cookie_path, "r", encoding="utf-8") as fh:
            cookie_line = fh.read()
        user, password = cookie_line.split(":", 1)
        assert user == "__cookie__"
        assert len(password) == 64

        resp = _post_rpc(node, auth=HTTPBasicAuth(user, password))

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("error") is None
        assert isinstance(body.get("result"), int)

        _set_rpc_credentials(node, user, password)

    def test_wrong_password_rejected(self, node_manager):
        """Configured user/password rejects requests sent with a bogus password."""
        node = _start_florestad(
            node_manager,
            extra_args=["--rpc-user=alice", "--rpc-password=correct-horse-battery"],
            load_cookie=False,
        )

        resp = _post_rpc(node, auth=HTTPBasicAuth("alice", "definitely-wrong"))

        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == 'Basic realm="jsonrpc"'
        assert resp.content == b""

        _set_rpc_credentials(node, "alice", "correct-horse-battery")

    def test_shutdown_keeps_a_cookie_it_did_not_write(self, node_manager):
        """Password auth writes no cookie, so shutdown must leave one alone."""
        node = _start_florestad(
            node_manager,
            extra_args=["--rpc-user=alice", "--rpc-password=correct-horse-battery"],
            load_cookie=False,
        )
        _set_rpc_credentials(node, "alice", "correct-horse-battery")

        # Stand in for another instance's cookie sharing this data dir.
        cookie_path = os.path.join(node.daemon.data_dir, "regtest", ".cookie")
        foreign = "__cookie__:not-ours"
        with open(cookie_path, "w", encoding="utf-8") as fh:
            fh.write(foreign)

        node.stop()

        assert os.path.isfile(cookie_path), "shutdown deleted a cookie we never wrote"
        with open(cookie_path, "r", encoding="utf-8") as fh:
            assert fh.read() == foreign, "shutdown overwrote a cookie we never wrote"

    def test_rpcauth_multiuser_accepted(self, node_manager):
        """A pre-hashed `-rpcauth` entry authenticates with its plaintext password."""
        node = _start_florestad(
            node_manager,
            extra_args=[f"--rpc-auth={RPCAUTH_ENTRY}"],
            load_cookie=False,
        )

        resp = _post_rpc(node, auth=HTTPBasicAuth(RPCAUTH_USER, RPCAUTH_PASSWORD))

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("error") is None
        assert isinstance(body.get("result"), int)

        _set_rpc_credentials(node, RPCAUTH_USER, RPCAUTH_PASSWORD)

    def test_password_only_authenticates_with_an_empty_user(self, node_manager):
        """`--rpc-password` alone configures an empty username and writes no
        cookie, matching `bitcoin-cli`. This is the contract floresta-cli
        relies on when only a password is given."""
        password = "correct-horse-battery"
        node = _start_florestad(
            node_manager,
            extra_args=[f"--rpc-password={password}"],
            load_cookie=False,
        )
        _set_rpc_credentials(node, "", password)

        cookie_path = os.path.join(node.daemon.data_dir, "regtest", ".cookie")
        assert not os.path.exists(
            cookie_path
        ), "password auth should not write a cookie to fall back to"

        resp = _post_rpc(node, auth=HTTPBasicAuth("", password))

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("error") is None
        assert isinstance(body.get("result"), int)

    def test_abandoned_guesses_still_hold_their_auth_slot(self, node_manager):
        """An attacker only needs the 200, so they hang up on the 401s. Timing
        the 401s would pass even uncapped; what has to be measured is whether
        the slot outlives the abandoned request. It does, so the next caller
        waits for one, which is the guess rate being capped."""
        password = "correct-horse-battery"
        node = _start_florestad(
            node_manager,
            extra_args=["--rpc-user=alice", f"--rpc-password={password}"],
            load_cookie=False,
        )
        _set_rpc_credentials(node, "alice", password)

        def unsaturated():
            start = monotonic()
            assert (
                _post_rpc(node, auth=HTTPBasicAuth("alice", password)).status_code
                == 200
            )
            return monotonic() - start

        baseline = unsaturated()

        def abandon(i):
            """Send a guess and hang up long before its 401 is due."""
            try:
                _post_rpc(
                    node,
                    auth=HTTPBasicAuth("alice", f"guess-{i}"),
                    timeout=REJECT_DELAY / 5,
                )
            except ReadTimeout:
                return True
            return False

        with ThreadPoolExecutor(max_workers=AUTH_SLOTS) as pool:
            abandoned = list(pool.map(abandon, range(AUTH_SLOTS)))
        assert all(abandoned), "guesses should have been abandoned mid-delay"

        saturated = unsaturated()

        # Half the delay, not the whole of it: some of the hold already elapsed
        # while the guesses were being abandoned. A lower bound cannot flake,
        # and freeing the slot on hang-up measured ~0.005s here.
        assert saturated >= REJECT_DELAY / 2, (
            f"a correct request should wait for a slot, took {saturated:.3f}s "
            f"against a {baseline:.3f}s baseline"
        )

    def test_empty_rpc_password_falls_back_to_cookie(self, node_manager):
        """An empty `--rpc-password` is treated as unset (Bitcoin Core parity):
        cookie auth stays on and the cookie file is written."""
        node = _start_florestad(
            node_manager,
            extra_args=["--rpc-user=alice", "--rpc-password="],
            load_cookie=True,
        )

        cookie_path = os.path.join(node.daemon.data_dir, "regtest", ".cookie")
        assert os.path.exists(
            cookie_path
        ), "empty --rpc-password should leave cookie auth enabled"

        resp = _post_rpc(
            node, auth=HTTPBasicAuth(node.rpc.config.user, node.rpc.config.password)
        )
        assert resp.status_code == 200

        resp = _post_rpc(node, auth=HTTPBasicAuth("alice", ""))
        assert resp.status_code == 401

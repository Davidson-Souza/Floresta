// SPDX-License-Identifier: MIT OR Apache-2.0

//! Floresta's JSON-RPC server.
//!
//! The server accepts both JSON-RPC 1.0 and 2.0 requests. Clients may send
//! `"jsonrpc": "1.0"`, `"jsonrpc": "2.0"`, or omit the field entirely
//! (JSON-RPC 1.0 style). All responses follow the JSON-RPC 2.0 format.
//!
//! Version acceptance is validated in [`server`] and covered by integration
//! tests in `tests/florestad/rpcserver_request_parsing.py`.

pub mod auth;
pub mod request;
pub mod res;
pub mod server;

// endpoint impls
mod blockchain;
mod control;
mod network;
mod raw_transaction;
mod wallet;

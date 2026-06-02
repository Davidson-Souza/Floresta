// SPDX-License-Identifier: MIT OR Apache-2.0

use core::net::SocketAddr;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::Method;
use axum::http::Response as HttpResponse;
use axum::http::StatusCode;
use axum::routing::post;
use bitcoin::Network;
use bitcoin::ScriptBuf;
use bitcoin::Txid;
use bitcoin::ecdsa::Signature as EcdsaSignature;
use bitcoin::hashes::hex::FromHex;
use bitcoin::taproot::Signature as TaprootSignature;
use floresta_chain::ThreadSafeChain;
use floresta_common::NetworkExt;
use floresta_compact_filters::flat_filters_store::FlatFiltersStore;
use floresta_compact_filters::network_filters::NetworkFilters;
use floresta_rpc::rpc_interfaces::BlockchainRpc;
use floresta_rpc::rpc_interfaces::ControlRpc;
use floresta_rpc::rpc_interfaces::NetworkRpc;
use floresta_rpc::rpc_interfaces::RawTransactionRpc;
use floresta_rpc::rpc_interfaces::RpcMethods;
use floresta_rpc::rpc_interfaces::WalletRpc;
use floresta_watch_only::AddressCache;
use floresta_watch_only::kv_database::KvDatabase;
use floresta_wire::node_handle::NodeHandle;
use serde_json::Value;
use serde_json::json;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::debug;
use tracing::error;
use tracing::info;

use super::auth::Auth;
use super::auth::auth_middleware;
use super::res::jsonrpc_interface::JsonRpcError;
use crate::json_rpc::request::RpcRequest;
use crate::json_rpc::request::arg_parser::get_at;
use crate::json_rpc::request::arg_parser::get_optional;
use crate::json_rpc::res::jsonrpc_interface::Response;

/// Expect message for `serde_json` serialization of types that implement `Serialize`.
pub(super) const SERIALIZATION_EXPECT_MSG: &str = "types used in RPC responses implement Serialize";

/// Expect message for HTTP response builder with hardcoded valid headers.
pub(super) const HTTP_RESPONSE_EXPECT: &str = "HTTP response built from valid hardcoded headers";

/// The server holds this to tell which rpc method is awaiting to be processed and when the request were made.
pub(super) struct InflightRpc {
    pub method: String,
    pub when: Instant,
}

/// Utility trait to ensure that the chain implements all the necessary traits
///
/// Instead of using this very complex trait bound declaration on every impl block
/// and function, this trait makes sure everything we need is implemented.
pub trait RpcChain: ThreadSafeChain + Clone {}

impl<T> RpcChain for T where T: ThreadSafeChain + Clone {}

pub struct RpcImpl<Blockchain: RpcChain> {
    pub(super) block_filter_storage: Option<Arc<NetworkFilters<FlatFiltersStore>>>,
    pub(super) network: Network,
    pub(super) chain: Blockchain,
    pub(super) wallet: Arc<AddressCache<KvDatabase>>,
    pub(super) node: NodeHandle,
    pub(super) kill_signal: Arc<RwLock<bool>>,
    pub(super) inflight: Arc<RwLock<HashMap<Value, InflightRpc>>>,
    pub(super) log_path: PathBuf,
    pub(super) start_time: Instant,
    pub(super) user_agent: String,
    pub(super) proxy: Option<SocketAddr>,
    pub(super) default_connection_is_v2: bool,
}

type Result<T> = std::result::Result<T, JsonRpcError>;

async fn handle_json_rpc_request(
    req: RpcRequest,
    state: Arc<RpcImpl<impl RpcChain>>,
) -> Result<Value> {
    let RpcRequest {
        jsonrpc,
        method,
        params,
        id,
    } = req;

    if let Some(version) = jsonrpc {
        if !["1.0", "2.0"].contains(&version.as_str()) {
            return Err(JsonRpcError::InvalidJsonRpcVersion);
        }
    }

    state.inflight.write().await.insert(
        id.clone(),
        InflightRpc {
            method: method.clone(),
            when: Instant::now(),
        },
    );
    let method = RpcMethods::from_str(&method).map_err(|_| JsonRpcError::MethodNotFound)?;
    let params = params.unwrap_or_default();

    match method {
        // Blockchain
        RpcMethods::FindTxOut => {
            let txid = get_at(&params, 0, "txid")?;
            let vout = get_at(&params, 1, "vout")?;
            let script: String = get_at(&params, 2, "script")?;
            let script = ScriptBuf::from_hex(&script).map_err(|_| JsonRpcError::InvalidScript)?;
            let height = get_optional(&params, 3, "height")?;

            state
                .clone()
                .find_tx_out(txid, vout, script, height)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetBestBlockHash => state
            .get_best_block_hash()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::GetBlock => {
            let hash = get_at(&params, 0, "block_hash")?;
            let verbosity = get_optional(&params, 1, "verbosity")?;

            state
                .get_block(hash, verbosity)
                .await
                .map(|v| serde_json::to_value(v).expect("GetBlockRes implements serde"))
        }
        RpcMethods::GetBlockFromPeer => {
            let hash = get_at(&params, 0, "block_hash")?;

            state.get_block(hash, Some(0)).await?;

            Ok(Value::Null)
        }
        RpcMethods::GetBlockchainInfo => state
            .get_blockchain_info()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::GetBlockCount => state
            .get_block_count()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::GetBlockHash => {
            let height = get_at(&params, 0, "block_height")?;
            state
                .get_block_hash(height)
                .await
                .map(|h| serde_json::to_value(h).expect(SERIALIZATION_EXPECT_MSG))
        }

        RpcMethods::GetBlockHeader => {
            let hash = get_at(&params, 0, "block_hash")?;
            let verbosity = get_optional(&params, 1, "verbosity")?;

            state
                .get_block_header(hash, verbosity)
                .await
                .map(|h| serde_json::to_value(h).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetDeploymentInfo => {
            let blockhash = get_optional(&params, 0, "blockhash")?;

            state
                .get_deployment_info(blockhash)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetDifficulty => state
            .get_difficulty()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::GetTxOut => {
            let txid = get_at(&params, 0, "txid")?;
            let vout = get_at(&params, 1, "vout")?;
            let include_mempool = get_optional(&params, 2, "include_mempool")?;

            state
                .get_tx_out(txid, vout, include_mempool)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetTxOutProof => {
            let txids: Vec<Txid> = get_at(&params, 0, "txids")?;
            let block_hash = get_optional(&params, 1, "block_hash")?;

            state
                .get_txout_proof(&txids, block_hash)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetRoots => state
            .get_roots()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),

        // Wallet
        RpcMethods::LoadDescriptor => {
            let descriptor = get_at(&params, 0, "descriptor")?;

            state
                .load_descriptor(descriptor)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::ListDescriptors => {
            return state
                .list_descriptors()
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        RpcMethods::RescanBlockchain => {
            let start_height = get_optional(&params, 0, "start_height")?;
            let stop_height = get_optional(&params, 1, "stop_height")?;
            let use_timestamp = get_optional(&params, 2, "use_timestamp")?;
            let confidence = get_optional(&params, 3, "confidence")?;

            state
                .rescan_blockchain(start_height, stop_height, use_timestamp, confidence)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        // Network
        RpcMethods::AddNode => {
            let node = get_at(&params, 0, "node")?;
            let command = get_at(&params, 1, "command")?;
            let v2transport = get_optional(&params, 2, "v2transport")?;

            state
                .add_node(node, command, v2transport)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::DisconnectNode => {
            let node_address = get_at(&params, 0, "node_address")?;
            let node_id = get_optional(&params, 1, "node_id")?;

            state
                .disconnect_node(node_address, node_id)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetAddrManInfo => state
            .get_addrman_info()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::GetConnectionCount => state
            .get_connection_count()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::GetNetworkInfo => state
            .get_network_info()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::GetPeerInfo => state
            .get_peer_info()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::Ping => {
            state.ping().await?;
            Ok(serde_json::json!(null))
        }

        // RawTransactions
        RpcMethods::SendRawTransaction => {
            let tx = get_at(&params, 0, "hex")?;
            state
                .send_raw_transaction(tx)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetRawTransaction => {
            let txid = get_at(&params, 0, "txid")?;
            let verbosity = get_optional(&params, 1, "verbosity")?;

            state
                .get_raw_transaction(txid, verbosity)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        // Control
        RpcMethods::Stop => state
            .stop()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        RpcMethods::Uptime => {
            Ok(serde_json::to_value(state.uptime().await?).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetMemoryInfo => {
            let mode = get_optional(&params, 0, "mode")?;

            let memory_info = state.get_memory_info(mode).await?;

            Ok(serde_json::to_value(memory_info).expect(SERIALIZATION_EXPECT_MSG))
        }
        RpcMethods::GetRpcInfo => state
            .get_rpc_info()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
    }
}

async fn json_rpc_request(
    State(state): State<Arc<RpcImpl<impl RpcChain>>>,
    body: Bytes,
) -> HttpResponse<Body> {
    let Ok(req): std::result::Result<RpcRequest, _> = serde_json::from_slice(&body) else {
        let error = JsonRpcError::InvalidRequest;
        let body = Response::error(error.rpc_error(), Value::Null);
        return HttpResponse::builder()
            .status(error.http_code())
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&body).expect(SERIALIZATION_EXPECT_MSG),
            ))
            .expect(HTTP_RESPONSE_EXPECT);
    };

    debug!("Received JSON-RPC request: {req:?}");

    let id = req.id.clone();
    let method_res = handle_json_rpc_request(req, state.clone()).await;

    state.inflight.write().await.remove(&id);

    let response = HttpResponse::builder()
        .status(match &method_res {
            Err(e) => e.http_code(),
            Ok(_) => StatusCode::OK,
        })
        .header("Content-Type", "application/json");

    let body = Response::from_result(method_res, id);

    response
        .body(Body::from(
            serde_json::to_vec(&body).expect(SERIALIZATION_EXPECT_MSG),
        ))
        .expect(HTTP_RESPONSE_EXPECT)
}

async fn cannot_get(_state: State<Arc<RpcImpl<impl RpcChain>>>) -> Json<Value> {
    Json(json!({
        "error": "Cannot get on this route",
    }))
}

impl<Blockchain: RpcChain> RpcImpl<Blockchain> {
    #[allow(clippy::too_many_arguments)]
    pub async fn create(
        chain: Blockchain,
        wallet: Arc<AddressCache<KvDatabase>>,
        node: NodeHandle,
        kill_signal: Arc<RwLock<bool>>,
        network: Network,
        block_filter_storage: Option<Arc<NetworkFilters<FlatFiltersStore>>>,
        address: Option<SocketAddr>,
        log_path: impl AsRef<Path>,
        user_agent: String,
        proxy: Option<SocketAddr>,
        default_connection_is_v2: bool,
        credentials: Arc<Auth>,
    ) {
        let address = address.unwrap_or_else(|| {
            format!("127.0.0.1:{}", network.default_rpc_port())
                .parse()
                .expect("hardcoded address is valid")
        });

        let listener = match tokio::net::TcpListener::bind(address).await {
            Ok(listener) => {
                let local_addr = listener
                    .local_addr()
                    .expect("Infallible: listener binding was `Ok`");
                info!("RPC server is running at {local_addr}");
                listener
            }
            Err(_) => {
                error!(
                    "Failed to bind to address {address}. Floresta is probably already running.",
                );
                std::process::exit(-1);
            }
        };

        let router = Router::new()
            .route("/", post(json_rpc_request).get(cannot_get))
            .layer(
                CorsLayer::new()
                    .allow_private_network(true)
                    .allow_methods([Method::POST, Method::HEAD]),
            )
            .layer(axum::middleware::from_fn_with_state(
                credentials,
                auth_middleware,
            ))
            .with_state(Arc::new(Self {
                chain,
                wallet,
                node,
                kill_signal,
                network,
                block_filter_storage,
                inflight: Arc::new(RwLock::new(HashMap::new())),
                log_path: log_path.as_ref().into(),
                start_time: Instant::now(),
                user_agent,
                proxy,
                default_connection_is_v2,
            }));

        axum::serve(listener, router)
            .await
            .expect("failed to start rpc server");
    }
}

/// Converts a script to ASM (assembly) format, displaying the script's operations
/// in a format similar to Bitcoin Core.
///
/// This function performs the following transformations:
/// 1. Removes OP_PUSHBYTES and OP_PUSHDATA opcodes (these are unnecessary in ASM output)
/// 2. Converts leading OP_0 to "0" and OP_PUSHNUM_1 to "1" (these represent witness versions)
/// 3. If `attempt_sighash_decode` is true, attempts to decode hexadecimal data as signatures
///    and appends their sighash type (useful for analyzing scripts in scriptSig)
///
/// # Arguments
/// * `script` - The script buffer to convert
/// * `attempt_sighash_decode` - If true, tries to parse data elements as signatures and format them
pub(super) fn to_core_asm_string(script: &ScriptBuf, attempt_sighash_decode: bool) -> String {
    let mut script_asm = script.to_asm_string();
    if !script_asm.contains(' ') {
        return script_asm;
    }

    // Remove OP_PUSHBYTES_X opcodes (these are only metadata for script serialization)
    for i in 0..=75 {
        script_asm = script_asm.replace(&format!("OP_PUSHBYTES_{} ", i), "");
    }

    // Remove OP_PUSHDATA1/2/4 opcodes (these are only metadata for script serialization)
    for i in 1..=4 {
        script_asm = script_asm.replace(&format!("OP_PUSHDATA{} ", i), "");
    }

    let mut array_script_asm: Vec<String> = script_asm.split(' ').map(String::from).collect();

    // Convert leading OP_0 to "0" - represents witness version 0
    if array_script_asm[0] == "OP_0" {
        array_script_asm[0] = "0".to_string();
    }

    // Convert leading OP_PUSHNUM_1 to "1" - represents witness version 1 (Taproot)
    if array_script_asm[0] == "OP_PUSHNUM_1" {
        array_script_asm[0] = "1".to_string();
    }

    // If enabled, attempt to decode data elements as signatures and format them
    // This is particularly useful for scriptSig analysis, where signatures are wrapped with their sighash type
    if attempt_sighash_decode {
        for word in array_script_asm.iter_mut() {
            // Skip OP codes and small words that are unlikely to be signatures
            if word.contains("OP") || word.len() <= 8 {
                continue;
            }

            if let Some(decoded) =
                try_parse_and_format_signature(&Vec::from_hex(word).unwrap_or_default())
            {
                *word = decoded;
            }
        }
    }

    array_script_asm.join(" ")
}

/// Attempts to decode a byte slice as a valid signature (ECDSA or Taproot).
/// If the bytes represent a valid signature, returns the signature with the sighash type appended.
fn try_parse_and_format_signature(signature_bytes: &[u8]) -> Option<String> {
    macro_rules! try_decode_signature {
        ($sig_type:ty) => {
            if let Ok(signature) = <$sig_type>::from_slice(signature_bytes) {
                // Extract the sighash type and remove the "SIGHASH_" prefix
                // The rust-bitcoin library prefixes "SIGHASH_" to the type name, but Bitcoin Core
                // does not include this prefix in the output
                let label = signature.sighash_type.to_string().replace("SIGHASH_", "");
                return Some(format!("{}[{}]", signature.signature, label));
            }
        };
    }

    // Attempt to parse as ECDSA signature
    try_decode_signature!(EcdsaSignature);

    // Attempt to parse as Taproot signature
    try_decode_signature!(TaprootSignature);

    // If the bytes don't match any known signature format, return None
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_script_into_asm_not_attempt_sighash_decode() {
        let test_cases = [
            // P2WPKH
            (
                "0014aabc2cd363103811113b040c541afe3759489c96",
                "0 aabc2cd363103811113b040c541afe3759489c96",
            ),
            // BECH32
            (
                "0014251619c32f6500664e71a6d0393ec4b5f6da549c",
                "0 251619c32f6500664e71a6d0393ec4b5f6da549c",
            ),
            (
                "0014aa138477d24cb7b7a84160ef55af14b7bfb98143",
                "0 aa138477d24cb7b7a84160ef55af14b7bfb98143",
            ),
            // P2PKH
            (
                "76a914e7d68c17e6275b2e5c1da053ef648c676c38962488ac",
                "OP_DUP OP_HASH160 e7d68c17e6275b2e5c1da053ef648c676c389624 OP_EQUALVERIFY OP_CHECKSIG",
            ),
            (
                "76a9144eb2df72d9befff81b6dd985044d2d1b3ed4de4188ac",
                "OP_DUP OP_HASH160 4eb2df72d9befff81b6dd985044d2d1b3ed4de41 OP_EQUALVERIFY OP_CHECKSIG",
            ),
            // P2SH
            (
                "a914fae946075d1f629d35ed4067eca928c1632f4fef87",
                "OP_HASH160 fae946075d1f629d35ed4067eca928c1632f4fef OP_EQUAL",
            ),
            // P2TR (Taproot)
            (
                "51209ec7be23a1ec17cd9c4b621d899eec02bacde1d754ab080f9e1ac8445820014e",
                "1 9ec7be23a1ec17cd9c4b621d899eec02bacde1d754ab080f9e1ac8445820014e",
            ),
        ];

        for (script_hex, expected_asm) in test_cases.iter() {
            let script = ScriptBuf::from_hex(script_hex).unwrap();
            let asm = to_core_asm_string(&script, false);

            assert_eq!(asm, *expected_asm);
        }
    }

    #[test]
    fn test_converter_script_into_asm_attempt_sighash_decode() {
        let test_cases = [
            // scriptSig with ECDSA signature and pubkey
            (
                "47304402205a9b7c4432f9d895cbf4ac78519ae4e9776d47776078521b93e06beda560dd9a02202b1afbda3c917c2698b38f78203e03d2743069939e3ce2b6a3a153e148502f19012103fde976887234670c672e33a4707356997df737f3e7ac6de809164b5a606b8bad",
                "304402205a9b7c4432f9d895cbf4ac78519ae4e9776d47776078521b93e06beda560dd9a02202b1afbda3c917c2698b38f78203e03d2743069939e3ce2b6a3a153e148502f19[ALL] 03fde976887234670c672e33a4707356997df737f3e7ac6de809164b5a606b8bad",
            ),
            (
                "47304402204ab6753b249205b01d938826189cefaa4176e32ca5aa64fc6fd51891fb78fed2022065b7ba08d8739884ba232f5f7bf6efbb36b2cf98917630c64343cad2fe9db3a2012102ecf8dfb67cae8fe66d700cb13c458e5cc59be2a1c5f3ca3c5a54745259cbe45c",
                "304402204ab6753b249205b01d938826189cefaa4176e32ca5aa64fc6fd51891fb78fed2022065b7ba08d8739884ba232f5f7bf6efbb36b2cf98917630c64343cad2fe9db3a2[ALL] 02ecf8dfb67cae8fe66d700cb13c458e5cc59be2a1c5f3ca3c5a54745259cbe45c",
            ),
            // P2WPKH
            (
                "160014bb180b7bf33f066f7b557c09a0bd3b6accc84fcf",
                "0014bb180b7bf33f066f7b557c09a0bd3b6accc84fcf",
            ),
        ];

        for (script_hex, expected_asm) in test_cases.iter() {
            let script = ScriptBuf::from_hex(script_hex).unwrap();
            let asm = to_core_asm_string(&script, true);

            assert_eq!(asm, *expected_asm);
        }
    }
}

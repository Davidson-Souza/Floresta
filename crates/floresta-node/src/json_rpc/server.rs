// SPDX-License-Identifier: MIT OR Apache-2.0

use core::net::SocketAddr;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
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
use bitcoin::Address;
use bitcoin::Network;
use bitcoin::ScriptBuf;
use bitcoin::Transaction;
use bitcoin::TxIn;
use bitcoin::TxOut;
use bitcoin::Txid;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::ecdsa::Signature as EcdsaSignature;
use bitcoin::hashes::hex::FromHex;
use bitcoin::hex;
use bitcoin::hex::DisplayHex;
use bitcoin::taproot::Signature as TaprootSignature;
use corepc_types::ScriptPubKey;
use corepc_types::ScriptSig;
use corepc_types::v30::GetRawTransactionVerbose;
use corepc_types::v31::RawTransactionInput;
use corepc_types::v31::RawTransactionOutput;
use floresta_chain::ThreadSafeChain;
use floresta_common::NetworkExt;
use floresta_compact_filters::filters_man::FilterManHandle;
use floresta_compact_filters::filters_man::RescanRequest;
use floresta_compact_filters::filters_man::RescanStatus;
use floresta_watch_only::AddressCache;
use floresta_watch_only::CachedTransaction;
use floresta_watch_only::kv_database::KvDatabase;
use floresta_wire::node_handle::NodeHandle;
use floresta_wire::node_interface::MempoolMethods;
use serde_json::Value;
use serde_json::json;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::debug;
use tracing::error;
use tracing::info;

use super::res::GetRawTransactionRes;
use super::res::jsonrpc_interface::JsonRpcError;
use crate::json_rpc::request::RpcRequest;
use crate::json_rpc::request::arg_parser::get_at;
use crate::json_rpc::request::arg_parser::get_optional;
use crate::json_rpc::request::arg_parser::get_with_default;
use crate::json_rpc::res::RescanConfidence;
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
    pub(super) network: Network,
    pub(super) chain: Blockchain,
    pub(super) wallet: Arc<AddressCache<KvDatabase>>,
    pub(super) node: NodeHandle,
    pub(super) filters: Option<FilterManHandle>,
    pub(super) kill_signal: Arc<RwLock<bool>>,
    pub(super) inflight: Arc<RwLock<HashMap<Value, InflightRpc>>>,
    pub(super) log_path: PathBuf,
    pub(super) start_time: Instant,
    pub(super) user_agent: String,
    pub(super) proxy: Option<SocketAddr>,
    pub(super) default_connection_is_v2: bool,
}

type Result<T> = std::result::Result<T, JsonRpcError>;

impl<Blockchain: RpcChain> RpcImpl<Blockchain> {
    fn get_raw_transaction(&self, tx_id: Txid, verbosity: u8) -> Result<GetRawTransactionRes> {
        if verbosity > 1 {
            return Err(JsonRpcError::InvalidVerbosityLevel);
        }

        let tx = self
            .wallet
            .get_transaction(&tx_id)
            .ok_or(JsonRpcError::TxNotFound)?;

        match verbosity {
            0 => Ok(GetRawTransactionRes::Zero(serialize_hex(&tx.tx))),
            1 => Ok(GetRawTransactionRes::One(Box::new(
                self.make_raw_transaction(tx)?,
            ))),
            _ => Err(JsonRpcError::InvalidVerbosityLevel),
        }
    }

    fn load_descriptor(&self, descriptor: String) -> Result<bool> {
        let addresses = self.wallet.push_descriptor(&descriptor)?;
        info!("Descriptor pushed: {descriptor}");
        debug!("Rescanning with block filters for addresses: {addresses:?}");

        if addresses.is_empty() {
            return Ok(true);
        }

        let filters = self.filters.clone().ok_or(JsonRpcError::NoBlockFilters)?;
        let chain = self.chain.clone();
        let wallet = self.wallet.clone();
        tokio::spawn(async move {
            if let Err(error) =
                Self::rescan_with_block_filters(addresses, chain, wallet, filters, None, None).await
            {
                error!(?error, "descriptor rescan failed");
            }
        });

        Ok(true)
    }

    fn rescan_blockchain(
        &self,
        start: u32,
        stop: u32,
        use_timestamp: bool,
        confidence: RescanConfidence,
    ) -> Result<bool> {
        let (start_height, stop_height) =
            self.get_rescan_interval(use_timestamp, start, stop, confidence)?;

        if stop_height != 0 && start_height > stop_height {
            return Err(JsonRpcError::InvalidRescanVal);
        }
        if self.chain.is_in_ibd() {
            return Err(JsonRpcError::InInitialBlockDownload);
        }

        let addresses = self.wallet.get_cached_addresses();
        if addresses.is_empty() {
            return Err(JsonRpcError::NoAddressesToRescan);
        }

        let filters = self.filters.clone().ok_or(JsonRpcError::NoBlockFilters)?;
        let chain = self.chain.clone();
        let wallet = self.wallet.clone();
        tokio::spawn(async move {
            let start = (start_height != 0).then_some(start_height);
            let stop = (stop_height != 0).then_some(stop_height);
            if let Err(error) =
                Self::rescan_with_block_filters(addresses, chain, wallet, filters, start, stop)
                    .await
            {
                error!(?error, "blockchain rescan failed");
            }
        });

        Ok(true)
    }

    async fn send_raw_transaction(&self, tx: String) -> Result<Txid> {
        let tx_hex = Vec::from_hex(&tx).map_err(|_| JsonRpcError::InvalidHex)?;
        let tx: Transaction =
            deserialize(&tx_hex).map_err(|e| JsonRpcError::Decode(e.to_string()))?;

        Ok(self
            .node
            .broadcast_transaction(tx)
            .await
            .map_err(|e| JsonRpcError::Node(e.to_string()))??)
    }
}

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

    // Methods that don't require params
    match method.as_str() {
        "getbestblockhash" => {
            return state
                .get_best_block_hash()
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "getblockchaininfo" => {
            return state
                .get_blockchain_info()
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "getblockcount" => {
            return state
                .get_block_count()
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "getconnectioncount" => {
            return state
                .get_connection_count()
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "getnetworkinfo" => {
            return state
                .get_network_info()
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "getpeerinfo" => {
            return state
                .get_peer_info()
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "getroots" => {
            return state
                .get_roots()
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "getrpcinfo" => {
            return state
                .get_rpc_info()
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "listdescriptors" => {
            return state
                .list_descriptors()
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "ping" => {
            state.ping().await?;
            return Ok(serde_json::json!(null));
        }
        "stop" => {
            return state
                .stop()
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG));
        }
        "uptime" => {
            return Ok(serde_json::to_value(state.uptime()).expect(SERIALIZATION_EXPECT_MSG));
        }
        _ => {}
    }

    // Methods that do require parameters.
    //
    // Here we use `unwrap_or_default()` because there are methods with only optional
    // parameters.
    // Therefore, even if the request is parsed and the `params` field was omitted it's nice
    // to turn it into `Some(Value)` so the job of gathering inputs for calling the inner
    // rpc method goes to the getters under request.rs.
    let params = params.unwrap_or_default();

    match method.as_str() {
        "addnode" => {
            let node = get_at(&params, 0, "node")?;
            let command = get_at(&params, 1, "command")?;
            let v2transport = get_optional(&params, 2, "v2transport")?;

            state
                .add_node(node, command, v2transport)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "disconnectnode" => {
            let node_address = get_at(&params, 0, "node_address")?;
            let node_id = get_optional(&params, 1, "node_id")?;

            state
                .disconnect_node(node_address, node_id)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "findtxout" => {
            let txid = get_at(&params, 0, "txid")?;
            let vout = get_at(&params, 1, "vout")?;
            let script: String = get_at(&params, 2, "script")?;
            let script = ScriptBuf::from_hex(&script).map_err(|_| JsonRpcError::InvalidScript)?;
            let height = get_with_default(&params, 3, "height", 0)?;

            state.clone().find_tx_out(txid, vout, script, height).await
        }

        "getblock" => {
            let hash = get_at(&params, 0, "block_hash")?;
            let verbosity = get_with_default(&params, 1, "verbosity", 1)?;

            state
                .get_block(hash, verbosity)
                .await
                .map(|v| serde_json::to_value(v).expect("GetBlockRes implements serde"))
        }

        "getblockfrompeer" => {
            let hash = get_at(&params, 0, "block_hash")?;

            state.get_block(hash, 0).await?;

            Ok(Value::Null)
        }

        "getblockhash" => {
            let height = get_at(&params, 0, "block_height")?;
            state
                .get_block_hash(height)
                .map(|h| serde_json::to_value(h).expect(SERIALIZATION_EXPECT_MSG))
        }

        "getblockheader" => {
            let hash = get_at(&params, 0, "block_hash")?;
            let verbosity = get_with_default(&params, 1, "verbosity", true)?;

            state
                .get_block_header(hash, verbosity)
                .await
                .map(|h| serde_json::to_value(h).expect(SERIALIZATION_EXPECT_MSG))
        }

        "getmemoryinfo" => {
            let mode: String = get_with_default(&params, 0, "mode", "stats".into())?;

            state
                .get_memory_info(&mode)
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "getrawtransaction" => {
            let txid = get_at(&params, 0, "txid")?;
            let verbosity = get_with_default(&params, 1, "verbosity", 0)?;

            state
                .get_raw_transaction(txid, verbosity)
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "getdeploymentinfo" => {
            let blockhash = get_optional(&params, 0, "blockhash")?;

            state
                .get_deployment_info(blockhash)
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "getdifficulty" => state
            .get_difficulty()
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),
        "getaddrmaninfo" => state
            .get_addrman_info()
            .await
            .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG)),

        "gettxout" => {
            let txid = get_at(&params, 0, "txid")?;
            let vout = get_at(&params, 1, "vout")?;
            let include_mempool = get_with_default(&params, 2, "include_mempool", false)?;

            state
                .get_tx_out(txid, vout, include_mempool)
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "gettxoutproof" => {
            let txids: Vec<Txid> = get_at(&params, 0, "txids")?;
            let block_hash = get_optional(&params, 1, "block_hash")?;

            state
                .get_txout_proof(&txids, block_hash)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "loaddescriptor" => {
            let descriptor = get_at(&params, 0, "descriptor")?;

            state
                .load_descriptor(descriptor)
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "rescanblockchain" => {
            let start_height = get_with_default(&params, 0, "start_height", 0)?;
            let stop_height = get_with_default(&params, 1, "stop_height", 0)?;
            let use_timestamp = get_with_default(&params, 2, "use_timestamp", false)?;
            let confidence = get_with_default(&params, 3, "confidence", RescanConfidence::Medium)?;

            state
                .rescan_blockchain(start_height, stop_height, use_timestamp, confidence)
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        "sendrawtransaction" => {
            let tx = get_at(&params, 0, "hex")?;
            state
                .send_raw_transaction(tx)
                .await
                .map(|v| serde_json::to_value(v).expect(SERIALIZATION_EXPECT_MSG))
        }

        _ => Err(JsonRpcError::MethodNotFound),
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
    pub(super) async fn rescan_with_block_filters(
        addresses: Vec<ScriptBuf>,
        chain: Blockchain,
        wallet: Arc<AddressCache<KvDatabase>>,
        filters: FilterManHandle,
        start_height: Option<u32>,
        stop_height: Option<u32>,
    ) -> Result<()> {
        let request = RescanRequest::new(addresses).with_range(start_height, stop_height);
        let ticket = filters
            .rescan(request)
            .await
            .map_err(|error| JsonRpcError::Filters(error.to_string()))?;

        loop {
            for block in filters
                .get_blocks(ticket)
                .await
                .map_err(|error| JsonRpcError::Filters(error.to_string()))?
            {
                let height = chain
                    .get_block_height(&block.block_hash())
                    .map_err(|_| JsonRpcError::Chain)?
                    .ok_or(JsonRpcError::BlockNotFound)?;
                wallet.block_process(&block, height);
            }

            match filters
                .get_info(ticket)
                .await
                .map_err(|error| JsonRpcError::Filters(error.to_string()))?
            {
                RescanStatus::Finished => return Ok(()),
                RescanStatus::Available => continue,
                RescanStatus::Started | RescanStatus::Waiting => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    fn make_vin(&self, input: TxIn, is_coinbase: bool) -> RawTransactionInput {
        let sequence = input.sequence.0;
        let txin_witness = (!input.witness.is_empty()).then_some(
            input
                .witness
                .iter()
                .map(|w| w.to_hex_string(hex::Case::Lower))
                .collect(),
        );

        if is_coinbase {
            return RawTransactionInput {
                coinbase: Some(input.script_sig.to_hex_string()),
                sequence,
                txin_witness,
                script_sig: None,
                txid: None,
                vout: None,
            };
        }

        let txid = Some(input.previous_output.txid.to_string());
        let vout = Some(input.previous_output.vout);
        let script_sig = ScriptSig {
            asm: to_core_asm_string(&input.script_sig, true),
            hex: input.script_sig.to_hex_string(),
        };

        RawTransactionInput {
            coinbase: None,
            txid,
            vout,
            script_sig: Some(script_sig),
            txin_witness,
            sequence,
        }
    }

    fn make_vout(&self, output: TxOut, index: u64) -> RawTransactionOutput {
        let value = output.value;
        RawTransactionOutput {
            value: value.to_btc(),
            index,
            script_pubkey: ScriptPubKey {
                asm: to_core_asm_string(&output.script_pubkey, false),
                hex: output.script_pubkey.to_hex_string(),
                // `Address::from_script` can fail for nonstandard scripts. Bitcoin Core
                // omits the `address` field entirely when `ExtractDestination` fails:
                // https://github.com/bitcoin/bitcoin/blob/f50d53c84736f8ada8419346c4d1734d5a6686d4/src/core_io.cpp#L424
                address: Address::from_script(&output.script_pubkey, self.network)
                    .map(|a| a.to_string())
                    .ok(),
                type_: Self::get_script_type_label(&output.script_pubkey).to_string(),
                descriptor: Some(Self::get_script_type_descriptor(
                    &output.script_pubkey,
                    &Address::from_script(&output.script_pubkey, self.network).ok(),
                )),
                required_signatures: None, // This field is deprecated in Core v22
                addresses: None,           // This field is deprecated in Core v22
            },
        }
    }

    fn make_raw_transaction(&self, tx: CachedTransaction) -> Result<GetRawTransactionVerbose> {
        let raw_tx = tx.tx;
        let in_active_chain = tx.height != 0;
        let hex = serialize_hex(&raw_tx);
        let txid = raw_tx.compute_txid().to_string();

        let mut block_hash = None;
        let mut block_time = None;
        let mut transaction_time = None;
        let mut confirmations = Some(0);
        if in_active_chain {
            confirmations = self.chain.get_height().ok().and_then(|tip| {
                if tip >= tx.height {
                    Some((tip - tx.height + 1).into())
                } else {
                    None
                }
            });

            if let Ok(hash) = self.chain.get_block_hash(tx.height) {
                if let Ok(header) = self.chain.get_block_header(&hash) {
                    block_hash = Some(header.block_hash().to_string());
                    block_time = Some(header.time.into());
                    transaction_time = Some(header.time.into());
                }
            }
        }

        Ok(GetRawTransactionVerbose {
            in_active_chain: Some(in_active_chain),
            hex,
            txid,
            hash: raw_tx.compute_wtxid().to_string(),
            size: raw_tx.total_size().try_into()?,
            vsize: raw_tx.vsize().try_into()?,
            weight: raw_tx.weight().to_wu(),
            version: raw_tx.version.0,
            lock_time: raw_tx.lock_time.to_consensus_u32(),
            inputs: raw_tx
                .input
                .iter()
                .map(|input| self.make_vin(input.clone(), raw_tx.is_coinbase()))
                .collect(),
            outputs: raw_tx
                .output
                .into_iter()
                .enumerate()
                .map(|(i, output)| -> Result<RawTransactionOutput> {
                    let index = i.try_into()?;
                    Ok(self.make_vout(output, index))
                })
                .collect::<Result<Vec<_>>>()?,
            block_hash,
            confirmations,
            block_time,
            transaction_time,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create(
        chain: Blockchain,
        wallet: Arc<AddressCache<KvDatabase>>,
        node: NodeHandle,
        filters: Option<FilterManHandle>,
        kill_signal: Arc<RwLock<bool>>,
        network: Network,
        address: Option<SocketAddr>,
        log_path: impl AsRef<Path>,
        user_agent: String,
        proxy: Option<SocketAddr>,
        default_connection_is_v2: bool,
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
            .with_state(Arc::new(Self {
                chain,
                wallet,
                node,
                filters,
                kill_signal,
                network,
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

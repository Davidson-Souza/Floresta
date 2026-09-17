// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use bitcoin::ScriptBuf;
use bitcoin::Transaction;
use bitcoin::TxOut;
use bitcoin::Txid;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::hashes::hex::FromHex;
use bitcoin::hashes::sha256;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_common::get_hash_from_u8;
use floresta_common::get_spk_hash;
use floresta_common::spsc::Channel;
use floresta_common::try_and_log;
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
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::net::TcpListener;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio_rustls::TlsAcceptor;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::trace;
use tracing::warn;

use crate::get_arg;
use crate::json_rpc_res;
use crate::request::Request;

/// How often do we re-broadcast our transactions, until it gets confirmed
///
/// One day, in seconds
const REBROADCAST_INTERVAL: u64 = 24 * 3600;

/// Type alias for u32 representing a ClientId
type ClientId = u32;

pub enum SenderMessage {
    Write(Vec<u8>),
    Shutdown,
}

pub trait AsyncStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<S: AsyncRead + AsyncWrite + Unpin + Send> AsyncStream for S {}

struct TcpActor<S: AsyncStream> {
    stream: S,
    receiver: UnboundedReceiver<SenderMessage>,
    message_transmitter: UnboundedSender<Message>,
    client_id: ClientId,
}

impl<S: AsyncStream> TcpActor<S> {
    async fn run(&mut self) {
        let (reader, mut writer) = tokio::io::split(&mut self.stream);
        let mut lines = BufReader::new(reader).lines();

        loop {
            tokio::select! {
                Some(message) = self.receiver.recv() => {
                    match message {
                        SenderMessage::Write(data) => {
                            if let Err(e) = writer.write_all(&data).await {
                                error!("Error writing to client: {e:?}");
                                break;
                            }
                        }
                        SenderMessage::Shutdown => {
                            break;
                        }
                    }
                }
                result = lines.next_line() => {
                    match result {
                        Ok(Some(line)) => {
                            self.message_transmitter
                                .send(Message::Message((self.client_id, line)))
                                .expect("Main loop is broken");
                        }
                        Ok(None) => {
                            info!("Client closed connection: {}", self.client_id);
                            self.message_transmitter
                                .send(Message::Disconnect(self.client_id))
                                .expect("Main loop is broken");
                            break;
                        }
                        Err(e) => {
                            if e.kind() == std::io::ErrorKind::InvalidData {
                                warn!(
                                    "Error reading from client: {e:?}. Perhaps you are sending \
                                     TLS traffic on a cleartext stream? Disable TLS in your \
                                     wallet, or connect to the SSL Electrum port instead."
                                );
                            } else {
                                warn!("Error reading from client: {e:?}");
                            }
                            self.message_transmitter
                                .send(Message::Disconnect(self.client_id))
                                .expect("Main loop is broken");
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// A client connected to the server
#[derive(Debug, Clone)]
pub struct Client {
    client_id: ClientId,
    _addresses: HashSet<ScriptBuf>,
    sender: UnboundedSender<SenderMessage>,
}

impl Client {
    /// Send a message to the client, should be a serialized JSON
    fn write(&self, data: &[u8]) -> Result<(), std::io::Error> {
        let _ = self.sender.send(SenderMessage::Write(data.to_vec()));
        let _ = self
            .sender
            .send(SenderMessage::Write("\n".to_string().as_bytes().to_vec()));

        Ok(())
    }

    /// Create a new client from a stream
    pub fn new<S: AsyncStream + 'static>(
        client_id: ClientId,
        stream: S,
        message_transmitter: UnboundedSender<Message>,
    ) -> Self {
        let (sender, receiver) = unbounded_channel();
        let mut actor = TcpActor {
            stream,
            receiver,
            message_transmitter,
            client_id,
        };
        tokio::spawn(async move {
            actor.run().await;
        });
        Self {
            client_id,
            _addresses: HashSet::new(),
            sender,
        }
    }
}

pub enum Message {
    /// A new client just connected to the server
    NewClient((ClientId, Arc<Client>)),
    /// Some client just sent a message
    Message((ClientId, String)),
    /// A client just disconnected
    Disconnect(ClientId),
}

pub struct ElectrumServer<Blockchain: BlockchainInterface> {
    /// The blockchain backend we are using. This will be used to query
    /// blockchain information and broadcast transactions.
    chain: Arc<Blockchain>,

    /// The address cache is used to store addresses and transactions, like a
    /// watch-only wallet, but it is adapted to the electrum protocol.
    address_cache: Arc<AddressCache<KvDatabase>>,

    /// The clients are the clients connected to our server, we keep track of them
    /// using a unique id.
    clients: HashMap<ClientId, Arc<Client>>,

    /// The message_receiver receive messages and handles them.
    message_receiver: UnboundedReceiver<Message>,

    /// The message_transmitter is used to send requests from clients or notifications
    /// like new or dropped clients
    message_transmitter: UnboundedSender<Message>,

    /// The client_addresses is used to keep track of the addresses of each client.
    /// We keep the script_hash and which client has it, so we can notify the
    /// clients when a new transaction is received.
    client_addresses: HashMap<sha256::Hash, Arc<Client>>,

    /// An interface to a running node, used to broadcast transactions and request
    /// blocks.
    node_interface: NodeHandle,

    /// Compact-filter service used for historical rescans.
    filter_handle: Option<FilterManHandle>,

    /// A list of addresses that we've just learned about and need to rescan for
    /// transactions.
    ///
    /// We accumulate those addresses here and then periodically
    /// scan, since a wallet will often send multiple addresses, but
    /// in different requests.
    addresses_to_scan: Vec<ScriptBuf>,

    /// Last time we've re-broadcasted our transactions. We want to do this every hour, to make
    /// sure our transactions don't get stuck in the mempool if they are not getting confirmed for
    /// some reason. We keep track of this time to know when to re-broadcast them.
    last_rebroadcast: Option<Instant>,
}

impl<Blockchain: BlockchainInterface + Send + Sync + 'static> ElectrumServer<Blockchain> {
    pub fn new(
        address_cache: Arc<AddressCache<KvDatabase>>,
        chain: Arc<Blockchain>,
        node_interface: NodeHandle,
        filter_handle: Option<FilterManHandle>,
    ) -> Result<Self, Box<dyn error::Error>> {
        let (tx, rx) = unbounded_channel();

        Ok(Self {
            last_rebroadcast: None,
            chain,
            address_cache,
            node_interface,
            filter_handle,
            clients: HashMap::new(),
            message_receiver: rx,
            message_transmitter: tx,
            client_addresses: HashMap::new(),
            addresses_to_scan: Vec::new(),
        })
    }

    /// Notifier to send messages to the main loop
    pub fn get_notifier(&self) -> UnboundedSender<Message> {
        self.message_transmitter.clone()
    }

    /// Handle a request from a client. All methods are defined in the electrum
    /// protocol.
    async fn handle_client_request(
        &mut self,
        client: Arc<Client>,
        request: Request,
    ) -> Result<Value, super::error::Error> {
        // Methods are in alphabetical order
        match request.method.as_str() {
            "blockchain.block.header" => {
                let height = get_arg!(request, u32, 0);
                let hash = self
                    .chain
                    .get_block_hash(height)
                    .map_err(|_| super::error::Error::InvalidParams)?;
                let header = self
                    .chain
                    .get_block_header(&hash)
                    .map_err(|e| super::error::Error::Blockchain(Box::new(e)))?;
                let header = serialize_hex(&header);
                json_rpc_res!(request, header)
            }
            "blockchain.block.headers" => {
                const MAX_COUNT: u32 = 2016;

                let start_height = get_arg!(request, u32, 0);
                let count = get_arg!(request, u32, 1).min(MAX_COUNT);

                let chain_height = self
                    .chain
                    .get_height()
                    .map_err(|e| super::error::Error::Blockchain(Box::new(e)))?;

                let end_height =
                    (chain_height.saturating_add(1)).min(start_height.saturating_add(count));

                let heights = start_height..end_height;
                let count = heights.len();

                let headers = heights.filter_map(|height| {
                    let hash = self.chain.get_block_hash(height).ok()?;
                    let header = self.chain.get_block_header(&hash).ok()?;
                    Some(serialize_hex(&header))
                });

                json_rpc_res!(request, {
                    "count": count,
                    "hex": String::from_iter(headers),
                    "max": MAX_COUNT,
                })
            }
            "blockchain.estimatefee" => json_rpc_res!(request, 0.0001),
            "blockchain.headers.subscribe" => {
                let (height, hash) = self
                    .chain
                    .get_best_block()
                    .map_err(|e| super::error::Error::Blockchain(Box::new(e)))?;
                let header = self
                    .chain
                    .get_block_header(&hash)
                    .map_err(|e| super::error::Error::Blockchain(Box::new(e)))?;
                let result = json!({
                    "height": height,
                    "hex": serialize_hex(&header)
                });
                json_rpc_res!(request, result)
            }
            // TODO: Ask Backend for fees
            "blockchain.relayfee" => json_rpc_res!(request, 0.00001),
            "blockchain.scripthash.get_balance" => {
                let script_hash = get_arg!(request, sha256::Hash, 0);
                let balance = self.address_cache.get_address_balance(&script_hash);
                let result = json!({
                    "confirmed": balance,
                    "unconfirmed": 0
                });
                json_rpc_res!(request, result)
            }
            "blockchain.scripthash.get_history" => {
                let script_hash = get_arg!(request, sha256::Hash, 0);
                self.address_cache
                    .get_address_history(&script_hash)
                    .map(|transactions| {
                        let res = Self::process_history(&transactions);
                        json_rpc_res!(request, res)
                    })
                    .unwrap_or_else(|| {
                        Ok(json!({
                            "jsonrpc": "2.0",
                            "result": [],
                            "id": request.id
                        }))
                    })
            }
            "blockchain.scripthash.get_mempool" => json_rpc_res!(request, []),
            "blockchain.scripthash.listunspent" => {
                let hash = get_arg!(request, sha256::Hash, 0);
                let utxos = self.address_cache.get_address_utxos(&hash);
                if utxos.is_none() {
                    return json_rpc_res!(request, []);
                }
                let mut final_utxos = Vec::new();
                for (utxo, prevout) in utxos.unwrap().into_iter() {
                    let height = self.address_cache.get_height(&prevout.txid).unwrap();

                    final_utxos.push(json!({
                        "height": height,
                        "tx_pos": prevout.vout,
                        "tx_hash": prevout.txid,
                        "value": utxo.value
                    }));
                }

                json_rpc_res!(request, final_utxos)
            }
            "blockchain.scripthash.subscribe" => {
                let hash = get_arg!(request, sha256::Hash, 0);
                self.client_addresses.insert(hash, client);

                let history = self.address_cache.get_address_history(&hash);
                match history {
                    Some(transactions) if !transactions.is_empty() => {
                        let res = get_status(transactions);
                        json_rpc_res!(request, res)
                    }
                    _ => {
                        json_rpc_res!(request, null)
                    }
                }
            }
            "blockchain.scripthash.unsubscribe" => {
                let address = get_arg!(request, sha256::Hash, 0);
                self.client_addresses.remove(&address);
                json_rpc_res!(request, true)
            }

            // those endpoinsts are experimental and aren't implemented by any other implementation yet
            "blockchain.scriptpubkey.get_balance" => {
                let script = get_arg!(request, ScriptBuf, 0);
                let hash = get_spk_hash(&script);

                if !self.address_cache.is_address_cached(&hash) {
                    self.address_cache.cache_address(script.clone());
                    self.addresses_to_scan.push(script);
                    let res = json!({
                        "confirmed": 0,
                        "unconfirmed": 0
                    });
                    return json_rpc_res!(request, res);
                }

                let balance = self.address_cache.get_address_balance(&hash);
                let result = json!({
                    "confirmed": balance,
                    "unconfirmed": 0
                });
                json_rpc_res!(request, result)
            }
            "blockchain.scriptpubkey.get_history" => {
                let script = get_arg!(request, ScriptBuf, 0);
                let hash = get_spk_hash(&script);

                if !self.address_cache.is_address_cached(&hash) {
                    self.address_cache.cache_address(script.clone());
                    self.addresses_to_scan.push(script);
                    return json_rpc_res!(request, null);
                }

                self.address_cache
                    .get_address_history(&hash)
                    .map(|transactions| {
                        let res = Self::process_history(&transactions);
                        json_rpc_res!(request, res)
                    })
                    .unwrap_or_else(|| {
                        Ok(json!({
                            "jsonrpc": "2.0",
                            "result": null,
                            "id": request.id
                        }))
                    })
            }
            "blockchain.scriptpubkey.subscribe" => {
                let script = get_arg!(request, ScriptBuf, 0);
                let hash = get_spk_hash(&script);
                self.client_addresses.insert(hash, client);

                let history = self.address_cache.get_address_history(&hash);
                match history {
                    Some(transactions) if !transactions.is_empty() => {
                        let res = get_status(transactions);
                        json_rpc_res!(request, res)
                    }
                    Some(_) => {
                        json_rpc_res!(request, null)
                    }
                    None => {
                        self.addresses_to_scan.push(script);
                        json_rpc_res!(request, null)
                    }
                }
            }
            "blockchain.scriptpubkey.unsubscribe" => {
                let script = get_arg!(request, ScriptBuf, 0);
                let hash = get_spk_hash(&script);
                self.client_addresses.remove(&hash);
                json_rpc_res!(request, true)
            }

            // end of experimental endpoints
            "blockchain.transaction.broadcast" => {
                let tx = get_arg!(request, String, 0);
                let hex: Vec<_> =
                    Vec::from_hex(&tx).map_err(|_| super::error::Error::InvalidParams)?;
                let tx: Transaction =
                    deserialize(&hex).map_err(|_| super::error::Error::InvalidParams)?;

                let txid = tx.compute_txid();
                if let Err(e) = self
                    .node_interface
                    .broadcast_transaction(tx.clone())
                    .await?
                {
                    error!("Could not broadcast transaction {txid} due to {e}");
                    return Err(super::error::Error::Mempool(Box::new(e)));
                };

                let updated = self
                    .address_cache
                    .cache_mempool_transaction(&tx)
                    .into_iter()
                    .map(|spend| (tx.clone(), spend))
                    .collect::<Vec<_>>();

                self.wallet_notify(&updated);
                json_rpc_res!(request, txid)
            }
            "blockchain.transaction.get" => {
                let tx_id = get_arg!(request, Txid, 0);
                let tx = self.address_cache.get_cached_transaction(&tx_id);
                if let Some(tx) = tx {
                    return json_rpc_res!(request, tx);
                }

                Err(super::error::Error::InvalidParams)
            }
            "blockchain.transaction.get_merkle" => {
                let tx_id = get_arg!(request, Txid, 0);
                let proof = self.address_cache.get_merkle_proof(&tx_id);
                let height = self.address_cache.get_height(&tx_id);
                if let Some(proof) = proof {
                    let result = json!({
                        "merkle": proof.hashes,
                        "block_height": height.unwrap_or(0),
                        "pos": proof.pos
                    });
                    return json_rpc_res!(request, result);
                }

                Err(super::error::Error::InvalidParams)
            }
            //blockchain.transaction.id_from_pos
            // TODO: Create an actual histogram
            "mempool.get_fee_histogram" => json_rpc_res!(request, []),
            "server.add_peer" => json_rpc_res!(request, true),
            "server.banner" => json_rpc_res!(request, "Welcome to Floresta's Electrum Server."),
            "server.donation_address" => {
                json_rpc_res!(request, "")
            }
            "server.features" => {
                let genesis_hash = self
                    .chain
                    .get_block_hash(0)
                    .expect("Genesis block should be present");
                let res = json!(
                    {
                        "genesis_hash": genesis_hash,
                        "hosts": {"127.0.0.1": {"tcp_port": 50001}},
                        "protocol_max": "1.5",
                        "protocol_min": "1.4",
                        "pruning": null,
                        "server_version": format!("Floresta {}", env!("CARGO_PKG_VERSION")),
                        "hash_function": "sha256"
                    }
                );
                json_rpc_res!(request, res)
            }
            "server.peers.subscribe" => json_rpc_res!(request, []),
            "server.ping" => json_rpc_res!(request, null),
            "server.version" => json_rpc_res!(
                request,
                [format!("Floresta {}", env!("CARGO_PKG_VERSION")), "1.4"]
            ),

            _ => Err(super::error::Error::InvalidParams),
        }
    }

    pub async fn rebroadcast_mempool_transactions(&self) {
        let unconfirmed = self.address_cache.find_unconfirmed().unwrap();
        for tx in unconfirmed {
            let txid = tx.compute_txid();
            if let Ok(Err(e)) = self.node_interface.broadcast_transaction(tx.clone()).await {
                error!("Could not rebroadcast transaction {txid} due to {e}");
            } else {
                debug!("Rebroadcasted transaction {txid}");
            }
        }
    }

    pub async fn main_loop(mut self) -> Result<(), crate::error::Error> {
        let blocks = Channel::new();
        let blocks = Arc::new(blocks);

        self.chain.subscribe(blocks.clone());

        loop {
            for (block, height) in blocks.recv() {
                self.handle_block(block, height);
            }

            // handles client requests
            while let Ok(request) = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                self.message_receiver.recv(),
            )
            .await
            {
                if let Some(message) = request {
                    self.handle_message(message).await?;
                }
            }

            // Handle transaction rebroadcast
            let should_rebroadcast = self
                .last_rebroadcast
                .map(|last| {
                    !self.chain.is_in_ibd()
                        && last.elapsed() > Duration::from_secs(REBROADCAST_INTERVAL)
                })
                .unwrap_or(true);

            if should_rebroadcast {
                self.rebroadcast_mempool_transactions().await;
                self.last_rebroadcast = Some(Instant::now());
            }

            // Rescan for scripts learned through the scriptpubkey endpoints.
            if !self.addresses_to_scan.is_empty() {
                if self.chain.is_in_ibd() {
                    continue;
                }

                self.addresses_to_scan.iter().for_each(|address| {
                    self.address_cache.cache_address(address.clone());
                });

                let addresses = std::mem::take(&mut self.addresses_to_scan);
                info!("Catching up with addresses {addresses:?}");
                if let Err(error) = self.rescan_for_addresses(&addresses).await {
                    error!(%error, "Electrum script rescan failed");
                    if !matches!(error, super::error::Error::CompactFiltersDisabled) {
                        self.addresses_to_scan.extend(addresses);
                    }
                }
            }
        }
    }

    async fn rescan_for_addresses(
        &mut self,
        addresses: &[ScriptBuf],
    ) -> Result<(), super::error::Error> {
        let filters = self
            .filter_handle
            .clone()
            .ok_or(super::error::Error::CompactFiltersDisabled)?;
        let ticket = filters
            .rescan(RescanRequest::new(addresses.to_vec()))
            .await
            .map_err(|error| super::error::Error::CompactFilters(error.to_string()))?;

        loop {
            let blocks = filters
                .get_blocks(ticket)
                .await
                .map_err(|error| super::error::Error::CompactFilters(error.to_string()))?;
            for block in blocks {
                let height = self
                    .chain
                    .get_block_height(&block.block_hash())
                    .map_err(|error| super::error::Error::Blockchain(Box::new(error)))?
                    .ok_or(super::error::Error::InvalidParams)?;
                self.handle_block(block, height);
            }

            match filters
                .get_info(ticket)
                .await
                .map_err(|error| super::error::Error::CompactFilters(error.to_string()))?
            {
                RescanStatus::Finished => return Ok(()),
                RescanStatus::Available => continue,
                RescanStatus::Started | RescanStatus::Waiting => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    fn process_history(transactions: &[CachedTransaction]) -> Vec<Value> {
        let mut res = Vec::new();
        for transaction in transactions {
            let entry = if transaction.height == 0 {
                json!({
                    "tx_hash": transaction.hash,
                    "height": transaction.height,
                    "fee": 2000
                })
            } else {
                json!({
                    "tx_hash": transaction.hash,
                    "height": transaction.height,
                })
            };

            res.push(entry);
        }
        res
    }

    fn handle_block(&self, block: bitcoin::Block, height: u32) {
        let result = json!({
            "jsonrpc": "2.0",
            "method": "blockchain.headers.subscribe",
            "params": [{
                "height": height,
                "hex": serialize_hex(&block.header)
            }]
        });

        let current_height = self.address_cache.get_cache_height();

        if (!self.chain.is_in_ibd() || height % 1000 == 0) && (height > current_height) {
            self.address_cache.bump_height(height);
        }

        if self.chain.get_height().unwrap() == height {
            for client in &mut self.clients.values() {
                try_and_log!(client.write(serde_json::to_string(&result).unwrap().as_bytes()));
            }
        }

        let transactions = self.address_cache.block_process(&block, height);

        self.wallet_notify(&transactions);
    }

    /// Handles each kind of Message
    async fn handle_message(&mut self, message: Message) -> Result<(), crate::error::Error> {
        match message {
            Message::NewClient((id, client)) => {
                self.clients.insert(id, client);
            }

            Message::Message((client, msg)) => {
                trace!("Message: {msg}");
                if let Ok(req) = serde_json::from_str::<Request>(msg.as_str()) {
                    let client = self.clients.get(&client);
                    if client.is_none() {
                        error!("Client sent a message but is not listed as client");
                        return Ok(());
                    }
                    let client = client.unwrap().to_owned();
                    let id = req.id.to_owned();
                    let res = self.handle_client_request(client.clone(), req).await;

                    if let Ok(res) = res {
                        client.write(serde_json::to_string(&res).unwrap().as_bytes())?;
                    } else {
                        let res = json!({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32000,
                                "message": "Internal JSON-RPC error.",
                                "data": null
                            },
                            "id": id
                        });
                        client.write(serde_json::to_string(&res).unwrap().as_bytes())?;
                    }
                } else if let Ok(requests) = serde_json::from_str::<Vec<Request>>(&msg) {
                    let mut results = Vec::new();
                    for req in requests {
                        let client = self.clients.get(&client);
                        if client.is_none() {
                            error!("Client sent a message but is not listed as client");
                            return Ok(());
                        }
                        let client = client.unwrap().to_owned();
                        let id = req.id.to_owned();
                        let res = self.handle_client_request(client.clone(), req).await;

                        if let Ok(res) = res {
                            results.push(res);
                        } else {
                            let res = json!({
                                "jsonrpc": "2.0",
                                "error": {
                                    "code": -32000,
                                    "message": "Internal JSON-RPC error.",
                                    "data": null
                                },
                                "id": id
                            });
                            results.push(res);
                        }
                    }
                    if let Some(client) = self.clients.get(&client) {
                        client.write(serde_json::to_string(&results).unwrap().as_bytes())?;
                    }
                } else {
                    let res = json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error. Invalid JSON was received by the server.",
                            "data": null
                        },
                        "id": null
                    });
                    if let Some(client) = self.clients.get(&client) {
                        client.write(serde_json::to_string(&res).unwrap().as_bytes())?;
                    }
                }
            }

            Message::Disconnect(id) => {
                self.clients.remove(&id);
            }
        }

        Ok(())
    }

    fn wallet_notify(&self, transactions: &[(Transaction, TxOut)]) {
        for (_, out) in transactions {
            let hash = get_spk_hash(&out.script_pubkey);
            if let Some(client) = self.client_addresses.get(&hash) {
                let history = self.address_cache.get_address_history(&hash);

                let status_hash = get_status(history.unwrap());
                let notify = json!({
                    "jsonrpc": "2.0",
                    "method": "blockchain.scripthash.subscribe",
                    "params": [hash, status_hash]
                });

                try_and_log!(client.write(serde_json::to_string(&notify).unwrap().as_bytes()));
            }
        }
    }
}

/// Listens to new TCP connections in a loop
pub async fn client_accept_loop(
    listener: Arc<TcpListener>,
    message_transmitter: UnboundedSender<Message>,
    tls_acceptor: Option<TlsAcceptor>,
) {
    let mut id_count = 0;
    loop {
        if let Ok((stream, _addr)) = listener.accept().await {
            info!("New client connection");
            let message_transmitter = message_transmitter.clone();
            if let Some(acceptor) = tls_acceptor.clone() {
                match acceptor.accept(stream).await {
                    Ok(tls_stream) => {
                        let client = Arc::new(Client::new(
                            id_count,
                            tls_stream,
                            message_transmitter.clone(),
                        ));
                        message_transmitter
                            .send(Message::NewClient((client.client_id, client)))
                            .expect("Main loop is broken");
                        id_count += 1;
                    }
                    Err(e) => {
                        error!("TLS accept error: {e:?}");
                    }
                }
            } else {
                let client = Arc::new(Client::new(id_count, stream, message_transmitter.clone()));
                message_transmitter
                    .send(Message::NewClient((client.client_id, client)))
                    .expect("Main loop is broken");
                id_count += 1;
            }
        }
    }
}

/// As per electrum documentation:
/// ### To calculate the status of a script hash (or address):
///
/// 1. order confirmed transactions to the script hash by increasing height (and position in the block if there are more than one in a block)
///
/// 2. form a string that is the concatenation of strings "tx_hash:height:" for each
///    transaction in order, where:
///
///  tx_hash is the transaction hash in hexadecimal
///  height is the height of the block it is in.
///
/// 3. Next, with mempool transactions in any order, append a similar string for those
///    transactions, but where height is -1 if the transaction has at least one unconfirmed
///    input, and 0 if all inputs are confirmed.
///
/// 4. The status of the script hash is the sha256() hash of the full string expressed
///    as a hexadecimal string, or null if the string is empty because there are no
///    transactions.
fn get_status(transactions: Vec<CachedTransaction>) -> sha256::Hash {
    let mut status_preimage = String::new();
    for transaction in transactions {
        status_preimage.extend(format!("{}:{}:", transaction.hash, transaction.height).chars());
    }
    get_hash_from_u8(status_preimage.as_bytes())
}

#[macro_export]
/// Builds the response as defined by jsonrpc v2.0. Request should have type [Request] and the
/// response is always a [json]
macro_rules! json_rpc_res {
    ($request: ident, $result: ident) => (
        Ok(json!({
            "jsonrpc": "2.0",
            "result": $result,
            "id": $request.id
        }))
    );
    ($request: ident, $result: literal) => (
        Ok(json!({
            "jsonrpc": "2.0",
            "result": $result,
            "id": $request.id
        }))
    );
    ($request: ident, $result: tt) => {
        Ok(json!({
            "jsonrpc": "2.0",
            "result": $result,
            "id": $request.id
        }))
    }
}

#[macro_export]
/// Returns and parses a value from the request json or fails with [super::error::Error::InvalidParams].
macro_rules! get_arg {
    ($request:ident, $arg_type:ty, $idx:literal) => {
        if let Some(arg) = $request.params.get($idx) {
            serde_json::from_value::<$arg_type>(arg.clone())?
        } else {
            return Err(super::error::Error::InvalidParams);
        }
    };
}

#[cfg(test)]
mod test {
    use core::str::FromStr;
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;

    use bitcoin::Address;
    use bitcoin::Network;
    use bitcoin::Transaction;
    use bitcoin::address::NetworkChecked;
    use bitcoin::block::Header as BlockHeader;
    use bitcoin::consensus::Decodable;
    use bitcoin::consensus::deserialize;
    use bitcoin::hashes::hex::FromHex;
    use bitcoin::hashes::sha256;
    use floresta_chain::AssumeValidArg;
    use floresta_chain::ChainState;
    use floresta_chain::FlatChainStore;
    use floresta_chain::FlatChainStoreConfig;
    use floresta_chain::pruned_utreexo::merkle::ConsensusMerkle;
    use floresta_common::assert_ok;
    use floresta_common::get_spk_hash;
    use floresta_mempool::Mempool;
    use floresta_watch_only::AddressCache;
    use floresta_watch_only::kv_database::KvDatabase;
    use floresta_watch_only::merkle::MerkleProof;
    use floresta_wire::UtreexoNodeConfig;
    use floresta_wire::address_man::AddressMan;
    use floresta_wire::address_man::ReachableNetworks;
    use floresta_wire::node::UtreexoNode;
    use floresta_wire::node::running_ctx::RunningNode;
    use rcgen::CertifiedKey;
    use rcgen::generate_simple_self_signed;
    use serde_json::Number;
    use serde_json::Value;
    use serde_json::json;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::AsyncWriteExt;
    use tokio::io::BufReader;
    use tokio::net::TcpListener;
    use tokio::net::TcpStream;
    use tokio::sync::Mutex;
    use tokio::sync::RwLock;
    use tokio::task;
    use tokio::time::timeout;
    use tokio_rustls::TlsAcceptor;
    use tokio_rustls::rustls::ServerConfig;
    use tokio_rustls::rustls::pki_types::PrivateKeyDer;
    use tokio_rustls::rustls::pki_types::pem::PemObject;

    use super::ElectrumServer;
    use super::client_accept_loop;

    /// A size used for mempool tests, no specific meaning just a randomly
    /// chosen size.
    const MEMPOOL_SIZE: usize = 10_000;

    fn get_test_transaction() -> (Transaction, MerkleProof) {
        // Signet transaction with id 6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea
        // block hash 0000009298f9e75a91fa763c78b66d1555cb059d9ca9d45601eed2b95166a151.

        let transaction = "020000000001017ca523c5e6df0c014e837279ab49be1676a9fe7571c3989aeba1e5d534f4054a0000000000fdffffff01d2410f00000000001600142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a02473044022071b8583ba1f10531b68cb5bd269fb0e75714c20c5a8bce49d8a2307d27a082df022069a978dac00dd9d5761aa48c7acc881617fa4d2573476b11685596b17d437595012103b193d06bd0533d053f959b50e3132861527e5a7a49ad59c5e80a265ff6a77605eece0100";
        let transaction = Vec::from_hex(transaction).unwrap();
        let transaction: Transaction = deserialize(&transaction).unwrap();

        let merkle_block = "0100000000000000ea530307089e3e6f6e8997a0ae48e1dc2bee84635bc4e6c6ecdcc7225166b06b010000000000000034086ef398efcdec47b37241221c8f4613e02bc31026cc74d07ddb3092e6d6e7";
        let merkle_block = Vec::from_hex(merkle_block).unwrap();
        let merkle_block: MerkleProof = deserialize(&merkle_block).unwrap();

        (transaction, merkle_block)
    }

    fn get_test_signet_headers() -> Vec<BlockHeader> {
        let file = include_bytes!("../../floresta-chain/testdata/signet_headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
        let mut buffer = uncompressed.as_slice();

        let mut headers: Vec<BlockHeader> = Vec::new();
        while let Ok(header) = BlockHeader::consensus_decode(&mut buffer) {
            headers.push(header);
        }

        headers.remove(0);
        headers
    }

    fn get_test_cache() -> Arc<AddressCache<KvDatabase>> {
        let test_id: u32 = rand::random();
        let cache = KvDatabase::new(format!("./tmp-db/{test_id}.floresta")).unwrap();
        let cache = AddressCache::new(cache, ConsensusMerkle);

        // Inserting test transactions in the wallet
        let (transaction, proof) = get_test_transaction();
        cache.cache_transaction(
            &transaction,
            118511,
            transaction.output[0].value.to_sat(),
            proof,
            1,
            0,
            false,
            get_spk_hash(&transaction.output[0].script_pubkey),
        );

        Arc::new(cache)
    }

    fn get_test_address() -> (Address<NetworkChecked>, sha256::Hash) {
        let address = Address::from_str("tb1q9d4zjf92nvd3zhg6cvyckzaqumk4zre26x02q9")
            .unwrap()
            .assume_checked();
        let script_hash = get_spk_hash(&address.script_pubkey());
        (address, script_hash)
    }

    async fn send_request(request: String, port: u16) -> Result<Value, io::Error> {
        let address = format!("localhost:{port}");
        let mut stream = TcpStream::connect(address).await?;

        stream.write_all(request.as_bytes()).await?;
        stream.flush().await?;

        // Wrap the stream in a buffered reader and read one line (i.e., the full response)
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        let timeout_duration = Duration::from_secs(10);

        let len = timeout(timeout_duration, reader.read_line(&mut line))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "Timeout occurred"))??;

        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "No data read"));
        }

        Ok(serde_json::from_str(&line)?)
    }

    // Returns the port assigned by the OS
    async fn start_electrum() -> u16 {
        let e_addr = "0.0.0.0:0";
        let ssl_e_addr = "0.0.0.0:0";
        let wallet = get_test_cache();

        // Create test_chain_state
        let test_id = rand::random::<u32>();
        let conf = FlatChainStoreConfig::new(format!("./tmp-db/{test_id}.floresta/"));
        let chainstore = FlatChainStore::new(conf).unwrap();
        let chain = ChainState::<FlatChainStore>::open(
            chainstore,
            Network::Signet,
            AssumeValidArg::Hardcoded,
        )
        .unwrap();

        let headers = get_test_signet_headers();
        chain.push_headers(headers, 1).unwrap();
        let chain = Arc::new(chain);
        // Create test_node_interface
        let u_config = UtreexoNodeConfig {
            disable_dns_seeds: true,
            network: Network::Signet,
            pow_fraud_proofs: true,
            datadir: "/tmp-db".into(),
            user_agent: "floresta".to_string(),
            ..Default::default()
        };

        let chain_provider: UtreexoNode<Arc<ChainState<FlatChainStore>>, RunningNode> =
            UtreexoNode::new(
                u_config,
                chain.clone(),
                Arc::new(Mutex::new(Mempool::new(MEMPOOL_SIZE))),
                Arc::new(RwLock::new(false)),
                AddressMan::new(None, &ReachableNetworks::SUPPORTED),
            )
            .unwrap();

        let node_interface = chain_provider.get_handle();

        let tls_config = Some(create_tls_config().expect("Failed to create TLS config"));
        let tls_acceptor = tls_config.map(TlsAcceptor::from);

        let electrum_server: ElectrumServer<ChainState<FlatChainStore>> =
            ElectrumServer::new(wallet, chain, node_interface, None).unwrap();

        let non_tls_listener = Arc::new(TcpListener::bind(e_addr).await.unwrap());
        let assigned_port = non_tls_listener.local_addr().unwrap().port();

        let (stop_signal, _) = tokio::sync::oneshot::channel();
        task::spawn(chain_provider.run(stop_signal));
        task::spawn(client_accept_loop(
            non_tls_listener,
            electrum_server.message_transmitter.clone(),
            None,
        ));

        // TLS Electrum accept loop
        if let Some(tls_acceptor) = tls_acceptor {
            let tls_listener = Arc::new(TcpListener::bind(ssl_e_addr).await.unwrap());
            task::spawn(client_accept_loop(
                tls_listener,
                electrum_server.message_transmitter.clone(),
                Some(tls_acceptor),
            ));
        }

        // Electrum main loop
        task::spawn(electrum_server.main_loop());
        assigned_port
    }

    /// Create a tls config thats valid for localhost with a random
    /// created key
    fn create_tls_config() -> Result<Arc<ServerConfig>, crate::error::Error> {
        let CertifiedKey { cert, signing_key } =
            generate_simple_self_signed(vec!["localhost".into()]).expect("Failed to generate cert");

        let private_key = signing_key.serialize_pem();

        let config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![cert.into()],
                PrivateKeyDer::from_pem_slice(private_key.as_ref()).expect("Failed to parse private key, this should never happen since we generated it an serialized as pem above"),
            )
            .expect("Failed to create server config. This should never happen since all values are hardcoded");

        Ok(Arc::new(config))
    }

    /// server.banner                           *
    /// blockchain.block.header                 *
    /// blockchain.block.headers                *
    /// blockchain.estimatefee                  *
    /// blockchain.headers.subscribe            *
    /// blockchain.relayfee                     *
    /// blockchain.scripthash.get_balance       *
    /// blockchain.scripthash.get_history       *
    /// blockchain.scripthash.get_mempool       *
    /// blockchain.scripthash.listunspent       *
    /// blockchain.scripthash.subscribe         *
    /// blockchain.scripthash.unsubscribe       *
    /// blockchain.transaction.broadcast        *
    /// blockchain.transaction.get              *
    /// blockchain.transaction.get_merkle       *
    /// mempool.get_fee_histogram               *
    /// server.add_peer                         *
    /// server.donation_address                 *
    /// server.features                         *
    /// server.peers.subscribe                  *
    /// server.ping                             *
    /// server.version                          *
    fn generate_request(req_params: &mut Vec<Value>) -> Value {
        let binding = req_params.pop().unwrap();
        let method = binding.as_str().unwrap();

        let params = match method {
            "server.banner" => vec![],
            "blockchain.block.header" => vec![req_params.pop().unwrap()],
            "blockchain.block.headers" => {
                vec![req_params.pop().unwrap(), req_params.pop().unwrap()]
            }
            "blockchain.estimatefee" => vec![],
            "blockchain.relayfee" => vec![],
            "blockchain.scripthash.subscribe" => vec![req_params.pop().unwrap()],
            "blockchain.scripthash.unsubscribe" => vec![req_params.pop().unwrap()],
            "blockchain.scripthash.get_mempool" => vec![],
            "blockchain.scripthash.get_balance" => vec![req_params.pop().unwrap()],
            "blockchain.scripthash.get_history" => vec![req_params.pop().unwrap()],
            "blockchain.scripthash.listunspent" => vec![req_params.pop().unwrap()],
            "blockchain.transaction.broadcast" => vec![req_params.pop().unwrap()],
            "blockchain.transaction.get" => vec![req_params.pop().unwrap()],
            "blockchain.transaction.get_merkle" => vec![req_params.pop().unwrap()],

            _ => vec![],
        };

        json!({
            "id": rand::random::<u32>(),
            "method": method,
            "jsonrpc": "2.0",
            "params": params
        })
    }

    fn generate_batch_request(batch_req_params: &mut Vec<Vec<Value>>) -> Value {
        let mut batch_request: Vec<Value> = Vec::new();

        for req_params in batch_req_params {
            batch_request.push(generate_request(req_params))
        }

        json!(batch_request)
    }

    /// SENDING MULTIPLE REQUESTS TO THE SERVER AT THE SAME TIME
    #[tokio::test]
    async fn test_blockchain_headers() {
        let port = start_electrum().await;

        let height: u32 = rand::random::<u32>() % 1500;
        let mut batch_req_params: Vec<Vec<Value>> = Vec::new();

        // blockchain.block.header
        let method = Value::String("blockchain.block.header".to_string());
        let block_header_req = vec![Value::Number(Number::from(height)), method];

        // blockchain.block.headers
        let method = Value::String("blockchain.block.headers".to_string());
        let block_headers_req = vec![
            Value::Number(Number::from(rand::random::<u32>() % 500 + 1)),
            Value::Number(Number::from(height)),
            method,
        ];

        // blockchain.headers.subscribe
        let method = Value::String("blockchain.headers.subscribe".to_string());
        let headers_subscribe_req = vec![method];

        batch_req_params.push(block_header_req);
        batch_req_params.push(block_headers_req);
        batch_req_params.push(headers_subscribe_req);

        // Create a JSON array of the batch requests
        let batch_req = generate_batch_request(&mut batch_req_params);
        let mut batch_req = json!(batch_req).to_string();
        batch_req.push('\n');

        let re = send_request(batch_req, port).await.unwrap();
        let answers = re.as_array().unwrap();
        assert_eq!(answers.len(), 3);
    }

    #[tokio::test]
    async fn test_server_banner() {
        let port = start_electrum().await;

        let method = Value::String("server.banner".to_string());
        let mut request = generate_request(&mut vec![method]).to_string();
        request.push('\n');

        assert_ok!(send_request(request, port).await);
    }

    #[tokio::test]
    async fn test_estimate_fee() {
        let port = start_electrum().await;

        let mut batch_req_params = Vec::new();

        // blockchain.estimatefee
        let method = Value::String("blockchain.estimatefee".to_string());
        let estimatefee_req = vec![method];

        // blockchain.relayfee
        let method = Value::String("blockchain.relayfee".to_string());
        let relayfee_req = vec![method];

        batch_req_params.push(estimatefee_req);
        batch_req_params.push(relayfee_req);

        // Create a JSON array of the batch requests
        let batch_req = generate_batch_request(&mut batch_req_params);
        let mut batch_req = json!(batch_req).to_string();
        batch_req.push('\n');

        let batch_response = send_request(batch_req, port).await.unwrap();

        assert_eq!(batch_response[0]["result"], 0.0001);
        assert_eq!(batch_response[1]["result"], 0.00001);
    }

    #[tokio::test]
    async fn test_scripthash_subscribe() {
        let port = start_electrum().await;

        let (_, script_hash) = get_test_address();

        // blockchain.scripthash.subscribe
        let method = Value::String("blockchain.scripthash.subscribe".to_string());
        let mut subscribe_req =
            generate_request(&mut vec![Value::String(script_hash.to_string()), method]).to_string();
        subscribe_req.push('\n');

        // blockchain.scripthash.unsubscribe
        let method = Value::String("blockchain.scripthash.unsubscribe".to_string());
        let mut unsubscribe_req =
            generate_request(&mut vec![Value::String(script_hash.to_string()), method]).to_string();
        unsubscribe_req.push('\n');

        // blockchain.scripthash.get_mempool
        let method = Value::String("blockchain.scripthash.get_mempool".to_string());
        let mut mempool_req = generate_request(&mut vec![method]).to_string();
        mempool_req.push('\n');

        assert_ok!(send_request(subscribe_req, port).await);

        assert_ok!(send_request(mempool_req, port).await);

        assert!(
            send_request(unsubscribe_req, port).await.unwrap()["result"]
                .as_bool()
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_scripthash_txs() {
        let port = start_electrum().await;

        let (_, hash) = get_test_address();

        // blockchain.scripthash.get_balance
        let method = Value::String("blockchain.scripthash.get_balance".to_string());
        let mut balance_req =
            generate_request(&mut vec![Value::String(hash.to_string()), method]).to_string();
        balance_req.push('\n');

        // blockchain.scripthash.get_history
        let method = Value::String("blockchain.scripthash.get_history".to_string());
        let mut history_req =
            generate_request(&mut vec![Value::String(hash.to_string()), method]).to_string();
        history_req.push('\n');

        // blockchain.scripthash.listunspent
        let method = Value::String("blockchain.scripthash.listunspent".to_string());
        let mut unspent_req =
            generate_request(&mut vec![Value::String(hash.to_string()), method]).to_string();
        unspent_req.push('\n');

        assert_eq!(
            send_request(balance_req, port).await.unwrap()["result"]["confirmed"],
            999890
        );

        assert_eq!(
            send_request(history_req, port).await.unwrap()["result"][0]["tx_hash"],
            "6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea".to_string()
        );

        let unspent_response = send_request(unspent_req, port).await.unwrap();

        assert_eq!(
            unspent_response["result"][0]["tx_hash"],
            "6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea".to_string()
        );

        assert_eq!(unspent_response["result"][0]["tx_pos"], 0);
    }

    #[tokio::test]
    async fn test_transactions() {
        let port = start_electrum().await;

        let unconfirmed_tx = Value::String("01000000010b7e3ac7e68944dc7a7115362391c3b7975d60f4fbe4af0ca924a172bfe7a7d9000000006b483045022100e0ff6984e5c2e16df6f309b759b75e04adf6930593b6043cd9134f87efb7e07c02206544a9f265f6041f0e3e2bd11a95ea75a112d3dc05647a9b01eca0d352feeb380121024f9c3deb05e81a3ddb17dadcf283fb132894aa70ab127395a03a3e9d382f13a3ffffffff022c92ae00000000001976a914ca9755ffb8f0e5aeca43478d8620e1a35b3baada88acc0894601000000001976a914b62ad08a3ffc469e9c0df75d1ceca49a88345fc888ac00000000".to_string());
        let confirmed_tx = "020000000001017ca523c5e6df0c014e837279ab49be1676a9fe7571c3989aeba1e5d534f4054a0000000000fdffffff01d2410f00000000001600142b6a2924aa9b1b115d1ac3098b0ba0e6ed510f2a02473044022071b8583ba1f10531b68cb5bd269fb0e75714c20c5a8bce49d8a2307d27a082df022069a978dac00dd9d5761aa48c7acc881617fa4d2573476b11685596b17d437595012103b193d06bd0533d053f959b50e3132861527e5a7a49ad59c5e80a265ff6a77605eece0100".to_string();

        // blockchain.transaction.broadcast
        let method = Value::String("blockchain.transaction.broadcast".to_string());
        let mut broadcast_req = generate_request(&mut vec![unconfirmed_tx, method]).to_string();
        broadcast_req.push('\n');

        // blockchain.transaction.get
        let method = Value::String("blockchain.transaction.get".to_string());
        let mut tx_get_req = generate_request(&mut vec![
            Value::String(
                "6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea".to_string(),
            ),
            method,
        ])
        .to_string();
        tx_get_req.push('\n');

        // blockchain.transaction.get_merkle
        let method = Value::String("blockchain.transaction.get_merkle".to_string());
        let mut get_merkle_req = generate_request(&mut vec![
            Value::String(
                "6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea".to_string(),
            ),
            method,
        ])
        .to_string();
        get_merkle_req.push('\n');

        assert_eq!(
            send_request(broadcast_req, port).await.unwrap()["result"],
            "197d099f6bc6c0b522cb04df4514622bb3d55094faf0af3474ab996e0b62b8ad".to_string()
        );

        assert_eq!(
            send_request(tx_get_req, port).await.unwrap()["result"],
            confirmed_tx
        );

        assert_eq!(
            send_request(get_merkle_req, port).await.unwrap()["result"]["merkle"][0],
            "e7d6e69230db7dd074cc2610c32be013468f1c224172b347eccdef98f36e0834".to_string()
        );
    }

    #[tokio::test]
    async fn test_server_info() {
        let port = start_electrum().await;

        let mut batch_req_params: Vec<Vec<Value>> = Vec::new();

        // mempool.get_fee_histogram
        let method = Value::String("mempool.get_fee_histogram".to_string());
        batch_req_params.push(vec![method]);

        // server.add_peer
        let method = Value::String("server.add_peer".to_string());
        batch_req_params.push(vec![method]);

        // server.donation_address
        let method = Value::String("server.donation_address".to_string());
        batch_req_params.push(vec![method]);

        // server.features
        let method = Value::String("server.features".to_string());
        batch_req_params.push(vec![method]);

        // server.peers.subscribe
        let method = Value::String("server.peers.subscribe".to_string());
        batch_req_params.push(vec![method]);

        // server.ping
        let method = Value::String("server.ping".to_string());
        batch_req_params.push(vec![method]);

        // server.version
        let method = Value::String("server.version".to_string());
        batch_req_params.push(vec![method]);

        // CREATE A JSON ARRAY OF THE BATCH REQUESTS
        let batch_req = generate_batch_request(&mut batch_req_params);
        let mut batch_req = json!(batch_req).to_string();
        batch_req.push('\n');

        let batch_response = send_request(batch_req, port).await.unwrap();

        assert!(batch_response[0]["result"].as_array().unwrap().is_empty());
        assert!(batch_response[1]["result"].as_bool().unwrap());
        assert_eq!(batch_response[2]["result"], "".to_string());
        assert_eq!(
            batch_response[3]["result"]["genesis_hash"],
            "00000008819873e925422c1ff0f99f7cc9bbb232af63a077a480a3633bee1ef6".to_string()
        );
        assert!(batch_response[4]["result"].as_array().unwrap().is_empty());
        assert!(batch_response[5]["result"].is_null());
        assert_eq!(
            batch_response[6]["result"][0],
            format!("Floresta {}", env!("CARGO_PKG_VERSION"))
        );
    }
}

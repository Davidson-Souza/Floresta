use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::hashes::hex::FromHex;
use bitcoin::hashes::sha256;
use bitcoin::p2p::message;
use bitcoin::ScriptBuf;
use bitcoin::Transaction;
use bitcoin::TxOut;
use bitcoin::Txid;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_common::get_hash_from_u8;
use floresta_common::get_spk_hash;
use floresta_common::spsc::Channel;
use floresta_compact_filters::flat_filters_store::FlatFiltersStore;
use floresta_compact_filters::network_filters::NetworkFilters;
use floresta_watch_only::kv_database::KvDatabase;
use floresta_watch_only::AddressCache;
use floresta_watch_only::CachedTransaction;
use floresta_wire::node_interface::NodeInterface;
use floresta_wire::node_interface::NodeMethods;
use log::error;
use log::info;
use log::trace;
use serde_json::json;
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::RwLock;

use crate::get_arg;
use crate::json_rpc_res;
use crate::request::Request;

/// Type alias for u32 representing a ClientId
type ClientId = u32;

pub enum SenderMessage {
    Write(Vec<u8>),
    Shutdown,
}

struct TcpActor {
    stream: TcpStream,
    receiver: UnboundedReceiver<SenderMessage>,
    message_transmitter: UnboundedSender<Message>,
    client_id: ClientId,
}

impl TcpActor {
    async fn run(&mut self) {
        let (reader, mut writer) = tokio::io::split(&mut self.stream);
        let mut lines = BufReader::new(reader).lines();

        loop {
            tokio::select! {
                Some(message) = self.receiver.recv() => {
                    match message {
                        SenderMessage::Write(data) => {
                            if let Err(e) = writer.write_all(&data).await {
                                error!("Error writing to client: {:?}", e);
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
                            error!("Error reading from client: {:?}", e);
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
    pub async fn write(&self, data: &[u8]) -> Result<(), std::io::Error> {
        let _ = self.sender.send(SenderMessage::Write(data.to_vec()));
        let _ = self
            .sender
            .send(SenderMessage::Write("\n".to_string().as_bytes().to_vec()));

        Ok(())
    }
    /// Create a new client from a stream
    pub fn new(
        client_id: ClientId,
        stream: TcpStream,
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
        Client {
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
    pub chain: Arc<Blockchain>,
    /// The address cache is used to store addresses and transactions, like a
    /// watch-only wallet, but it is adapted to the electrum protocol.
    pub address_cache: Arc<RwLock<AddressCache<KvDatabase>>>,
    /// The TCP listener is used to accept new connections to our server.
    pub tcp_listener: Arc<TcpListener>,
    /// The clients are the clients connected to our server, we keep track of them
    /// using a unique id.
    pub clients: HashMap<ClientId, Arc<Client>>,
    /// The message_receiver receive messages and handles them.
    pub message_receiver: UnboundedReceiver<Message>,
    /// The message_transmitter is used to send requests from clients or notifications
    /// like new or dropped clients
    pub message_transmitter: UnboundedSender<Message>,
    /// The client_addresses is used to keep track of the addresses of each client.
    /// We keep the script_hash and which client has it, so we can notify the
    /// clients when a new transaction is received.
    pub client_addresses: HashMap<sha256::Hash, Arc<Client>>,
    /// A Arc-ed copy of the block filters backend that we can use to check if a
    /// block contains a transaction that we are interested in.
    pub block_filters: Option<Arc<NetworkFilters<FlatFiltersStore>>>,
    /// An interface to a running node, used to broadcast transactions and request
    /// blocks.
    pub node_interface: Arc<NodeInterface>,
    /// A list of addresses that we've just learned about and need to rescan for
    /// transactions.
    ///
    /// We accumulate those addresses here and then periodically
    /// scan, since a wallet will often send multiple addresses, but
    /// in different requests.
    pub addresses_to_scan: Vec<ScriptBuf>,
}

impl<Blockchain: BlockchainInterface> ElectrumServer<Blockchain> {
    pub async fn new(
        address: String,
        address_cache: Arc<RwLock<AddressCache<KvDatabase>>>,
        chain: Arc<Blockchain>,
        block_filters: Option<Arc<NetworkFilters<FlatFiltersStore>>>,
        node_interface: Arc<NodeInterface>,
    ) -> Result<ElectrumServer<Blockchain>, Box<dyn std::error::Error>> {
        let listener = Arc::new(TcpListener::bind(address).await?);
        let (tx, rx) = unbounded_channel();
        let unconfirmed = address_cache.read().await.find_unconfirmed().unwrap();
        for tx in unconfirmed {
            chain.broadcast(&tx).expect("Invalid chain");
        }
        Ok(ElectrumServer {
            chain,
            address_cache,
            block_filters,
            node_interface,
            tcp_listener: listener,
            clients: HashMap::new(),
            message_receiver: rx,
            message_transmitter: tx,
            client_addresses: HashMap::new(),
            addresses_to_scan: Vec::new(),
        })
    }

    /// Handle a request from a client. All methods are defined in the electrum
    /// protocol.
    pub async fn handle_client_request(
        &mut self,
        client: Arc<Client>,
        request: Request,
    ) -> Result<Value, super::error::Error> {
        // Methods are in alphabetical order
        match request.method.as_str() {
            "blockchain.block.header" => {
                let height = get_arg!(request, u64, 0);
                let hash = self
                    .chain
                    .get_block_hash(height as u32)
                    .map_err(|_| super::error::Error::InvalidParams)?;
                let header = self
                    .chain
                    .get_block_header(&hash)
                    .map_err(|e| super::error::Error::Blockchain(Box::new(e)))?;
                let header = serialize_hex(&header);
                json_rpc_res!(request, header)
            }
            "blockchain.block.headers" => {
                let start_height = get_arg!(request, u64, 0);
                let count = get_arg!(request, u64, 1);
                let mut headers = String::new();
                let count = if count < 2016 { count } else { 2016 };
                for height in start_height..(start_height + count) {
                    let hash = self
                        .chain
                        .get_block_hash(height as u32)
                        .map_err(|_| super::error::Error::InvalidParams)?;

                    let header = self
                        .chain
                        .get_block_header(&hash)
                        .map_err(|e| super::error::Error::Blockchain(Box::new(e)))?;
                    let header = serialize_hex(&header);
                    headers.push_str(&header);
                }
                json_rpc_res!(request, {
                    "count": count,
                    "hex": headers,
                    "max": 2016
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
                let balance = self
                    .address_cache
                    .read()
                    .await
                    .get_address_balance(&script_hash);
                let result = json!({
                    "confirmed": balance,
                    "unconfirmed": 0
                });
                json_rpc_res!(request, result)
            }
            "blockchain.scripthash.get_history" => {
                let script_hash = get_arg!(request, sha256::Hash, 0);
                self.address_cache
                    .read()
                    .await
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
                let utxos = self.address_cache.read().await.get_address_utxos(&hash);
                if utxos.is_none() {
                    return json_rpc_res!(request, []);
                }
                let mut final_utxos = Vec::new();
                for (utxo, prevout) in utxos.unwrap().into_iter() {
                    let height = self
                        .address_cache
                        .read()
                        .await
                        .get_height(&prevout.txid)
                        .unwrap();

                    let position = self
                        .address_cache
                        .read()
                        .await
                        .get_position(&prevout.txid)
                        .unwrap();

                    final_utxos.push(json!({
                        "height": height,
                        "tx_pos": position,
                        "tx_hash": prevout.txid,
                        "value": utxo.value
                    }));
                }

                json_rpc_res!(request, final_utxos)
            }
            "blockchain.scripthash.subscribe" => {
                let hash = get_arg!(request, sha256::Hash, 0);
                self.client_addresses.insert(hash, client);

                let history = self.address_cache.read().await.get_address_history(&hash);
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

                if !self.address_cache.read().await.is_address_cached(&hash) {
                    self.address_cache
                        .write()
                        .await
                        .cache_address(script.clone());
                    self.addresses_to_scan.push(script);
                    let res = json!({
                        "confirmed": 0,
                        "unconfirmed": 0
                    });
                    return json_rpc_res!(request, res);
                }

                let balance = self.address_cache.read().await.get_address_balance(&hash);
                let result = json!({
                    "confirmed": balance,
                    "unconfirmed": 0
                });
                json_rpc_res!(request, result)
            }
            "blockchain.scriptpubkey.get_history" => {
                let script = get_arg!(request, ScriptBuf, 0);
                let hash = get_spk_hash(&script);

                if !self.address_cache.read().await.is_address_cached(&hash) {
                    self.address_cache
                        .write()
                        .await
                        .cache_address(script.clone());
                    self.addresses_to_scan.push(script);
                    return json_rpc_res!(request, null);
                }

                self.address_cache
                    .read()
                    .await
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

                let history = self.address_cache.read().await.get_address_history(&hash);
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
                self.chain
                    .broadcast(&tx)
                    .map_err(|e| super::error::Error::Blockchain(Box::new(e)))?;
                let id = tx.txid();
                let updated = self
                    .address_cache
                    .write()
                    .await
                    .cache_mempool_transaction(&tx)
                    .into_iter()
                    .map(|spend| (tx.clone(), spend))
                    .collect::<Vec<_>>();
                self.wallet_notify(&updated).await;
                json_rpc_res!(request, id)
            }
            "blockchain.transaction.get" => {
                let tx_id = get_arg!(request, Txid, 0);
                let tx = self
                    .address_cache
                    .read()
                    .await
                    .get_cached_transaction(&tx_id);
                if let Some(tx) = tx {
                    return json_rpc_res!(request, tx);
                }

                Err(super::error::Error::InvalidParams)
            }
            "blockchain.transaction.get_merkle" => {
                let tx_id = get_arg!(request, Txid, 0);
                let proof = self.address_cache.read().await.get_merkle_proof(&tx_id);
                let height = self.address_cache.read().await.get_height(&tx_id);
                if let Some((proof, position)) = proof {
                    let result = json!({
                        "merkle": proof,
                        "block_height": height.unwrap_or(0),
                        "pos": position
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

    pub async fn main_loop(mut self) -> Result<(), crate::error::Error> {
        let blocks = Channel::new();
        let blocks = Arc::new(blocks);

        self.chain.subscribe(blocks.clone());

        loop {
            for (block, height) in blocks.recv() {
                self.handle_block(block, height).await;
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

            // rescan for new addresses, if any
            if !self.addresses_to_scan.is_empty() {
                if self.chain.is_in_idb() {
                    continue;
                }

                let mut lock = self.address_cache.write().await;
                self.addresses_to_scan.iter().for_each(|address| {
                    lock.cache_address(address.clone());
                });

                drop(lock);

                info!("Catching up with addresses {:?}", self.addresses_to_scan);
                let addresses: Vec<_> = self.addresses_to_scan.drain(..).collect();
                self.rescan_for_addresses(addresses).await?;
            }
        }
    }

    /// If a user adds a new address that we didn't have cached, this method
    /// will look for historical transactions for it.
    ///
    /// Usually, we'll relly on compact block filters to speed things up. If
    /// we don't have compact block filters, we may rescan using the older,
    /// more bandwidth-intensive method of actually downloading blocks.
    async fn rescan_for_addresses(
        &mut self,
        addresses: Vec<ScriptBuf>,
    ) -> Result<(), super::error::Error> {
        // If compact block filters are enabled, use them. Otherwise, fallback
        // to the "old-school" rescaning.
        match &self.block_filters {
            Some(cfilters) => {
                self.rescan_with_block_filters(cfilters.clone(), addresses)
                    .await
            }
            None => self
                .chain
                .rescan(1)
                .map_err(|e| super::error::Error::Blockchain(Box::new(e))),
        }
    }

    /// If we have compact block filters enabled, this method will use them to
    /// find blocks of interest and download for our wallet to learn about new
    /// transactions, once a new address is added by subscription.
    async fn rescan_with_block_filters(
        &mut self,
        cfilters: Arc<NetworkFilters<FlatFiltersStore>>,
        addresses: Vec<ScriptBuf>,
    ) -> Result<(), super::error::Error> {
        // By default, we look from 1..tip
        let height = cfilters.get_height().unwrap();
        let mut _addresses = addresses
            .iter()
            .map(|address| address.as_bytes())
            .collect::<Vec<_>>();

        // TODO (Davidson): Let users select what the starting and end height is
        let blocks = cfilters.match_any(_addresses, height, self.chain.clone());
        if blocks.is_err() {
            error!("error while rescanning with block filters: {:?}", blocks);
            self.addresses_to_scan.extend(addresses); // push them back to get a retry
            return Ok(());
        }

        info!("filters told us to scan blocks: {:?}", blocks);

        let blocks = blocks
            .unwrap()
            .into_iter()
            .flat_map(|hash| self.node_interface.get_block(hash))
            .collect::<Vec<_>>();

        // Tells users about the transactions we found
        for block in blocks {
            let Some(block) = block else {
                self.addresses_to_scan.extend(addresses); // push them back to get a retry
                return Ok(());
            };

            let height = self
                .chain
                .get_block_height(&block.block_hash())
                .ok()
                .flatten()
                .unwrap();
            self.handle_block(block, height).await;
        }

        Ok(())
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

    async fn handle_block(&self, block: bitcoin::Block, height: u32) {
        let result = json!({
            "jsonrpc": "2.0",
            "method": "blockchain.headers.subscribe",
            "params": [{
                "height": height,
                "hex": serialize_hex(&block.header)
            }]
        });

        let current_height = self.address_cache.read().await.get_cache_height();

        if (!self.chain.is_in_idb() || height % 1000 == 0) && (height > current_height) {
            let lock = self.address_cache.write().await;
            lock.bump_height(height);
        }

        if self.chain.get_height().unwrap() == height {
            for client in &mut self.clients.values() {
                let res = client
                    .write(serde_json::to_string(&result).unwrap().as_bytes())
                    .await;
                if res.is_err() {
                    info!("Could not write to client {:?}", client);
                }
            }
        }

        let transactions = self
            .address_cache
            .write()
            .await
            .block_process(&block, height);

        self.wallet_notify(&transactions).await;
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
                        client
                            .write(serde_json::to_string(&res).unwrap().as_bytes())
                            .await?;
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
                        client
                            .write(serde_json::to_string(&res).unwrap().as_bytes())
                            .await?;
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
                        client
                            .write(serde_json::to_string(&results).unwrap().as_bytes())
                            .await?;
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
                        client
                            .write(serde_json::to_string(&res).unwrap().as_bytes())
                            .await?;
                    }
                }
            }

            Message::Disconnect(id) => {
                self.clients.remove(&id);
            }
        }

        Ok(())
    }

    async fn wallet_notify(&self, transactions: &[(Transaction, TxOut)]) {
        for (_, out) in transactions {
            let hash = get_spk_hash(&out.script_pubkey);
            if let Some(client) = self.client_addresses.get(&hash) {
                let history = self.address_cache.read().await.get_address_history(&hash);

                let status_hash = get_status(history.unwrap());
                let notify = json!({
                    "jsonrpc": "2.0",
                    "method": "blockchain.scripthash.subscribe",
                    "params": [hash, status_hash]
                });
                if let Err(err) = client
                    .write(serde_json::to_string(&notify).unwrap().as_bytes())
                    .await
                {
                    error!("{err}");
                }
            }
        }
    }
}

/// Each client gets one loop to deal with their requests
// async fn client_broker_loop(
//     client: Arc<Client>,
//     message_transmitter: UnboundedSender<Message>,
// ) -> Result<(), std::io::Error> {
//     let mut _stream = &*client.stream;
//     let mut lines = BufReader::new(_stream).lines();

//     while let Some(Ok(line)) = lines.next().await {
//         message_transmitter
//             .send(Message::Message((client.client_id, line)))
//             .expect("Main loop is broken");
//     }

//     info!("Lost client with ID: {}", client.client_id);

//     message_transmitter
//         .send(Message::Disconnect(client.client_id))
//         .expect("Main loop is broken");

//     Ok(())
// }

/// Listens to new TCP connections in a loop
pub async fn client_accept_loop(
    listener: Arc<TcpListener>,
    message_transmitter: UnboundedSender<Message>,
) {
    let mut id_count = 0;
    loop {
        if let Ok((stream, _addr)) = listener.accept().await {
            info!("New client connection");
            // let stream = Arc::new(stream);
            let client = Arc::new(Client::new(id_count, stream, message_transmitter.clone()));
            // tokio::task::spawn(client_broker_loop(
            //     client.clone(),
            //     message_transmitter.clone(),
            // ));

            message_transmitter
                .send(Message::NewClient((client.client_id, client)))
                .expect("Main loop is broken");
            id_count += 1;
        }
    }
}

/// As per electrum documentation:
/// ### To calculate the status of a script hash (or address):
///
/// 1. order confirmed transactions to the script hash by increasing height (and position in the block if there are more than one in a block)
///
/// 2. form a string that is the concatenation of strings "tx_hash:height:" for each
/// transaction in order, where:
///
///  tx_hash is the transaction hash in hexadecimal
///  height is the height of the block it is in.
///
/// 3. Next, with mempool transactions in any order, append a similar string for those
/// transactions, but where height is -1 if the transaction has at least one unconfirmed
/// input, and 0 if all inputs are confirmed.
///
/// 4. The status of the script hash is the sha256() hash of the full string expressed
/// as a hexadecimal string, or null if the string is empty because there are no
/// transactions.
fn get_status(transactions: Vec<CachedTransaction>) -> sha256::Hash {
    let mut status_preimage = String::new();
    for transaction in transactions {
        status_preimage.extend(format!("{}:{}:", transaction.hash, transaction.height).chars());
    }
    get_hash_from_u8(status_preimage.as_bytes())
}

#[macro_export]
/// Builds the response as defined by jsonrpc v2.0. Request should have type [Request] and the
/// response is always a [Json]
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
    use std::io::Cursor;
    use std::io::{self};
    use std::str::FromStr;
    use std::sync::Arc;
    use std::time::Duration;

    use bitcoin::address::NetworkUnchecked;
    use bitcoin::block::Header as BlockHeader;
    use bitcoin::consensus::deserialize;
    use bitcoin::consensus::Decodable;
    use bitcoin::hashes::hex::FromHex;
    use bitcoin::hashes::sha256;
    use bitcoin::Address;
    use bitcoin::Transaction;
    use floresta_chain::AssumeValidArg;
    use floresta_chain::ChainState;
    use floresta_chain::KvChainStore;
    use floresta_chain::Network;
    use floresta_common::get_spk_hash;
    use floresta_watch_only::kv_database::KvDatabase;
    use floresta_watch_only::merkle::MerkleProof;
    use floresta_watch_only::AddressCache;
    use floresta_wire::mempool::Mempool;
    use floresta_wire::node::UtreexoNode;
    use floresta_wire::running_node::RunningNode;
    use floresta_wire::UtreexoNodeConfig;
    use futures::executor::block_on;
    use serde_json::json;
    use serde_json::Number;
    use serde_json::Value;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::AsyncReadExt;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;
    use tokio::task::{self};
    use tokio::time::timeout;

    use super::client_accept_loop;
    use super::ElectrumServer;

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
        let file =
            include_bytes!("../../floresta-chain/src/pruned_utreexo/testdata/signet_headers.zst");
        let uncompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(file)).unwrap();
        let mut cursor = Cursor::new(uncompressed);
        let mut headers: Vec<BlockHeader> = Vec::new();
        while let Ok(header) = BlockHeader::consensus_decode(&mut cursor) {
            headers.push(header);
        }
        headers.remove(0);

        headers
    }
    fn get_test_cache() -> Arc<tokio::sync::RwLock<AddressCache<KvDatabase>>> {
        let test_id: u32 = rand::random();
        let cache = KvDatabase::new(format!("./data/{test_id}.floresta")).unwrap();
        let mut cache = AddressCache::new(cache);

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

        let cache = Arc::new(tokio::sync::RwLock::new(cache));
        cache
    }

    fn get_test_address() -> (Address<NetworkUnchecked>, sha256::Hash) {
        let address = Address::from_str("tb1q9d4zjf92nvd3zhg6cvyckzaqumk4zre26x02q9").unwrap();
        let script_hash = get_spk_hash(&address.payload().script_pubkey());
        (address, script_hash)
    }

    async fn send_request(request: String, port: u16) -> Result<Value, io::Error> {
        let address = format!("localhost:{}", port);
        let mut stream = TcpStream::connect(address).await?;

        stream.write_all(request.as_bytes()).await?;
        stream.flush().await?;

        let mut response = vec![0u8; 100000000];
        let timeout_duration = Duration::from_secs(10);

        let read_result = timeout(timeout_duration, stream.read(&mut response)).await;
        match read_result {
            Ok(Ok(0)) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "No data received",
            )),
            Ok(Ok(n)) => {
                let response: std::borrow::Cow<str> = String::from_utf8_lossy(&response[..n]);
                let response: Value = serde_json::from_str(&response).unwrap();
                Ok(response)
            }
            Ok(Err(e)) => {
                eprintln!("Error reading from socket: {}", e);
                Err(e)
            }
            Err(_) => Err(io::Error::new(io::ErrorKind::TimedOut, "Timeout occured")),
        }
    }

    async fn start_electrum(port: u16) {
        let e_addr = format!("0.0.0.0:{}", port);
        let wallet = get_test_cache();

        // Create test_chain_state
        let test_id = rand::random::<u32>();
        let chainstore = KvChainStore::new(format!("./data/{test_id}.floresta/")).unwrap();
        let chain =
            ChainState::<KvChainStore>::new(chainstore, Network::Signet, AssumeValidArg::Hardcoded);

        let headers = get_test_signet_headers();
        chain.push_headers(headers, 1).unwrap();
        let chain = Arc::new(chain);

        // Create test_node_interface
        let u_config = UtreexoNodeConfig {
            network: bitcoin::Network::Signet,
            pow_fraud_proofs: true,
            proxy: None,
            datadir: "/data".to_string(),
            fixed_peer: None,
            max_banscore: 50,
            compact_filters: false,
            max_outbound: 10,
            max_inflight: 20,
            assume_utreexo: None,
            backfill: false,
            filter_start_height: None,
        };

        let chain_provider: UtreexoNode<RunningNode, Arc<ChainState<KvChainStore>>> =
            UtreexoNode::new(
                u_config,
                chain.clone(),
                Arc::new(tokio::sync::RwLock::new(Mempool::new())),
                None,
            );

        let node_interface = chain_provider.get_handle();

        let electrum_server: ElectrumServer<ChainState<KvChainStore>> = block_on(
            ElectrumServer::new(e_addr, wallet, chain, None, node_interface),
        )
        .unwrap();

        task::spawn(client_accept_loop(
            electrum_server.tcp_listener.clone(),
            electrum_server.message_transmitter.clone(),
        ));
        // Electrum main loop
        task::spawn(electrum_server.main_loop());
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
    /// sserver.peers.subscribe                 *
    /// server.ping                             *
    /// server.version                          *

    fn generate_request(req_params: &mut Vec<Value>) -> Value {
        let params: Vec<Value>;
        let binding = req_params.pop().unwrap();
        let method = binding.as_str().unwrap();

        match method {
            "server.banner" => params = vec![],
            "blockchain.block.header" => params = vec![req_params.pop().unwrap()],
            "blockchain.block.headers" => {
                params = vec![req_params.pop().unwrap(), req_params.pop().unwrap()]
            }
            "blockchain.estimatefee" => params = vec![],
            "blockchain.relayfee" => params = vec![],
            "blockchain.scripthash.subscribe" => params = vec![req_params.pop().unwrap()],
            "blockchain.scripthash.unsubscribe" => params = vec![req_params.pop().unwrap()],
            "blockchain.scripthash.get_mempool" => params = vec![],
            "blockchain.scripthash.get_balance" => params = vec![req_params.pop().unwrap()],
            "blockchain.scripthash.get_history" => params = vec![req_params.pop().unwrap()],
            "blockchain.scripthash.listunspent" => params = vec![req_params.pop().unwrap()],
            "blockchain.transaction.broadcast" => params = vec![req_params.pop().unwrap()],
            "blockchain.transaction.get" => params = vec![req_params.pop().unwrap()],
            "blockchain.transaction.get_merkle" => params = vec![req_params.pop().unwrap()],

            _ => params = vec![],
        }

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
        let port = rand::random::<u16>() % 1000 + 18443;
        start_electrum(port).await;

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

        assert!(send_request(batch_req, port).await.is_ok());
    }

    #[tokio::test]
    async fn test_server_banner() {
        let port = rand::random::<u16>() % 1000 + 18443;
        start_electrum(port).await;

        let method = Value::String("server.banner".to_string());
        let mut request = generate_request(&mut vec![method]).to_string();
        request.push('\n');
        println!("{}", request);
        assert!(send_request(request, port).await.is_ok())
    }

    #[tokio::test]
    async fn test_estimate_fee() {
        let port = rand::random::<u16>() % 1000 + 18443;
        start_electrum(port).await;

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
        let port = rand::random::<u16>() % 1000 + 18443;
        start_electrum(port).await;

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

        assert!(send_request(subscribe_req, port).await.is_ok());

        assert!(send_request(mempool_req, port).await.is_ok());

        assert!(send_request(unsubscribe_req, port).await.unwrap()["result"]
            .as_bool()
            .unwrap());
    }

    #[tokio::test]
    async fn test_scripthash_txs() {
        let port = rand::random::<u16>() % 1000 + 18443;
        start_electrum(port).await;

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

        assert_eq!(
            send_request(unspent_req, port).await.unwrap()["result"][0]["tx_hash"],
            "6bb0665122c7dcecc6e6c45b6384ee2bdce148aea097896e6f3e9e08070353ea".to_string()
        )
    }

    #[tokio::test]
    async fn test_transactions() {
        let port = rand::random::<u16>() % 1000 + 18443;
        start_electrum(port).await;

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
        let port = rand::random::<u16>() % 1000 + 18443;
        start_electrum(port).await;

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
        assert_eq!(batch_response[6]["result"][0], "Floresta 0.1.0".to_string());
    }
}

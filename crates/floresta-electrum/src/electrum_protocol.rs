use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use async_std::channel::unbounded;
use async_std::channel::Receiver;
use async_std::channel::Sender;
use async_std::io::BufReader;
use async_std::net::TcpListener;
use async_std::net::TcpStream;
use async_std::prelude::*;
use async_std::sync::RwLock;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode::serialize_hex;
use bitcoin::hashes::hex::FromHex;
use bitcoin::hashes::sha256;
use bitcoin::hashes::Hash;
use bitcoin::ScriptBuf;
use bitcoin::Transaction;
use bitcoin::TxOut;
use bitcoin::Txid;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_common::get_hash_from_u8;
use floresta_common::get_spk_hash;
use floresta_common::spsc::Channel;
use floresta_compact_filters::BlockFilterBackend;
use floresta_compact_filters::QueryType;
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
use openssl::ssl::{SslAcceptor, SslFiletype, SslMethod};

use crate::get_arg;
use crate::json_rpc_res;
use crate::request::Request;

/// Type alias for u32 representing a ClientId
type ClientId = u32;

/// A client connected to the server
#[derive(Debug, Clone)]
pub struct Client {
    client_id: ClientId,
    _addresses: HashSet<ScriptBuf>,
    stream: Arc<TcpStream>,
}

impl Client {
    /// Send a message to the client, should be a serialized JSON
    pub async fn write(&self, data: &[u8]) -> Result<(), std::io::Error> {
        let mut stream = self.stream.as_ref();
        let _ = stream.write(data).await;
        let _ = stream.write('\n'.to_string().as_bytes()).await;

        Ok(())
    }
    /// Create a new client from a stream
    pub fn new(client_id: ClientId, stream: Arc<TcpStream>) -> Self {
        Client {
            client_id,
            _addresses: HashSet::new(),
            stream,
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
    pub message_receiver: Receiver<Message>,
    /// The message_transmitter is used to send requests from clients or notifications
    /// like new or dropped clients
    pub message_transmitter: Sender<Message>,
    /// The client_addresses is used to keep track of the addresses of each client.
    /// We keep the script_hash and which client has it, so we can notify the
    /// clients when a new transaction is received.
    pub client_addresses: HashMap<sha256::Hash, Arc<Client>>,
    /// A Arc-ed copy of the block filters backend that we can use to check if a
    /// block contains a transaction that we are interested in.
    pub block_filters: Option<Arc<BlockFilterBackend>>,
    /// An interface to a running node, used to broadcast transactions and request
    /// blocks.
    pub node_interface: Arc<NodeInterface>,
    /// A list of addresses that we've just learned about and need to rescan for
    /// transactions.
    ///
    /// We accumulate those addresses here and then periodically
    /// scan, since a wallet will often send multiple addresses, but
    /// in different requests.
    pub addresses_to_scan: Vec<sha256::Hash>,

    pub ssl_acceptor: Arc<SslAcceptor>
}

impl<Blockchain: BlockchainInterface> ElectrumServer<Blockchain> {
    pub async fn new(
        address: String,
        address_cache: Arc<RwLock<AddressCache<KvDatabase>>>,
        chain: Arc<Blockchain>,
        block_filters: Option<Arc<BlockFilterBackend>>,
        node_interface: Arc<NodeInterface>,
        cert_path: &str,
        key_path: &str,
    ) -> Result<ElectrumServer<Blockchain>, Box<dyn std::error::Error>> {
        let mut builder = SslAcceptor::mozilla_intermediate(SslMethod::tls()).unwrap();
        builder.set_private_key_file(key_path, SslFiletype::PEM).unwrap();
        builder.set_certificate_chain_file(cert_path).unwrap();

        let listener = Arc::new(TcpListener::bind(address).await?);
        let ssl_acceptor = Arc::new(builder.build());
        let (tx, rx) = unbounded();
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
            ssl_acceptor
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
                            "result": null,
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
                    Some(_) => {
                        json_rpc_res!(request, null)
                    }
                    None => {
                        self.addresses_to_scan.push(hash);
                        json_rpc_res!(request, null)
                    }
                }
            }
            "blockchain.scripthash.unsubscribe" => {
                let address = get_arg!(request, sha256::Hash, 0);
                self.client_addresses.remove(&address);
                json_rpc_res!(request, true)
            }
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
                        "protocol_max": "1.4",
                        "protocol_min": "1.0",
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
            while let Ok(request) = async_std::future::timeout(
                std::time::Duration::from_secs(1),
                self.message_receiver.recv(),
            )
            .await
            {
                if let Ok(message) = request {
                    self.handle_message(message).await?;
                }
            }

            // rescan for new addresses, if any
            if !self.addresses_to_scan.is_empty() {
                info!("Catching up with addresses {:?}", self.addresses_to_scan);
                let addresses: Vec<sha256::Hash> = self.addresses_to_scan.drain(..).collect();
                for address in addresses.iter().copied() {
                    self.address_cache.write().await.cache_address_hash(address);
                }
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
        addresses: Vec<sha256::Hash>,
    ) -> Result<(), super::error::Error> {
        // If compact block filters are enabled, use them. Otherwise, fallback
        // to the "old-school" rescaning.
        match &self.block_filters {
            Some(cfilters) => self.rescan_with_block_filters(cfilters, addresses).await,
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
        &self,
        cfilters: &BlockFilterBackend,
        addresses: Vec<sha256::Hash>,
    ) -> Result<(), super::error::Error> {
        // By default, we look from 1..tip
        let height = self.chain.get_height().unwrap_or(0) as u64;

        let addresses = addresses
            .into_iter()
            .map(|a| QueryType::ScriptHash(a.to_byte_array()))
            .collect::<Vec<_>>();

        // TODO (Davidson): Let users select what the starting and end height is
        let blocks: Vec<_> = cfilters
            .match_any(1, height, &addresses)
            .unwrap_or_default()
            .into_iter()
            .flat_map(|height| {
                self.chain
                    .get_block_hash(height as u32)
                    .into_iter()
                    .zip(Some(height))
            })
            .flat_map(|(hash, height)| {
                self.node_interface
                    .get_block(hash)
                    .ok()
                    .flatten()
                    .map(|block| (block, height))
            })
            .collect();

        // Tells users about the transactions we found
        for (block, height) in blocks {
            self.handle_block(block, height as u32).await;
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
async fn client_broker_loop(
    client: Arc<Client>,
    message_transmitter: Sender<Message>,
) -> Result<(), std::io::Error> {
    let mut _stream = &*client.stream;
    let mut lines = BufReader::new(_stream).lines();

    while let Some(Ok(line)) = lines.next().await {
        message_transmitter
            .send(Message::Message((client.client_id, line)))
            .await
            .expect("Main loop is broken");
    }

    info!("Lost client with ID: {}", client.client_id);

    message_transmitter
        .send(Message::Disconnect(client.client_id))
        .await
        .expect("Main loop is broken");

    Ok(())
}

/// Listens to new TCP connections in a loop
pub async fn client_accept_loop(listener: Arc<TcpListener>, message_transmitter: Sender<Message>) {
    let mut id_count = 0;
    loop {
        if let Ok((stream, _addr)) = listener.accept().await {
            info!("New client connection");
            let stream = Arc::new(stream);
            let client = Arc::new(Client::new(id_count, stream));
            async_std::task::spawn(client_broker_loop(
                client.clone(),
                message_transmitter.clone(),
            ));

            message_transmitter
                .send(Message::NewClient((client.client_id, client)))
                .await
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

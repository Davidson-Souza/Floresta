use crate::request::Request;
use crate::{get_arg, json_rpc_res};
use bitcoin::hashes::hex::FromHex;
use floresta_chain::pruned_utreexo::BlockchainInterface;
use floresta_common::spsc::Channel;
use floresta_common::{get_hash_from_u8, get_spk_hash};
use floresta_watch_only::kv_database::KvDatabase;
use floresta_watch_only::{AddressCache, CachedTransaction};

use async_std::io::prelude::BufReadExt;
use async_std::io::WriteExt;
use async_std::stream::StreamExt;
use async_std::{
    io::BufReader,
    net::{TcpListener, TcpStream},
};
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::RwLock;

use bitcoin::hashes::{hex::ToHex, sha256};
use bitcoin::{
    consensus::{deserialize, serialize},
    Script, Txid,
};
use bitcoin::{Transaction, TxOut};

use log::{error, info, trace};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Type alias for u32 representing a ClientId
type ClientId = u32;

/// A client connected to the server
#[derive(Debug, Clone)]
pub struct Client {
    client_id: ClientId,
    _addresses: HashSet<Script>,
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
    // pub tcp_listener: TcpListener,
    /// The clients are the clients connected to our server, we keep track of them
    /// using a unique id.
    pub clients: HashMap<ClientId, Arc<Client>>,
    /// The message_receiver receive messages and handles them.
    // pub message_receiver: UnboundedReceiver<Message>,
    /// The message_transmitter is used to send requests from clients or notifications
    /// like new or dropped clients
    // pub message_transmitter: UnboundedSender<Message>,
    /// The client_addresses is used to keep track of the addresses of each client.
    /// We keep the script_hash and which client has it, so we can notify the
    /// clients when a new transaction is received.
    pub client_addresses: HashMap<sha256::Hash, Arc<Client>>,
}

impl<Blockchain: BlockchainInterface> ElectrumServer<Blockchain> {
    pub async fn new(
        address_cache: Arc<RwLock<AddressCache<KvDatabase>>>,
        chain: Arc<Blockchain>,
    ) -> Result<ElectrumServer<Blockchain>, Box<dyn std::error::Error>> {
        // let (::tx, rx) = unbounded_channel::<Message>();
        let unconfirmed = address_cache.read().await.find_unconfirmed().unwrap();
        for tx in unconfirmed {
            chain.broadcast(&tx).expect("Invalid chain");
        }
        Ok(ElectrumServer {
            chain,
            address_cache,
            clients: HashMap::new(),
            // message_receiver: rx,
            // message_transmitter: tx,
            client_addresses: HashMap::new(),
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
                let header = serialize(&header).to_hex();
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
                    let header = serialize(&header).to_hex();
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
                    "hex": serialize(&header).to_hex()
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
                let transactions = self
                    .address_cache
                    .read()
                    .await
                    .get_address_history(&script_hash);
                let mut res = Vec::new();
                for transaction in transactions {
                    let entry = if transaction.height == 0 {
                        json!({
                            "tx_hash": transaction.hash.to_hex(),
                            "height": transaction.height,
                            "fee": 2000
                        })
                    } else {
                        json!({
                            "tx_hash": transaction.hash.to_hex(),
                            "height": transaction.height,
                        })
                    };

                    res.push(entry);
                }

                json_rpc_res!(request, res)
            }
            "blockchain.scripthash.get_mempool" => json_rpc_res!(request, []),
            "blockchain.scripthash.listunspent" => {
                let hash = get_arg!(request, sha256::Hash, 0);
                let utxos = self.address_cache.read().await.get_address_utxos(&hash);
                if utxos.is_none() {
                    return Err(crate::error::Error::InvalidParams);
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

                if history.is_empty() {
                    return json_rpc_res!(request, null);
                }
                let status_hash = get_status(history);
                json_rpc_res!(request, status_hash)
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

    pub async fn main_loop(mut self, address: String) -> Result<(), crate::error::Error> {
        let (tx, mut rx) = unbounded_channel::<Message>();
        let tcp_listener = TcpListener::bind(address).await?;

        tokio::spawn(client_accept_loop(tcp_listener, tx));

        let blocks = Channel::new();
        let blocks = Arc::new(blocks);

        self.chain.subscribe(blocks.clone());

        loop {
            for (block, height) in blocks.recv() {
                self.handle_block(block, height).await;
            }
            let request = async_std::future::timeout(std::time::Duration::from_secs(1), rx.recv())
                .await
                .unwrap();

            if let Some(message) = request {
                self.handle_message(message).await?;
            }
        }
    }

    async fn handle_block(&mut self, block: bitcoin::Block, height: u32) {
        let result = json!({
            "jsonrpc": "2.0",
            "method": "blockchain.headers.subscribe",
            "params": [{
                "height": height,
                "hex": serialize(&block.header).to_hex()
            }]
        });
        if !self.chain.is_in_idb() || height % 1000 == 0 {
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

                let status_hash = get_status(history);
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
    message_transmitter: UnboundedSender<Message>,
) -> Result<(), std::io::Error> {
    let mut _stream = &*client.stream;
    let mut lines = BufReader::new(_stream).lines();

    while let Some(Ok(line)) = lines.next().await {
        message_transmitter
            .send(Message::Message((client.client_id, line)))
            .expect("Main loop is broken");
    }

    info!("Lost client with ID: {}", client.client_id);

    message_transmitter
        .send(Message::Disconnect(client.client_id))
        .expect("Main loop is broken");

    Ok(())
}

/// Listens to new TCP connections in a loop
pub async fn client_accept_loop(
    listener: TcpListener,
    message_transmitter: UnboundedSender<Message>,
) {
    let mut id_count = 0;
    loop {
        if let Ok((stream, _addr)) = listener.accept().await {
            info!("New client connection");
            let stream = Arc::new(stream);
            let client = Arc::new(Client::new(id_count, stream));

            tokio::task::spawn(client_broker_loop(
                client.clone(),
                message_transmitter.clone(),
            ));

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
    ($request: ident, $arg_type: ty, $idx: literal) => {
        if let Some(arg) = $request.params.get($idx) {
            serde_json::from_value::<$arg_type>(arg.clone())?
        } else {
            return Err(super::error::Error::InvalidParams);
        }
    };
}

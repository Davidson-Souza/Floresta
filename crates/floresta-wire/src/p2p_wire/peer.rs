// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::fmt::Debug;
use core::fmt::Display;
use core::fmt::Formatter;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::Transaction;
use bitcoin::bip158::BlockFilter;
use bitcoin::block::Header as BlockHeader;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::encode;
use bitcoin::consensus::serialize;
use bitcoin::hashes::Hash;
use bitcoin::p2p::PROTOCOL_VERSION;
use bitcoin::p2p::ServiceFlags;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::message::CommandString;
use bitcoin::p2p::message::NetworkMessage;
use bitcoin::p2p::message_blockdata::Inventory;
use bitcoin::p2p::message_filter::CFHeaders;
use bitcoin::p2p::message_filter::GetCFHeaders;
use bitcoin::p2p::message_network::VersionMessage;
use floresta_common::impl_error_from;
use floresta_domain::mempool::MempoolBase;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::spawn;
use tokio::sync::Mutex;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::debug;
use tracing::error;
use tracing::warn;

use self::peer_utils::make_pong;
use super::node::NodeNotification;
use super::node::NodeRequest;
use super::transport::TransportError;
use super::transport::TransportProtocol;
use super::transport::WriteTransport;
use crate::address_man::LocalAddress;
use crate::block_proof::UtreexoProofMask;
use crate::node::ConnectionKind;
use crate::node::MAX_ADDRV2_ADDRESSES;
use crate::p2p_wire::block_proof::GetUtreexoProof;
use crate::p2p_wire::block_proof::UtreexoProof;
use crate::p2p_wire::transport::ReadTransport;

/// If we send a ping, and our peer takes more than PING_TIMEOUT to
/// reply, disconnect.
const PING_TIMEOUT: Duration = Duration::from_secs(30);

/// If the last message we've got was more than 60, send out a ping
const SEND_PING_TIMEOUT: Duration = Duration::from_secs(60);

/// The command string for the "utreexo proof" message
const UTREEXO_PROOF_CMD_STRING: &str = "uproof";

/// The command string for the "get utreexo proof" message
const GET_UTREEXO_PROOF_CMD: &str = "getuproof";

/// How many block announcements per inv a peer can send
const MAX_BLOCKS_PER_INV: u32 = 500;

/// To avoid being eclipsed with an address spam attack, we limit
/// the rate of addrv2 messages a peer can send us to one every
/// 10 seconds.
const ADDRV2_MESSAGE_INTERVAL: Duration = Duration::from_secs(10);

/// How long a node must wait to send another inv
const INV_MESSAGE_INTERVAL: Duration = Duration::from_secs(30); // 30 seconds

/// How many messages/sec a peer is allowed to send.
///
/// If a peer sends more than this, we disconnect it.
const MAX_MSGS_PER_SEC: u64 = 10_000;

/// The version for BIP158 basic filter type.
const BASIC_FILTER_VERSION: u8 = 0;

/// How many filter (or filter headers) are allowed in a single message.
const MAX_FILTERS_PER_MESSAGE: usize = 2_000;

#[derive(Debug, PartialEq)]
enum State {
    None,
    SentVersion(Instant),
    SentVerack,
    Connected,
}

pub struct MessageActor<R: AsyncRead + Unpin + Send> {
    pub transport: ReadTransport<R>,
    pub sender: UnboundedSender<ReaderMessage>,
}

impl<R: AsyncRead + Unpin + Send> MessageActor<R> {
    async fn inner(&mut self) -> std::result::Result<(), PeerError> {
        loop {
            let msg = self.transport.read_message().await?;
            let now = Instant::now();
            self.sender.send(ReaderMessage::Message(msg, now))?;
        }
    }

    pub async fn run(mut self) -> Result<()> {
        if let Err(err) = self.inner().await {
            self.sender.send(ReaderMessage::Error(err))?;
        }
        Ok(())
    }
}

pub fn create_actors<R: AsyncRead + Unpin + Send>(
    transport: ReadTransport<R>,
) -> (UnboundedReceiver<ReaderMessage>, MessageActor<R>) {
    let (actor_sender, actor_receiver) = unbounded_channel();
    let actor = MessageActor {
        transport,
        sender: actor_sender,
    };
    (actor_receiver, actor)
}

pub struct Peer<T: AsyncWrite + Unpin + Send + Sync> {
    mempool: Arc<Mutex<dyn MempoolBase>>,
    blocks_only: bool,
    services: ServiceFlags,
    time_offset: i64,
    user_agent: String,
    messages: u64,
    start_time: Instant,
    last_message: Instant,
    last_addrv2: Instant,
    last_inv: Instant,
    current_best_block: i32,
    last_ping: Option<Instant>,
    id: u32,
    node_tx: UnboundedSender<NodeNotification>,
    state: State,
    send_headers: bool,
    node_requests: UnboundedReceiver<NodeRequest>,
    address: LocalAddress,
    kind: ConnectionKind,
    wants_addrv2: bool,
    shutdown: bool,
    actor_receiver: UnboundedReceiver<ReaderMessage>, // Add the receiver for messages from TcpStreamActor
    writer: WriteTransport<T>,
    our_user_agent: String,
    our_best_block: u32,
    // This is kept as an option to avoid the need to keep the other half around during tests.
    cancellation_sender: Option<oneshot::Sender<()>>,
    transport_protocol: TransportProtocol,
}

#[derive(Debug)]
/// Enum for diverse variants of errors when dealing with a [`Peer`]
pub enum PeerError {
    /// Error while sending data to a peer
    Send,

    /// Error while reading data from a peer
    Read(std::io::Error),

    /// Error while parsing message
    Parse(encode::Error),

    /// Peer sent us a message that we aren't expecting
    UnexpectedMessage,

    /// Peer sent us a message that is too big
    MessageTooBig,

    /// Peer sent us a message with the wrong magic bits
    MagicBitsMismatch,

    /// Peer sent us too many messages in a short period of time
    TooManyMessages,

    /// Peer timed out a ping message
    PingTimeout,

    /// Channel error
    Channel,

    /// Transport error
    Transport(TransportError),
}

impl Display for PeerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Send => write!(f, "Error while sending to peer"),
            Self::Read(err) => write!(f, "Error while reading from peer: {err:?}"),
            Self::Parse(err) => write!(f, "Error while parsing message: {err:?}"),
            Self::UnexpectedMessage => {
                write!(f, "Peer sent us a message that we aren't expecting")
            }
            Self::MessageTooBig => write!(f, "Peer sent us a message that is too big"),
            Self::MagicBitsMismatch => {
                write!(f, "Peer sent us a message with the wrong magic bits")
            }
            Self::TooManyMessages => {
                write!(
                    f,
                    "Peer sent us too many messages in a short period of time"
                )
            }
            Self::PingTimeout => write!(f, "Peer timed out a ping"),
            Self::Channel => write!(f, "Channel error with empty data"),
            Self::Transport(err) => write!(f, "Transport error: {err:?}"),
        }
    }
}

impl_error_from!(PeerError, TransportError, Transport);
impl_error_from!(PeerError, std::io::Error, Read);
impl_error_from!(PeerError, encode::Error, Parse);

impl From<SendError<ReaderMessage>> for PeerError {
    fn from(_: SendError<ReaderMessage>) -> Self {
        Self::Channel
    }
}

pub enum ReaderMessage {
    Message(NetworkMessage, Instant),
    Error(PeerError),
}

impl<T: AsyncWrite + Unpin + Send + Sync> Debug for Peer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)?;
        Ok(())
    }
}

type Result<T> = std::result::Result<T, PeerError>;

impl<T: AsyncWrite + Unpin + Send + Sync> Peer<T> {
    pub async fn read_loop(mut self) -> Result<Self> {
        let result = self.peer_loop_inner().await;

        let now = Instant::now();
        self.send_to_node(PeerMessages::Disconnected(self.address.id), now);

        // force the stream to shutdown to prevent leaking resources
        if let Err(shutdown_err) = self.writer.shutdown().await {
            debug!(
                "Failed to shutdown writer for Peer {}: {shutdown_err:?}",
                self.id
            );
        }

        if let Some(Err(cancellation_err)) = self.cancellation_sender.take().map(|ch| ch.send(())) {
            debug!(
                "Failed to propagate cancellation signal for Peer {}: {cancellation_err:?}",
                self.id
            );
        }

        if let Err(e) = result {
            debug!("Peer {} connection loop closed: {e:?}", self.id);
            return Err(e);
        }

        Ok(self)
    }

    async fn peer_loop_inner(&mut self) -> Result<()> {
        // Send a `version` message to the peer.
        let message_version = peer_utils::build_version_message(
            self.our_user_agent.clone(),
            self.our_best_block,
            &self.address,
        );
        self.write(message_version).await?;
        self.state = State::SentVersion(Instant::now());
        loop {
            tokio::select! {
                request = tokio::time::timeout(Duration::from_secs(2), self.node_requests.recv()) => {
                    match request {
                        Ok(None) => {
                            return Err(PeerError::Channel);
                        },
                        Ok(Some(request)) => {
                            self.handle_node_request(request).await?;
                        },
                        Err(_) => {
                            // Timeout, do nothing
                        }
                    }
                },
                message = self.actor_receiver.recv() => {
                    match message {
                        None => {
                            return Err(PeerError::Channel);
                        }
                        Some(ReaderMessage::Error(e)) => {
                            return Err(e);
                        }
                        Some(ReaderMessage::Message(msg, time)) => {
                            self.handle_peer_message(msg, time).await?;
                        }
                    }
                }
            }

            if self.shutdown {
                return Ok(());
            }

            // If we send a ping and our peer doesn't respond in time, disconnect
            if let Some(when) = self.last_ping {
                if when.elapsed() > PING_TIMEOUT {
                    return Err(PeerError::PingTimeout);
                }
            }

            // Send a ping to check if this peer is still good
            let last_message = self.last_message.elapsed();
            if last_message > SEND_PING_TIMEOUT {
                if self.last_ping.is_some() {
                    continue;
                }
                let nonce = rand::random();
                self.last_ping = Some(Instant::now());
                self.write(NetworkMessage::Ping(nonce)).await?;
            }

            // divide the number of messages by the number of seconds we've been connected,
            // if it's more than 10 msg/sec, this peer is sending us too many messages, and we should
            // disconnect.
            let msg_sec = self
                .messages
                .checked_div(Instant::now().duration_since(self.start_time).as_secs())
                .unwrap_or(0);

            if msg_sec > MAX_MSGS_PER_SEC {
                error!(
                    "Peer {} is sending us too many messages, disconnecting",
                    self.id
                );
                return Err(PeerError::TooManyMessages);
            }

            if let State::SentVersion(when) = self.state {
                if Instant::now().duration_since(when) > Duration::from_secs(10) {
                    return Err(PeerError::UnexpectedMessage);
                }
            }
        }
    }

    pub async fn handle_node_request(&mut self, request: NodeRequest) -> Result<()> {
        assert_eq!(self.state, State::Connected);
        debug!("Handling node request: {request:?}");
        match request {
            NodeRequest::GetBlock(block_hashes) => {
                let inv = block_hashes
                    .iter()
                    .map(|block| Inventory::WitnessBlock(*block))
                    .collect();

                let _ = self.write(NetworkMessage::GetData(inv)).await;
            }
            NodeRequest::GetUtreexoState((block_hash, height)) => {
                let get_filter = bitcoin::p2p::message_filter::GetCFilters {
                    filter_type: 1,
                    start_height: height,
                    stop_hash: block_hash,
                };

                let _ = self.write(NetworkMessage::GetCFilters(get_filter)).await;
            }
            NodeRequest::GetHeaders(locator) => {
                let _ = self
                    .write(NetworkMessage::GetHeaders(
                        bitcoin::p2p::message_blockdata::GetHeadersMessage {
                            version: 0,
                            locator_hashes: locator,
                            stop_hash: BlockHash::all_zeros(),
                        },
                    ))
                    .await;
            }
            NodeRequest::Shutdown => {
                self.shutdown = true;
                self.writer.shutdown().await?;
            }
            NodeRequest::GetAddresses => {
                self.write(NetworkMessage::GetAddr).await?;
            }
            NodeRequest::BroadcastTransaction(tx) => {
                self.write(NetworkMessage::Inv(vec![Inventory::Transaction(tx)]))
                    .await?;
            }
            NodeRequest::MempoolTransaction(txid) => {
                self.write(NetworkMessage::GetData(vec![Inventory::Transaction(txid)]))
                    .await?;
            }
            NodeRequest::SendAddresses(addresses) => {
                self.write(NetworkMessage::AddrV2(addresses)).await?;
            }
            NodeRequest::GetFilter((stop_hash, start_height)) => {
                let get_filter = bitcoin::p2p::message_filter::GetCFilters {
                    filter_type: BASIC_FILTER_VERSION,
                    start_height,
                    stop_hash,
                };

                self.write(NetworkMessage::GetCFilters(get_filter)).await?;
            }
            NodeRequest::Ping => {
                let nonce = rand::random();
                self.last_ping = Some(Instant::now());
                self.write(NetworkMessage::Ping(nonce)).await?;
            }
            NodeRequest::GetBlockProof((block_hash, proof_hashes_bitmap, leaf_index_bitmap)) => {
                let get_block_proof = GetUtreexoProof {
                    block_hash,
                    request_bitmap: UtreexoProofMask::request_all(),
                    proof_hashes_bitmap,
                    leaf_index_bitmap,
                };

                self.write(NetworkMessage::Unknown {
                    command: CommandString::try_from_static(GET_UTREEXO_PROOF_CMD)
                        .expect("Invalid command string"),
                    payload: serialize(&get_block_proof),
                })
                .await?;
            }

            NodeRequest::GetCFHeaders {
                start_height,
                stop_hash,
            } => {
                let get_cfheaders = GetCFHeaders {
                    filter_type: BASIC_FILTER_VERSION,
                    start_height,
                    stop_hash,
                };

                self.write(NetworkMessage::GetCFHeaders(get_cfheaders))
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn handle_peer_message(
        &mut self,
        message: NetworkMessage,
        time: Instant,
    ) -> Result<()> {
        self.last_message = time;
        self.messages += 1;

        debug!("Received {} from peer {}", message.command(), self.id);
        match self.state {
            State::Connected => match message {
                NetworkMessage::Inv(inv) => {
                    let mut block_inv_elements = 0;

                    // Silently drop
                    if self.last_inv.elapsed() < INV_MESSAGE_INTERVAL {
                        return Ok(());
                    }

                    self.last_inv = Instant::now();

                    for inv_entry in inv {
                        match inv_entry {
                            Inventory::Error => {}
                            Inventory::Transaction(_) => {}
                            Inventory::Block(block_hash)
                            | Inventory::WitnessBlock(block_hash)
                            | Inventory::CompactBlock(block_hash) => {
                                block_inv_elements += 1;
                                if block_inv_elements >= MAX_BLOCKS_PER_INV {
                                    return Err(PeerError::MessageTooBig);
                                }

                                self.send_to_node(PeerMessages::NewBlock(block_hash), time);
                            }
                            _ => {}
                        }
                    }
                }
                NetworkMessage::GetHeaders(_) => {
                    self.write(NetworkMessage::Headers(Vec::new())).await?;
                }
                NetworkMessage::Headers(headers) => {
                    self.send_to_node(PeerMessages::Headers(headers), time);
                }
                NetworkMessage::SendHeaders => {
                    self.send_headers = true;
                    self.write(NetworkMessage::SendHeaders).await?;
                }
                NetworkMessage::Ping(nonce) => {
                    self.handle_ping(nonce).await?;
                }
                NetworkMessage::FeeFilter(_) => {
                    self.write(NetworkMessage::FeeFilter(1000)).await?;
                }
                NetworkMessage::AddrV2(addresses) => {
                    // As per BIP 155, limit the number of addresses to 1,000
                    if addresses.len() > MAX_ADDRV2_ADDRESSES {
                        return Err(PeerError::MessageTooBig);
                    }

                    // Rate limit addrv2 messages
                    let now = Instant::now();
                    let elapsed = now.duration_since(self.last_addrv2);
                    self.last_addrv2 = Instant::now();

                    if elapsed < ADDRV2_MESSAGE_INTERVAL {
                        debug!(
                            "Peer {} sent addrv2 messages too frequently, ignoring",
                            self.id
                        );

                        // just drop the message
                        return Ok(());
                    }

                    self.send_to_node(PeerMessages::Addr(addresses), time);
                }
                NetworkMessage::GetBlocks(_) => {
                    self.write(NetworkMessage::Inv(Vec::new())).await?;
                }
                NetworkMessage::GetAddr => {
                    if self.wants_addrv2 {
                        self.write(NetworkMessage::AddrV2(Vec::new())).await?;
                        return Ok(());
                    }
                    self.write(NetworkMessage::Addr(Vec::new())).await?;
                }
                NetworkMessage::GetData(inv) => {
                    for inv_el in inv {
                        self.handle_get_data(inv_el).await?;
                    }
                }
                NetworkMessage::Tx(tx) => {
                    self.send_to_node(PeerMessages::Transaction(tx), time);
                }
                NetworkMessage::NotFound(inv) => {
                    for inv_el in inv {
                        self.send_to_node(PeerMessages::NotFound(inv_el), time);
                    }
                }
                NetworkMessage::SendAddrV2 => {
                    warn!("Peer {} sent SendAddrV2 after handshake completed", self.id);
                    return Err(PeerError::UnexpectedMessage);
                }
                NetworkMessage::Pong(_) => {
                    self.last_ping = None;
                }
                NetworkMessage::Unknown { command, payload } => {
                    let utreexo_proof_cmd =
                        CommandString::try_from_static(UTREEXO_PROOF_CMD_STRING)
                            .expect("Invalid command string");

                    if command != utreexo_proof_cmd {
                        warn!("Unknown command string: {command}");
                        return Ok(());
                    }

                    let utreexo_proof: UtreexoProof = deserialize(&payload)?;
                    self.send_to_node(PeerMessages::UtreexoProof(utreexo_proof), time);

                    return Ok(());
                }
                NetworkMessage::Block(block) => {
                    self.send_to_node(PeerMessages::Block(block), time);
                }
                NetworkMessage::CFilter(filter_msg) => match filter_msg.filter_type {
                    0 => {
                        let filter = BlockFilter::new(&filter_msg.filter);

                        self.send_to_node(
                            PeerMessages::BlockFilter((filter_msg.block_hash, filter)),
                            time,
                        );
                    }
                    1 => {
                        self.send_to_node(PeerMessages::UtreexoState(filter_msg.filter), time);
                    }
                    _ => {}
                },

                NetworkMessage::CFHeaders(cfheaders) => {
                    if cfheaders.filter_hashes.len() > MAX_FILTERS_PER_MESSAGE {
                        return Err(PeerError::MessageTooBig);
                    }

                    if cfheaders.filter_type != BASIC_FILTER_VERSION {
                        warn!("Unknown filter header type {}", cfheaders.filter_type);
                        return Err(PeerError::UnexpectedMessage);
                    }

                    self.send_to_node(PeerMessages::CFHeaders(cfheaders), time);
                }

                // Explicitly ignore these messages, if something changes in the future
                // this would cause a compile error.
                NetworkMessage::Verack
                | NetworkMessage::Version(_)
                | NetworkMessage::WtxidRelay
                | NetworkMessage::Reject(_)
                | NetworkMessage::Alert(_)
                | NetworkMessage::BlockTxn(_)
                | NetworkMessage::CFCheckpt(_)
                | NetworkMessage::CmpctBlock(_)
                | NetworkMessage::FilterAdd(_)
                | NetworkMessage::FilterClear
                | NetworkMessage::FilterLoad(_)
                | NetworkMessage::GetBlockTxn(_)
                | NetworkMessage::GetCFCheckpt(_)
                | NetworkMessage::GetCFHeaders(_)
                | NetworkMessage::Addr(_)
                | NetworkMessage::GetCFilters(_)
                | NetworkMessage::MemPool
                | NetworkMessage::MerkleBlock(_)
                | NetworkMessage::SendCmpct(_) => {}
            },
            State::None | State::SentVersion(_) => match message {
                bitcoin::p2p::message::NetworkMessage::Version(version) => {
                    self.handle_version(version).await?;
                }
                _ => {
                    warn!("unexpected message: {:?} from peer {}", message, self.id);
                    return Err(PeerError::UnexpectedMessage);
                }
            },
            State::SentVerack => match message {
                bitcoin::p2p::message::NetworkMessage::Verack => {
                    self.state = State::Connected;
                    self.send_to_node(
                        PeerMessages::Ready(Version {
                            user_agent: self.user_agent.clone(),
                            protocol_version: 0,
                            id: self.id,
                            blocks: self.current_best_block.unsigned_abs(),
                            address_id: self.address.id,
                            services: self.services,
                            time_offset: self.time_offset,
                            kind: self.kind,
                            transport_protocol: self.transport_protocol,
                        }),
                        time,
                    );
                }
                bitcoin::p2p::message::NetworkMessage::SendAddrV2 => {
                    self.wants_addrv2 = true;
                }
                bitcoin::p2p::message::NetworkMessage::SendHeaders => {
                    self.send_headers = true;
                }
                bitcoin::p2p::message::NetworkMessage::WtxidRelay => {}
                _ => {
                    warn!("unexpected message: {:?} from peer {}", message, self.id);
                    return Err(PeerError::UnexpectedMessage);
                }
            },
        }
        Ok(())
    }
}

impl<T: AsyncWrite + Unpin + Send + Sync> Peer<T> {
    pub async fn write(&mut self, msg: NetworkMessage) -> Result<()> {
        debug!("Writing {} to peer {}", msg.command(), self.id);
        self.writer.write_message(msg).await?;
        Ok(())
    }

    pub async fn handle_get_data(&mut self, inv: Inventory) -> Result<()> {
        match inv {
            Inventory::WitnessTransaction(txid) => {
                let tx = self.mempool.lock().await.get_from_mempool(txid).cloned();
                if let Some(tx) = tx {
                    self.write(NetworkMessage::Tx(tx)).await?;
                }
            }
            Inventory::Transaction(txid) => {
                let tx = self.mempool.lock().await.get_from_mempool(txid).cloned();
                if let Some(tx) = tx {
                    self.write(NetworkMessage::Tx(tx)).await?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_peer<W: AsyncWrite + Unpin + Send + Sync + 'static>(
        id: u32,
        address: LocalAddress,
        mempool: Arc<Mutex<dyn MempoolBase>>,
        node_tx: UnboundedSender<NodeNotification>,
        node_requests: UnboundedReceiver<NodeRequest>,
        kind: ConnectionKind,
        actor_receiver: UnboundedReceiver<ReaderMessage>,
        writer: WriteTransport<W>,
        our_user_agent: String,
        our_best_block: u32,
        cancellation_sender: tokio::sync::oneshot::Sender<()>,
        transport_protocol: TransportProtocol,
    ) -> JoinHandle<Result<Peer<W>>> {
        let peer = Peer {
            address,
            blocks_only: false,
            current_best_block: -1,
            id,
            mempool,
            last_ping: None,
            last_message: Instant::now(),
            last_inv: Instant::now() - INV_MESSAGE_INTERVAL,
            last_addrv2: Instant::now() - ADDRV2_MESSAGE_INTERVAL,
            node_tx,
            services: ServiceFlags::NONE,
            time_offset: 0,
            messages: 0,
            start_time: Instant::now(),
            user_agent: "".into(),
            state: State::None,
            send_headers: false,
            node_requests,
            kind,
            wants_addrv2: false,
            shutdown: false,
            actor_receiver, // Add the receiver for messages from TcpStreamActor
            writer,
            our_user_agent,
            our_best_block,
            cancellation_sender: Some(cancellation_sender),
            transport_protocol,
        };

        spawn(peer.read_loop())
    }

    async fn handle_ping(&mut self, nonce: u64) -> Result<()> {
        let pong = make_pong(nonce);
        self.write(pong).await
    }

    async fn handle_version(&mut self, version: VersionMessage) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should not be before UNIX epoch")
            .as_secs() as i64;
        self.time_offset = version.timestamp.saturating_sub(now);
        self.user_agent = version.user_agent;
        self.blocks_only = !version.relay;
        self.current_best_block = version.start_height;
        self.services = version.services;
        if version.version >= PROTOCOL_VERSION {
            self.write(NetworkMessage::SendAddrV2).await?;
        }
        self.state = State::SentVerack;
        let verack = NetworkMessage::Verack;
        self.state = State::SentVerack;
        self.write(verack).await
    }

    fn send_to_node(&self, message: PeerMessages, time: Instant) {
        let message = NodeNotification::FromPeer(self.id, message, time);
        let _ = self.node_tx.send(message);
    }
}

pub(super) mod peer_utils {
    use core::net::IpAddr;
    use core::net::Ipv4Addr;
    use core::net::SocketAddr;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    use bitcoin::p2p::Address;
    use bitcoin::p2p::message::NetworkMessage;
    use bitcoin::p2p::message_network::VersionMessage;
    use floresta_common::PROTOCOL_VERSION;
    use floresta_common::advertised_services;
    use rand::RngExt;
    use rand::rng;

    use crate::address_man::LocalAddress;

    /// Build the [pong](NetworkMessage::Pong) message, which must be sent whenever a peer sends us a
    /// [ping](NetworkMessage::Ping). Note that the nonce received in the ping must be reused in the pong.
    pub(super) fn make_pong(nonce: u64) -> NetworkMessage {
        NetworkMessage::Pong(nonce)
    }

    /// Build the [version](NetworkMessage::Version) message used to perform the peer connection
    /// handshake, as described in the [Bitcoin Wiki](https://en.bitcoin.it/wiki/Protocol_documentation#version).
    pub(crate) fn build_version_message(
        user_agent: String,
        best_block: u32,
        peer_address: &LocalAddress,
    ) -> NetworkMessage {
        // The set of services supported by this node.
        let services = advertised_services();

        // The current UNIX timestamp.
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Great Scott!")
            .as_secs() as i64;

        // This node's `Address`.
        // Per the version message specification, we can use a dummy address.
        let fake_socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38332);
        let sender_address = Address::new(&fake_socket, services);

        // The remote peer's `Address`.
        let peer_addr = peer_address.get_socket_addr().unwrap_or(fake_socket);
        let receiver_address = Address::new(&peer_addr, peer_address.get_services());

        // Generate a per-message nonce.
        let mut prng = rng();
        let nonce: u64 = prng.random();

        // Inform the peer of this node's chain tip.
        let start_height = best_block as i32;

        // Floresta does not implement transaction relay.
        let relay = false;

        NetworkMessage::Version(VersionMessage {
            version: PROTOCOL_VERSION,
            services,
            timestamp,
            sender: sender_address,
            receiver: receiver_address,
            nonce,
            user_agent,
            start_height,
            relay,
        })
    }
}

#[derive(Debug)]
pub struct Version {
    pub user_agent: String,
    pub protocol_version: u32,
    pub blocks: u32,
    pub id: u32,
    pub address_id: usize,
    pub services: ServiceFlags,
    pub time_offset: i64,
    pub kind: ConnectionKind,
    pub transport_protocol: TransportProtocol,
}

/// Messages passed from different modules to the main node to process. They should minimal
/// and only if it requires global states, everything else should be handled by the module
/// itself.
#[derive(Debug)]
pub enum PeerMessages {
    /// A new block just arrived, we should ask for it and update our chain
    NewBlock(BlockHash),

    /// We got a full block from our peer, presumptively we asked for it
    Block(Block),

    /// A response to a `getheaders` request
    Headers(Vec<BlockHeader>),

    /// We got some p2p addresses, add this to our local database
    Addr(Vec<AddrV2Message>),

    /// Peer notify its readiness
    Ready(Version),

    /// Remote peer disconnected
    Disconnected(usize),

    /// Remote peer doesn't know the data we asked for
    NotFound(Inventory),

    /// Remote peer sent us a transaction
    Transaction(Transaction),

    /// Remote peer sent us a Utreexo state
    UtreexoState(Vec<u8>),

    /// Remote peer sent us a compact block filter
    BlockFilter((BlockHash, BlockFilter)),

    /// Remote peer sent us a Utreexo proof,
    UtreexoProof(UtreexoProof),

    /// Remote peer sent us compact block filter headers
    CFHeaders(CFHeaders),
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;
    use std::sync::Arc;
    use std::time::Duration;
    use std::time::Instant;

    use bitcoin::Network;
    use bitcoin::p2p::ServiceFlags;
    use bitcoin::p2p::address::AddrV2;
    use bitcoin::p2p::message::NetworkMessage;
    use floresta_mempool::Mempool;
    use tokio::sync::Mutex;
    use tokio::sync::mpsc::UnboundedReceiver;
    use tokio::sync::mpsc::UnboundedSender;
    use tokio::sync::mpsc::unbounded_channel;
    use tokio::sync::oneshot;

    use crate::TransportProtocol;
    use crate::address_man::AddressState;
    use crate::address_man::LocalAddress;
    use crate::bitcoin_socket_addr::BitcoinSocketAddr;
    use crate::node::ConnectionKind;
    use crate::node::NodeNotification;
    use crate::node::NodeRequest;
    use crate::p2p_wire::peer::Peer;
    use crate::p2p_wire::peer::PeerError;
    use crate::p2p_wire::peer::ReaderMessage;
    use crate::p2p_wire::peer::State;
    use crate::p2p_wire::peer::peer_utils;
    use crate::p2p_wire::transport::WriteTransport;
    use crate::p2p_wire::transport::test_transport::Writer;

    /// All the data needed to run a test.
    struct SetupData {
        /// The actual peer, it should be spawned and the future must not be dropped.
        peer: Peer<Writer>,

        /// This is used to send a message to a peer, mimicking a real network message.
        actor_sender: UnboundedSender<ReaderMessage>,

        /// Channel used to send requests to a peer, this will mimic the `UtreexoNode` sending
        /// something to our peer.
        node_sender: UnboundedSender<NodeRequest>,

        /// This is the opposite of node_sender, when a peer receives a message, you can read it
        /// here.
        node_receiver: UnboundedReceiver<NodeNotification>,
    }

    fn create_peer() -> SetupData {
        let (node_tx, node_receiver) = unbounded_channel();
        let (node_sender, node_requests) = unbounded_channel();
        let (actor_sender, actor_receiver) = unbounded_channel();
        let (cancellation_sender, _) = oneshot::channel();

        let address = LocalAddress::new(
            BitcoinSocketAddr::new(AddrV2::Ipv4(Ipv4Addr::new(127, 0, 0, 1)), 8333),
            0,
            AddressState::NeverTried,
            ServiceFlags::NONE,
            0,
        );

        let peer = Peer {
            address,
            our_best_block: 0,
            writer: WriteTransport::V1(Writer, Network::Regtest),
            state: State::Connected,
            kind: ConnectionKind::Manual,
            id: 0,
            mempool: Arc::new(Mutex::new(Mempool::new(1000))),
            node_tx,
            services: ServiceFlags::NONE,
            time_offset: 0,
            messages: 0,
            shutdown: false,
            last_ping: Some(Instant::now()),
            user_agent: "/Mock-Peer:0.0.0/".into(),
            start_time: Instant::now(),
            blocks_only: true,
            last_addrv2: Instant::now(),
            last_message: Instant::now(),
            last_inv: Instant::now(),
            send_headers: true,
            wants_addrv2: true,
            node_requests,
            actor_receiver,
            our_user_agent: "/Floresta-test:0.0.0/".into(),
            current_best_block: 0,
            transport_protocol: TransportProtocol::V1,
            cancellation_sender: Some(cancellation_sender),
        };

        SetupData {
            peer,
            actor_sender,
            node_sender,
            node_receiver,
        }
    }

    fn send_to_peer(
        actor_sender: &mut UnboundedSender<ReaderMessage>,
        network_message: NetworkMessage,
    ) {
        actor_sender
            .send(ReaderMessage::Message(network_message, Instant::now()))
            .unwrap();
    }

    #[tokio::test]
    async fn test_unexpected_message_handshake() {
        let SetupData {
            peer,
            mut actor_sender,
            node_receiver,
            node_sender,
        } = create_peer();

        let fut = tokio::spawn(peer.read_loop());

        // Send a ping before the handshake completes
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(0));

        let err = fut.await.unwrap().unwrap_err();
        assert!(matches!(err, PeerError::UnexpectedMessage));

        // Prevents those channels from being dropped, so we don't get a `Channel` error
        drop(node_receiver);
        drop(node_sender);
    }

    #[tokio::test]
    async fn test_increment_peer_messages() {
        let SetupData {
            peer,
            mut actor_sender,
            node_receiver,
            node_sender,
        } = create_peer();
        let address = peer.address.clone();
        let fut = tokio::spawn(peer.read_loop());

        send_to_peer(
            &mut actor_sender,
            peer_utils::build_version_message("/Floresta-test:0.0.0/".into(), 0, &address),
        );

        send_to_peer(&mut actor_sender, NetworkMessage::Verack);

        send_to_peer(&mut actor_sender, NetworkMessage::Ping(2));
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(3));
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(4));
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(5));
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(6));
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(7));
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(8));
        send_to_peer(&mut actor_sender, NetworkMessage::Ping(9));

        // give the peer a little time to process everything
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Asks the peer to shutdown
        node_sender.send(NodeRequest::Shutdown).unwrap();

        let peer = fut.await.unwrap().unwrap();
        assert_eq!(peer.messages, 10);

        // Prevents those channels from being dropped, so we don't get a `Channel` error
        drop(node_receiver);
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::fmt::Debug;
use core::fmt::Display;
use core::fmt::Formatter;
use std::io;

use bip324::Role;
use bip324::futures::Protocol;
use bip324::futures::ProtocolReader;
use bip324::futures::ProtocolWriter;
use bip324::io::Payload;
use bip324::io::ProtocolError;
use bip324::io::ProtocolFailureSuggestion;
use bitcoin::Network;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::Encodable;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::deserialize_partial;
use bitcoin::consensus::encode;
use bitcoin::consensus::serialize;
use bitcoin::hashes::Hash;
use bitcoin::hashes::sha256d;
use bitcoin::hex::DisplayHex;
use bitcoin::p2p::Magic;
use bitcoin::p2p::message::CommandString;
use bitcoin::p2p::message::MAX_MSG_SIZE;
use bitcoin::p2p::message::NetworkMessage;
use bitcoin::p2p::message::RawNetworkMessage;
use floresta_common::impl_error_from;
use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::io::ReadHalf;
use tokio::io::WriteHalf;
use tokio::net::TcpStream;
use tokio::net::ToSocketAddrs;
use tracing::debug;
use tracing::error;

use super::network_message_ext::NetworkMessageExt;
use super::network_message_ext::V2MessageError;
use super::socks::Socks5Addr;
use super::socks::Socks5Error;
use super::socks::Socks5StreamBuilder;
use crate::address_man::LocalAddress;

type TcpReadTransport = ReadTransport<BufReader<ReadHalf<TcpStream>>>;
type TcpWriteTransport = WriteTransport<WriteHalf<TcpStream>>;
type TransportResult =
    Result<(TcpReadTransport, TcpWriteTransport, TransportProtocol), TransportError>;

#[derive(Copy, Clone, PartialEq, Eq)]
/// A wrapper type for a network checksum
///
/// This checksum accompanies every P2PV1 message to detect corruption.
/// Computed as the first 4 bytes of `SHA-265d(<msg_payload>)`.
pub struct P2PV1MessageChecksum([u8; 4]);

impl Display for P2PV1MessageChecksum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.as_hex())
    }
}

impl Debug for P2PV1MessageChecksum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl AsRef<[u8]> for P2PV1MessageChecksum {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl P2PV1MessageChecksum {
    pub fn from_payload(payload: &[u8]) -> Self {
        // Compute the `SHA-256d` digest of the message payload.
        let hash = sha256d::Hash::hash(payload);

        // The checksum is the first 4 bytes of the digest.
        let mut checksum = [0; 4];
        checksum.copy_from_slice(&hash.as_byte_array()[0..4]);
        Self(checksum)
    }
}

#[derive(Debug)]
/// Enum that deals with transport errors
pub enum TransportError {
    /// I/O error
    Io(io::Error),

    /// V2 protocol error
    Protocol(ProtocolError),

    /// V2 message serialization error
    SerdeV2(V2MessageError),

    /// V1 serde error
    SerdeV1(encode::Error),

    /// Proxy error
    Proxy(Socks5Error),

    /// Message is too big
    OversizedMessage {
        max_size: usize,
        message_size: usize,
    },

    /// Peer sent us a corrupted message
    BadChecksum {
        expected: P2PV1MessageChecksum,
        provided: P2PV1MessageChecksum,
    },

    /// Peer sent us a message with invalid magic bits
    BadMagicBits { expected: Magic, provided: Magic },

    /// Received address is invalid/unreachable
    InvalidAddress,
}

impl Display for TransportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "IO error: {err:?}"),
            Self::Protocol(err) => write!(f, "V2 protocol error: {err:?}"),
            Self::SerdeV2(err) => write!(f, "V2 serde error: {err:?}"),
            Self::SerdeV1(err) => write!(f, "V1 serde error: {err:?}"),
            Self::Proxy(err) => write!(f, "Proxy error: {err:?}"),
            Self::OversizedMessage {
                max_size,
                message_size,
            } => write!(
                f,
                "Peer sent us an oversized message: size {message_size} is greater than the max of {max_size}"
            ),
            Self::BadChecksum { expected, provided } => write!(
                f,
                "Peer sent us a corrupted message: expected {expected}, got {provided}"
            ),
            Self::BadMagicBits { expected, provided } => {
                write!(
                    f,
                    "Peer sent us a message with invalid magic bits: expected {expected}, got {provided}"
                )
            }
            Self::InvalidAddress => {
                write!(f, "provided address is either invalid or unreachable")
            }
        }
    }
}

impl_error_from!(TransportError, io::Error, Io);
impl_error_from!(TransportError, ProtocolError, Protocol);
impl_error_from!(TransportError, V2MessageError, SerdeV2);
impl_error_from!(TransportError, encode::Error, SerdeV1);
impl_error_from!(TransportError, Socks5Error, Proxy);

pub enum ReadTransport<R: AsyncRead + Unpin + Send> {
    V1(R, Network),
    V2(ProtocolReader<R>),
}

pub enum WriteTransport<W: AsyncWrite + Unpin + Send + Sync> {
    V1(W, Network),
    V2(ProtocolWriter<W>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Bitcoin nodes can communicate using different transport layer protocols.
pub enum TransportProtocol {
    /// Encrypted V2 protocol defined in BIP-324.
    V2,

    /// Original unencrypted V1 protocol.
    V1,
}

struct V1MessageHeader {
    magic: Magic,
    command: [u8; 12],
    length: u32,
    checksum: P2PV1MessageChecksum,
}

impl Decodable for V1MessageHeader {
    fn consensus_decode<R: bitcoin::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, encode::Error> {
        let magic = Magic::consensus_decode(reader)?;
        let command = <[u8; 12]>::consensus_decode(reader)?;
        let length = u32::consensus_decode(reader)?;
        let checksum = <[u8; 4]>::consensus_decode(reader)?;

        Ok(Self {
            magic,
            command,
            length,
            checksum: P2PV1MessageChecksum(checksum),
        })
    }
}

impl Encodable for V1MessageHeader {
    fn consensus_encode<W: bitcoin::io::Write + ?Sized>(
        &self,
        writer: &mut W,
    ) -> Result<usize, bitcoin::io::Error> {
        let mut size = 0;
        size += self.magic.consensus_encode(writer)?;
        size += self.command.consensus_encode(writer)?;
        size += self.length.consensus_encode(writer)?;
        size += self.checksum.0.consensus_encode(writer)?;

        Ok(size)
    }
}

/// Establishes a TCP connection and negotiates the bitcoin protocol.
///
/// This function tries to connect to the specified address and negotiate the bitcoin protocol
/// with the remote node. It first attempts to use the V2 protocol, and if that fails with a specific
/// error suggesting fallback to V1 protocol (and `allow_v1_fallback` is true), it will retry
/// the connection with the V1 protocol.
///
/// # Arguments
///
/// * `address` - The address of a target node
/// * `network` - The bitcoin network
/// * `allow_v1_fallback` - Whether to allow fallback to V1 protocol if V2 negotiation fails
///
/// # Returns
///
/// Returns a tuple of read and write transports that can be used to communicate with the node.
///
/// # Errors
///
/// Returns a `TransportError` if the connection cannot be established or protocol negotiation fails.
pub async fn connect<A: ToSocketAddrs>(
    address: A,
    network: Network,
    allow_v1_fallback: bool,
) -> TransportResult {
    match try_connection(&address, network, false).await {
        Ok(transport) => Ok(transport),
        Err(TransportError::Protocol(ProtocolError::Io(_, ProtocolFailureSuggestion::RetryV1)))
            if allow_v1_fallback =>
        {
            try_connection(&address, network, true).await
        }
        Err(e) => Err(e),
    }
}

async fn try_connection<A: ToSocketAddrs>(
    address: &A,
    network: Network,
    force_v1: bool,
) -> TransportResult {
    let tcp_stream = TcpStream::connect(address).await?;
    // Data is buffered until there is enough to send out
    // thus reducing the amount of packages going through
    // the network.
    tcp_stream.set_nodelay(false)?;

    let peer_addr = match tcp_stream.peer_addr() {
        Ok(addr) => addr.to_string(),
        Err(_) => String::from("unknown peer"),
    };
    let (reader, writer) = tokio::io::split(tcp_stream);
    let reader = BufReader::new(reader);

    match force_v1 {
        true => {
            debug!("Established a P2PV1 connection with peer={peer_addr}");
            Ok((
                ReadTransport::V1(reader, network),
                WriteTransport::V1(writer, network),
                TransportProtocol::V1,
            ))
        }
        false => match Protocol::new(network.magic(), Role::Initiator, None, None, reader, writer)
            .await
        {
            Ok(protocol) => {
                debug!("Established a P2PV2 connection with peer={peer_addr}");
                let (reader_protocol, writer_protocol) = protocol.into_split();
                Ok((
                    ReadTransport::V2(reader_protocol),
                    WriteTransport::V2(writer_protocol),
                    TransportProtocol::V2,
                ))
            }
            Err(e) => {
                debug!("Failed to establish a P2PV2 connection with peer={peer_addr}: {e:?}");
                Err(TransportError::Protocol(e))
            }
        },
    }
}

/// Establishes a connection through a SOCKS5 proxy and negotiates the bitcoin protocol.
///
/// This function connects to a SOCKS5 proxy, establishes a connection to the target address
/// through the proxy, and then negotiates the bitcoin protocol. Like `connect`, it first tries
/// the V2 protocol and can fall back to V1 if needed and allowed.
///
/// # Arguments
///
/// * `proxy_addr` - The address of the SOCKS5 proxy
/// * `address` - The target address to connect to through the proxy
/// * `port` - The port to connect to on the target
/// * `network` - The bitcoin network
/// * `allow_v1_fallback` - Whether to allow fallback to V1 protocol if V2 negotiation fails
///
/// # Returns
///
/// Returns a tuple of read and write transports that can be used to communicate with the node.
///
/// # Errors
///
/// Returns a `TransportError` if the proxy connection cannot be established, the connection
/// to the target fails, or protocol negotiation fails.
pub async fn connect_proxy<A: ToSocketAddrs + Clone + Debug>(
    proxy_addr: A,
    address: LocalAddress,
    network: Network,
    allow_v1_fallback: bool,
) -> TransportResult {
    let addr = Socks5Addr::try_from(&address)?;

    match try_proxy_connection(&proxy_addr, &addr, address.get_port(), network, false).await {
        Ok(transport) => Ok(transport),
        Err(TransportError::Protocol(ProtocolError::Io(_, ProtocolFailureSuggestion::RetryV1)))
            if allow_v1_fallback =>
        {
            try_proxy_connection(&proxy_addr, &addr, address.get_port(), network, true).await
        }
        Err(e) => Err(e),
    }
}

async fn try_proxy_connection<A: ToSocketAddrs + Clone + Debug>(
    proxy_addr: A,
    target_addr: &Socks5Addr,
    port: u16,
    network: Network,
    force_v1: bool,
) -> TransportResult {
    let proxy = TcpStream::connect(proxy_addr.clone()).await?;
    let stream = Socks5StreamBuilder::connect(proxy, target_addr, port).await?;
    let (reader, writer) = tokio::io::split(stream);
    let reader = BufReader::new(reader);
    match force_v1 {
        true => {
            debug!(
                "Established a P2PV1 connection over SOCKS5 using proxy={proxy_addr:?} with peer={target_addr:?}"
            );
            Ok((
                ReadTransport::V1(reader, network),
                WriteTransport::V1(writer, network),
                TransportProtocol::V1,
            ))
        }
        false => match Protocol::new(network.magic(), Role::Initiator, None, None, reader, writer)
            .await
        {
            Ok(protocol) => {
                debug!(
                    "Established a P2PV2 connection over SOCKS5 using proxy={proxy_addr:?} with peer={target_addr:?}"
                );
                let (reader_protocol, writer_protocol) = protocol.into_split();
                Ok((
                    ReadTransport::V2(reader_protocol),
                    WriteTransport::V2(writer_protocol),
                    TransportProtocol::V2,
                ))
            }
            Err(e) => {
                error!(
                    "Failed to establish a P2PV2 connection over SOCKS5 using proxy={proxy_addr:?} with peer={target_addr:?}: {e:?}"
                );
                Err(TransportError::Protocol(e))
            }
        },
    }
}

impl<R> ReadTransport<R>
where
    R: AsyncRead + Unpin + Send,
{
    /// Read the next [`NetworkMessage`] from the transport's [`ProtocolReader`] buffer.
    pub async fn read_message(&mut self) -> Result<NetworkMessage, TransportError> {
        match self {
            Self::V2(protocol) => {
                let payload = protocol.read().await?;
                let contents = payload.contents();

                Ok(NetworkMessage::deserialize_v2(contents)?)
            }
            Self::V1(reader, network) => {
                let mut data: Vec<u8> = vec![0; 24];
                reader.read_exact(&mut data).await?;

                let header: V1MessageHeader = deserialize_partial(&data)?.0;
                if header.length as usize > MAX_MSG_SIZE {
                    return Err(TransportError::OversizedMessage {
                        max_size: MAX_MSG_SIZE,
                        message_size: header.length as usize,
                    });
                }

                if header.magic != network.magic() {
                    return Err(TransportError::BadMagicBits {
                        provided: header.magic,
                        expected: network.magic(),
                    });
                }

                data.resize(24 + header.length as usize, 0);
                reader.read_exact(&mut data[24..]).await?;

                let checksum = P2PV1MessageChecksum::from_payload(&data[24..]);
                if header.checksum != checksum {
                    return Err(TransportError::BadChecksum {
                        expected: checksum,
                        provided: header.checksum,
                    });
                }

                let msg: RawNetworkMessage = deserialize(&data)?;
                Ok(msg.into_payload())
            }
        }
    }
}

impl<W> WriteTransport<W>
where
    W: AsyncWrite + Unpin + Send + Sync,
{
    /// Write a [`NetworkMessage`] to the transport's [`ProtocolWriter`] buffer.
    pub async fn write_message(&mut self, message: NetworkMessage) -> Result<(), TransportError> {
        match self {
            Self::V2(protocol) => {
                let data = message.serialize_v2();
                protocol.write(&Payload::genuine(data)).await?;
            }
            Self::V1(writer, network) => {
                if let NetworkMessage::Unknown { payload, command } = message {
                    let expected_cmd = CommandString::try_from_static("getuproof").unwrap();
                    assert_eq!(
                        command, expected_cmd,
                        "Only getuproof is supported as unknown message"
                    );

                    // FIXME: This little bit of ugliness is due to https://github.com/rust-bitcoin/rust-bitcoin/issues/4413
                    // Once that is solved upstream (or utreexo messages are added to
                    // rust-bitcoin), this can be removed.
                    let checksum = P2PV1MessageChecksum::from_payload(&payload);

                    let mut message_header = [0u8; 24];
                    message_header[0..4].copy_from_slice(&network.magic().to_bytes());
                    message_header[4..13].copy_from_slice("getuproof".as_bytes());
                    message_header[16..20].copy_from_slice(&(payload.len() as u32).to_le_bytes());
                    message_header[20..24].copy_from_slice(checksum.as_ref());

                    writer.write_all(&message_header).await?;
                    writer.write_all(&payload).await?;
                    writer.flush().await?;
                    return Ok(());
                }

                let data = &mut RawNetworkMessage::new(network.magic(), message);
                let data = serialize(&data);
                writer.write_all(&data).await?;
                writer.flush().await?;
            }
        }
        Ok(())
    }

    /// Shutdown the transport.
    pub async fn shutdown(&mut self) -> Result<(), TransportError> {
        match self {
            // The V2 transport does not require an explicit `writer.shutdown()` call,
            // since the buffer is already flushed internally on each `write()` call.
            Self::V2(_) => {}
            Self::V1(writer, _) => {
                writer.shutdown().await?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
/// Helper methods for writing tests involving transports
///
/// This module defines dummy transports that can be used for writing tests where real I/O isn't
/// desirable. You can manually pass the data that will be consumed and test specific behaviour
/// without external dependencies.
pub(crate) mod test_transport {
    use core::error;
    use core::fmt;
    use core::fmt::Display;
    use core::fmt::Formatter;
    use std::io;
    use std::io::ErrorKind;
    use std::pin::Pin;
    use std::task::Context;
    use std::task::Poll;

    use bitcoin::Network;
    use tokio::io::AsyncRead;
    use tokio::io::AsyncWrite;
    use tokio::io::ReadBuf;

    use super::ReadTransport;

    #[derive(Debug, Default, Clone, Copy)]
    pub struct UnexpectedEofError;

    impl Display for UnexpectedEofError {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "unexpected eof")
        }
    }

    impl error::Error for UnexpectedEofError {}

    #[derive(Debug, Default)]
    pub struct Reader {
        data: Vec<u8>,
    }

    impl AsyncRead for Reader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            let size = buf.capacity();
            if size > self.data.len() {
                return Poll::Ready(Err(io::Error::new(
                    ErrorKind::UnexpectedEof,
                    UnexpectedEofError,
                )));
            }

            buf.put_slice(&self.data.drain(0..size).collect::<Vec<_>>());

            Poll::Ready(Ok(()))
        }
    }

    pub fn create_reader_v1(data: Vec<u8>) -> ReadTransport<Reader> {
        ReadTransport::V1(Reader { data }, Network::Regtest)
    }

    pub struct Writer;

    impl AsyncWrite for Writer {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            // No-op writer
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn is_write_vectored(&self) -> bool {
            true
        }

        fn poll_write_vectored(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            bufs: &[io::IoSlice<'_>],
        ) -> Poll<io::Result<usize>> {
            let len = bufs.iter().map(|buf| buf.len()).sum();
            // No-op writer
            Poll::Ready(Ok(len))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use bitcoin::Network;
    use bitcoin::consensus::serialize;
    use bitcoin::p2p::message::NetworkMessage;
    use bitcoin::p2p::message::RawNetworkMessage;

    use super::test_transport::*;
    use crate::p2p_wire::transport::P2PV1MessageChecksum;
    use crate::p2p_wire::transport::TransportError;
    use crate::p2p_wire::transport::V1MessageHeader;

    #[tokio::test]
    async fn test_oversized_message() {
        let oversized_message_header = V1MessageHeader {
            magic: Network::Regtest.magic(),
            checksum: P2PV1MessageChecksum([0; 4]),
            command: [0; 12],
            length: u32::MAX,
        };

        let data = serialize(&oversized_message_header);
        let mut transport_reader = create_reader_v1(data);

        let error = transport_reader.read_message().await.unwrap_err();

        assert!(matches!(error, TransportError::OversizedMessage { .. }));
    }

    #[tokio::test]
    async fn test_bad_magic() {
        let bad_magic_msg_header = V1MessageHeader {
            magic: Network::Signet.magic(),
            checksum: P2PV1MessageChecksum([0; 4]),
            command: [0; 12],
            length: 0,
        };

        let data = serialize(&bad_magic_msg_header);
        let mut transport_reader = create_reader_v1(data);

        let error = transport_reader.read_message().await.unwrap_err();

        assert!(matches!(error, TransportError::BadMagicBits { .. }));
    }

    #[tokio::test]
    async fn test_bad_checksum() {
        let payload = NetworkMessage::Ping(0);
        let message = RawNetworkMessage::new(Network::Regtest.magic(), payload);
        let mut data = serialize(&message);
        // mess with the checksum
        data[23] ^= 1;

        let mut transport_reader = create_reader_v1(data);

        let error = transport_reader.read_message().await.unwrap_err();

        assert!(matches!(error, TransportError::BadChecksum { .. }));
    }

    #[tokio::test]
    async fn test_wrong_length() {
        let payload = NetworkMessage::Ping(0);
        let message = RawNetworkMessage::new(Network::Regtest.magic(), payload);
        let mut data = serialize(&message);
        // make the size look one byte bigger than the actual message is, this will cause an EOF
        data[16] = 9;
        let mut transport_reader = create_reader_v1(data);

        let error = transport_reader.read_message().await.unwrap_err();

        match error {
            TransportError::Io(e) => assert_eq!(e.kind(), ErrorKind::UnexpectedEof),
            _ => panic!("Expected an IO error"),
        }
    }

    #[tokio::test]
    async fn test_valid_message() {
        let payload = NetworkMessage::Ping(0);
        let message = RawNetworkMessage::new(Network::Regtest.magic(), payload);
        let data = serialize(&message);
        // make the size look one byte bigger than the actual message is, this will cause an EOF
        let mut transport_reader = create_reader_v1(data);

        let res = transport_reader
            .read_message()
            .await
            .expect("Message should be a valid ping");

        assert_eq!(res, NetworkMessage::Ping(0));
    }
}

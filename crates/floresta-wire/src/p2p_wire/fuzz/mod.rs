// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzzing harnesses for the P2P transports.

use core::net::Ipv4Addr;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use bip324::Role;
use bip324::futures::Protocol;
use bip324::io::Payload;
use bitcoin::Network;
use bitcoin::consensus::serialize;
use bitcoin::p2p::ServiceFlags;
use bitcoin::p2p::address::AddrV2;
use bitcoin::p2p::message::NetworkMessage;
use bitcoin::p2p::message::RawNetworkMessage;
use floresta_domain::mempool::MempoolBase;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::io::BufReader;
use tokio::io::duplex;
use tokio::io::sink;
use tokio::sync::Mutex;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot;
use tokio::time::timeout;

use super::address_man::AddressState;
use super::address_man::LocalAddress;
use super::bitcoin_socket_addr::BitcoinSocketAddr;
use super::network_message_ext::NetworkMessageExt;
use super::node::ConnectionKind;
use super::peer::Peer;
use super::peer::create_actors;
use super::peer::peer_utils;
use super::transport::ReadTransport;
use super::transport::TransportProtocol;
use super::transport::WriteTransport;

/// The user agent used by the mocked peers.
const FUZZ_USER_AGENT: &str = "/Floresta-fuzz:0.0.0/";

/// A timeout for V2 sessions, because the `bip324` crate sometimes won't
/// return after an EOF, making the fuzz hung up forever.
const V2_TIMEOUT: Duration = Duration::from_millis(10);

/// Feeds arbitrary bytes to a P2P V1 peer through an in-memory reader.
pub async fn v1_peer<M>(data: Vec<u8>, mempool: M)
where
    M: MempoolBase + 'static,
{
    let data = add_v1_handshake(data);
    let reader = BufReader::new(Cursor::new(data));
    let reader = ReadTransport::V1(reader, Network::Regtest);
    let writer = WriteTransport::V1(sink(), Network::Regtest);

    run_peer(reader, writer, TransportProtocol::V1, mempool).await;
}

/// Feeds arbitrary bytes to a P2P V2 peer over an in-memory BIP324 connection.
pub async fn v2_peer<M>(data: Vec<u8>, mempool: M)
where
    M: MempoolBase + 'static,
{
    let (local_stream, remote_stream) = duplex(8 * 1024);
    let (local_reader, local_writer) = tokio::io::split(local_stream);
    let (remote_reader, remote_writer) = tokio::io::split(remote_stream);

    let local_protocol = Protocol::new(
        Network::Regtest.magic(),
        Role::Initiator,
        None,
        None,
        BufReader::new(local_reader),
        local_writer,
    );
    let remote_protocol = Protocol::new(
        Network::Regtest.magic(),
        Role::Responder,
        None,
        None,
        BufReader::new(remote_reader),
        remote_writer,
    );
    let (local_protocol, remote_protocol) = tokio::join!(local_protocol, remote_protocol);
    let local_protocol = local_protocol.expect("in-memory initiator handshake should succeed");
    let remote_protocol = remote_protocol.expect("in-memory responder handshake should succeed");
    let (local_reader, local_writer) = local_protocol.into_split();
    let (mut remote_reader, mut remote_writer) = remote_protocol.into_split();

    // The first byte selects a Bitcoin handshake prelude and the BIP324 packet type.
    let control = data.first().copied().unwrap_or_default();
    let payload = data.get(1..).unwrap_or_default().to_vec();
    let address = peer_address();
    let remote_sender = tokio::spawn(async move {
        if control & 1 != 0 {
            let messages = [
                peer_utils::build_version_message(FUZZ_USER_AGENT.into(), 0, &address),
                NetworkMessage::Verack,
            ];
            for message in messages {
                remote_writer
                    .write(&Payload::genuine(message.serialize_v2()))
                    .await?;
            }
        }

        let payload = match control & 2 {
            0 => Payload::genuine(payload),
            _ => Payload::decoy(payload),
        };
        remote_writer.write(&payload).await
    });
    let remote_receiver = tokio::spawn(async move { while remote_reader.read().await.is_ok() {} });

    let _ = timeout(
        V2_TIMEOUT,
        run_peer(
            ReadTransport::V2(local_reader),
            WriteTransport::V2(local_writer),
            TransportProtocol::V2,
            mempool,
        ),
    )
    .await;

    let _ = remote_sender
        .await
        .expect("mock V2 sender should not panic");
    remote_receiver
        .await
        .expect("mock V2 receiver should not panic");
}

async fn run_peer<R, W, M>(
    reader: ReadTransport<R>,
    writer: WriteTransport<W>,
    transport_protocol: TransportProtocol,
    mempool: M,
) where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + Sync + 'static,
    M: MempoolBase + 'static,
{
    let (node_tx, _node_rx) = unbounded_channel();
    let (_request_tx, request_rx) = unbounded_channel();
    let (cancellation_tx, cancellation_rx) = oneshot::channel();
    let (actor_rx, actor) = create_actors(reader);
    let actor_task = tokio::spawn(async move {
        tokio::select! {
            result = actor.run() => result,
            _ = cancellation_rx => Ok(()),
        }
    });
    let mempool: Arc<Mutex<dyn MempoolBase>> = Arc::new(Mutex::new(mempool));
    let peer_task = Peer::<W>::create_peer(
        0,
        peer_address(),
        mempool,
        node_tx,
        request_rx,
        ConnectionKind::Manual,
        actor_rx,
        writer,
        FUZZ_USER_AGENT.into(),
        0,
        cancellation_tx,
        transport_protocol,
    );

    let _ = peer_task.await.expect("peer task should not panic");
    let _ = actor_task.await.expect("message actor should not panic");
}

fn add_v1_handshake(data: Vec<u8>) -> Vec<u8> {
    let Some((&control, payload)) = data.split_first() else {
        return data;
    };

    // Keep one mode fully raw and one mode past the Bitcoin handshake state machine.
    if control & 1 == 0 {
        return payload.to_vec();
    }

    let address = peer_address();
    let version = peer_utils::build_version_message(FUZZ_USER_AGENT.into(), 0, &address);
    let mut socket_data = serialize(&RawNetworkMessage::new(Network::Regtest.magic(), version));
    socket_data.extend(serialize(&RawNetworkMessage::new(
        Network::Regtest.magic(),
        NetworkMessage::Verack,
    )));
    socket_data.extend(payload);
    socket_data
}

fn peer_address() -> LocalAddress {
    LocalAddress::new(
        BitcoinSocketAddr::new(AddrV2::Ipv4(Ipv4Addr::LOCALHOST), 18_444),
        0,
        AddressState::NeverTried,
        ServiceFlags::NONE,
        0,
    )
}

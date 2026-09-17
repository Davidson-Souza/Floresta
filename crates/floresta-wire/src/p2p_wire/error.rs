// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;
use std::io;

use floresta_chain::BlockchainError;
use floresta_common::impl_error_from;
use tokio::sync::mpsc::error::SendError;

use super::peer::PeerError;
use super::transport::TransportError;
use crate::address_man::LocalAddress;
use crate::bitcoin_socket_addr::BitcoinSocketAddr;
use crate::bitcoin_socket_addr::InvalidAddressError;
use crate::node::NodeRequest;

#[derive(Debug)]
pub enum WireError {
    /// Blockchain-related error.
    ///
    /// This error kind is returned by our `ChainState`.
    Blockchain(BlockchainError),

    /// Error while writing into a channel
    ChannelSend(SendError<NodeRequest>),

    /// Attempted to connect with a network we can' reach
    UnreachableNetwork,

    /// Peer error
    PeerError(PeerError),

    /// Coinbase isn't mature
    CoinbaseNotMatured,

    /// Peer not found in our current connections
    PeerNotFound,

    /// We don't have any peers that could fulfill such request.
    NoPeersAvailable,

    /// Our peer is misbehaving
    PeerMisbehaving,

    /// Failed to init Utreexo peers: anchors.json does not exist yet
    AnchorFileNotFound,

    /// Peer already exists in our peers list
    PeerAlreadyExists(LocalAddress),

    /// Peer not found with this given address and port, in our peer list
    PeerNotFoundAtAddress(BitcoinSocketAddr),

    /// Generic io error
    Io(std::io::Error),

    /// JSON (de)serialization error
    Serde(serde_json::Error),

    /// Failed to save Utreexo peers: no peers to save to anchors.json
    NoUtreexoPeersAvailable,

    /// We couldn't find a peer to send a request
    NoPeerToSendRequest,

    /// Peer timed out some request
    PeerTimeout,

    /// Poisoned lock
    PoisonedLock,

    /// We couldn't parse the provided address
    InvalidAddress(InvalidAddressError),

    /// Transport error
    Transport(TransportError),

    /// Can't send back response for user request
    ResponseSendError,

    /// No addresses available to connect to
    NoAddressesAvailable,

    /// We tried to work on a block we don't have. This is a bug!
    BlockNotFound,

    /// We tried to work on a block that we don't have a proof for yet. This is a bug!
    BlockProofNotFound,

    /// Couldn't find the leaf data for a block
    LeafDataNotFound,
}

impl Display for WireError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnreachableNetwork => {
                write!(f, "The provided network is invalid or unreachable")
            }
            Self::Blockchain(err) => write!(f, "Blockchain error: {err:?}"),
            Self::ChannelSend(err) => write!(f, "Error while writing into channel: {err:?}"),
            Self::PeerError(err) => write!(f, "Peer error: {err:?}"),
            Self::CoinbaseNotMatured => write!(f, "Coinbase isn't mature yet"),
            Self::PeerNotFound => write!(f, "Peer not found in our current connections list"),
            Self::NoPeersAvailable => write!(f, "We don't have peers to send a given request"),
            Self::PeerMisbehaving => write!(f, "Our peer is misbehaving"),
            Self::AnchorFileNotFound => write!(
                f,
                "Failed to init Utreexo peers: anchors.json does not exist yet"
            ),
            Self::PeerAlreadyExists(address) => write!(f, "Peer {address} already exists"),
            Self::PeerNotFoundAtAddress(address) => write!(f, "Peer {address} not found"),
            Self::Io(err) => write!(f, "Generic IO error: {err:?}"),
            Self::Serde(err) => write!(f, "Serde error: {err:?}"),
            Self::NoUtreexoPeersAvailable => write!(
                f,
                "Failed to save Utreexo peers: no peers to save to anchors.json"
            ),
            Self::NoPeerToSendRequest => {
                write!(f, "We couldn't find a peer to send the request")
            }
            Self::PeerTimeout => write!(f, "Peer timed out"),
            Self::PoisonedLock => write!(f, "Poisoned lock"),
            Self::InvalidAddress(err) => {
                write!(f, "We couldn't parse the provided address due to: {err:?}")
            }
            Self::Transport(err) => write!(f, "Transport error: {err:?}"),
            Self::ResponseSendError => write!(f, "Can't send back response for user request"),
            Self::NoAddressesAvailable => write!(f, "No addresses available to connect to"),
            Self::BlockNotFound => write!(f, "We tried to work on a block we don't have"),
            Self::BlockProofNotFound => write!(
                f,
                "We tried to work on a block that we don't have a proof for yet"
            ),
            Self::LeafDataNotFound => write!(f, "Couldn't find the leaf data for a block"),
        }
    }
}

impl_error_from!(WireError, PeerError, PeerError);
impl_error_from!(WireError, BlockchainError, Blockchain);
impl_error_from!(WireError, InvalidAddressError, InvalidAddress);
impl_error_from!(WireError, SendError<NodeRequest>, ChannelSend);
impl_error_from!(WireError, serde_json::Error, Serde);
impl_error_from!(WireError, io::Error, Io);

impl From<tokio::sync::oneshot::error::RecvError> for WireError {
    fn from(_: tokio::sync::oneshot::error::RecvError) -> Self {
        Self::ResponseSendError
    }
}

impl From<TransportError> for WireError {
    fn from(e: TransportError) -> Self {
        match e {
            TransportError::Io(io) => Self::Io(io),
            other => Self::Transport(other),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AddrParseError {
    InvalidIpv6,
    InvalidIpv4,
    InvalidHostname,
    InvalidPort,
    Inconclusive,
}

impl Display for AddrParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::InvalidIpv6 => write!(f, "Invalid ipv6"),
            Self::InvalidIpv4 => write!(f, "Invalid ipv4"),
            Self::InvalidHostname => write!(f, "Invalid hostname"),
            Self::InvalidPort => write!(f, "Invalid port"),
            Self::Inconclusive => write!(f, "Inconclusive"),
        }
    }
}

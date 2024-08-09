use floresta_chain::BlockchainError;
use floresta_common::impl_error_from;
use floresta_compact_filters::IteratableFilterStoreError;
use thiserror::Error;

use super::peer::PeerError;
use crate::node::NodeRequest;

#[derive(Error, Debug)]
pub enum WireError {
    #[error("Blockchain error")]
    Blockchain(BlockchainError),
    #[error("Error while writing into a channel")]
    ChannelSend(tokio::sync::mpsc::error::SendError<NodeRequest>),
    #[error("Peer error")]
    PeerError(PeerError),
    #[error("Coinbase didn't mature")]
    CoinbaseNotMatured,
    #[error("Peer not found in our current connections")]
    PeerNotFound,
    #[error("We don't have any peers")]
    NoPeersAvailable,
    #[error("Our peer is misbehaving")]
    PeerMisbehaving,
    #[error("Error while reading from a channel")]
    ChannelRecv(#[from] tokio::sync::mpsc::error::RecvError),
    #[error("Generic io error")]
    Io(std::io::Error),
    #[error("We don't have any utreexo peers")]
    NoUtreexoPeersAvailable,
    #[error("We couldn't find a peer to send the request")]
    NoPeerToSendRequest,
    #[error("Peer timed out")]
    PeerTimeout,
    #[error("Compact block filters error")]
    CompactBlockFiltersError(IteratableFilterStoreError),
}

impl_error_from!(WireError, PeerError, PeerError);
impl_error_from!(WireError, BlockchainError, Blockchain);
impl_error_from!(
    WireError,
    IteratableFilterStoreError,
    CompactBlockFiltersError
);

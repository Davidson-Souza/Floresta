//! A simple async reader that reads data from a [Source] and builds a [Item] from it, assuming
//! item is [Decodable]. The main intent of this module is to read [RawNetworkMessages] from [TcpStream], because
//! we don't know how many bytes to read upfront, so we might read an incomplete message and try
//! to deserialize it, causing unrelated error. This module first reads the message reader
//! that has constant size (for RawNetworkMessage is 24). Then we look for payload size inside
//! this header. With payload size we can finally read the entire message and return a parsed
//! structure.

use std::marker::PhantomData;

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::mpsc::UnboundedSender;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::deserialize_partial;
use bitcoin::consensus::Decodable;
use bitcoin::p2p::Magic;


use super::peer::PeerError;

/// A simple type that wraps a stream and returns T, if T is [Decodable].
pub struct StreamReader<Source: Sync + Send + AsyncReadExt + Unpin + AsyncRead, Item: Decodable + Send> {
    /// Were we read bytes from, usually a TcpStream
    source: Source,
    /// Item is what we return, since we don't actually hold any concrete type, just use a
    /// phantom data to bind a type.
    phantom: PhantomData<Item>,
    /// Magic bits, we expect this at the beginning of all messages
    magic: Magic,
    /// Where should we send data
    sender: UnboundedSender<Result<Item, PeerError>>,
}
impl<Source, Item> StreamReader<Source, Item>
where
    Item: Decodable + Unpin + Send + 'static,
    Source: Sync + Send + AsyncReadExt + Unpin + AsyncRead,
{
    /// Creates a new reader from a given stream
    pub fn new(stream: Source, magic: Magic, sender: UnboundedSender<Result<Item, PeerError>>) -> Self {
        StreamReader {
            source: stream,
            phantom: PhantomData,
            magic,
            sender,
        }
    }
    async fn read_loop_inner(&mut self) -> Result<(), PeerError> {
        loop {
            let mut data: Vec<u8> = vec![0; 24];

            // Read the reader first, so learn the payload size
            self.source.read_exact(&mut data).await?;
            let header: P2PMessageHeader = deserialize_partial(&data)?.0;
            if header.magic != self.magic {
                return Err(PeerError::MagicBitsMismatch);
            }
            // Network Message too big
            if header.length > (1024 * 1024 * 32) as u32 {
                return Err(PeerError::MessageTooBig);
            }

            data.resize(24 + header.length as usize, 0);
            // Read everything else
            self.source.read_exact(&mut data[24..]).await?;
            let message = deserialize(&data)?;
            let _ = self.sender.send(Ok(message));
        }
    }
    /// Tries to read from a parsed [Item] from [Source]. Only returns on error or if we have
    /// a valid Item to return
    pub async fn read_loop(mut self) {
        let value = self.read_loop_inner().await;
        if let Err(e) = value {
            let _ = self.sender.send(Err(e));
        }
    }
}

#[derive(Debug)]
pub struct P2PMessageHeader {
    magic: Magic,
    _command: [u8; 12],
    length: u32,
    _checksum: u32,
}
impl Decodable for P2PMessageHeader {
    fn consensus_decode<R: std::io::Read + ?Sized>(
        reader: &mut R,
    ) -> Result<Self, bitcoin::consensus::encode::Error> {
        let magic = Magic::consensus_decode(reader)?;
        let _command = <[u8; 12]>::consensus_decode(reader)?;
        let length = u32::consensus_decode(reader)?;
        let _checksum = u32::consensus_decode(reader)?;
        Ok(Self {
            _checksum,
            _command,
            length,
            magic,
        })
    }
}

//! A wrapper around a TCP stream that handles the SOCKS5 handshake. This module only
//! drives the handshake and returns the stream back to the caller. It does not
//! perform any I/O on the stream after the handshake is complete. The caller is
//! responsible for performing I/O on the stream.
//! This module is built on top of the `futures` crate instead of an specific
//! async runtime. This allows the caller to use this module with any async runtime
//! they want.

use std::net::Ipv4Addr;
use std::net::Ipv6Addr;
use std::net::SocketAddr;

use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;

#[derive(Clone, Debug)]
pub struct Socks5StreamBuilder {
    pub address: SocketAddr,
}
/// The version of the SOCKS protocol we support, only SOCKS5 is supported.
const SOCKS_VERSION: u8 = 5;
/// The SOCKS authentication method we support, only no authentication is supported.
const SOCKS_AUTH_METHOD_NONE: u8 = 0;
/// The cmd value for a SOCKS5 connect request.
const SOCKS_CMD_CONNECT: u8 = 1;
/// Magic value to indicate an IPv4 address.
const SOCKS_ADDR_TYPE_IPV4: u8 = 1;
/// Magic value to indicate a domain address.
const SOCKS_ADDR_TYPE_DOMAIN: u8 = 3;
/// Magic value to indicate an IPv6 address.
const SOCKS_ADDR_TYPE_IPV6: u8 = 4;

#[derive(Debug)]
#[allow(dead_code)]
pub enum Socks5Addr {
    Ipv4(Ipv4Addr),
    Ipv6(Ipv6Addr),
    Domain(Box<[u8]>),
}
impl From<Socks5Addr> for u8 {
    fn from(val: Socks5Addr) -> Self {
        match val {
            Socks5Addr::Ipv4(_) => SOCKS_ADDR_TYPE_IPV4,
            Socks5Addr::Ipv6(_) => SOCKS_ADDR_TYPE_IPV6,
            Socks5Addr::Domain(_) => SOCKS_ADDR_TYPE_DOMAIN,
        }
    }
}
impl Socks5StreamBuilder {
    pub fn new(address: SocketAddr) -> Self {
        Self { address }
    }
    pub async fn connect<Stream: AsyncRead + AsyncWrite + Unpin>(
        mut socket: Stream,
        address: Socks5Addr,
        port: u16,
    ) -> Result<Stream, Socks5Error> {
        socket
            .write_all(&[SOCKS_VERSION, 1, SOCKS_AUTH_METHOD_NONE])
            .await
            .unwrap();
        let address = match address {
            Socks5Addr::Ipv4(addr) => addr.octets().to_vec(),
            Socks5Addr::Ipv6(addr) => addr.octets().to_vec(),
            Socks5Addr::Domain(domain) => {
                let mut buf = vec![domain.len() as u8];
                buf.extend_from_slice(&domain);
                buf
            }
        };
        let mut buf = [0_u8; 2];
        socket.read_exact(&mut buf).await?;

        if buf[0] != SOCKS_VERSION {
            return Err(Socks5Error::InvalidVersion);
        }

        if buf[1] != SOCKS_AUTH_METHOD_NONE {
            return Err(Socks5Error::InvalidAuthMethod);
        }

        socket
            .write_all(&[SOCKS_VERSION, SOCKS_CMD_CONNECT, 0, SOCKS_ADDR_TYPE_IPV4])
            .await?;
        socket.write_all(&address).await?;
        socket.write_all(&port.to_be_bytes()).await?;

        let mut buf = [0_u8; 4];
        socket.read_exact(&mut buf).await?;

        if buf[0] != SOCKS_VERSION {
            return Err(Socks5Error::InvalidVersion);
        }
        if buf[1] != 0 {
            return Err(Socks5Error::ConnectionFailed);
        }

        match buf[3] {
            SOCKS_ADDR_TYPE_IPV4 => {
                let mut buf = [0_u8; 6];
                socket.read_exact(&mut buf).await?;
            }
            SOCKS_ADDR_TYPE_IPV6 => {
                let mut buf = [0_u8; 18];
                socket.read_exact(&mut buf).await?;
            }
            SOCKS_ADDR_TYPE_DOMAIN => {
                let mut buf = [0_u8; 1];
                socket.read_exact(&mut buf).await?;
                let mut buf = vec![0_u8; buf[0] as usize + 2];
                socket.read_exact(&mut buf).await?;
            }
            _ => return Err(Socks5Error::ConnectionFailed),
        }
        Ok(socket)
    }
}
#[derive(Debug)]
pub enum Socks5Error {
    InvalidVersion,
    InvalidAuthMethod,
    ConnectionFailed,
    InvalidAddress,
    ReadError,
}

impl From<futures::io::Error> for Socks5Error {
    fn from(_error: futures::io::Error) -> Self {
        Socks5Error::ReadError
    }
}

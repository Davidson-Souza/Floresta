// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::fmt::Formatter;

use bitcoin::VarInt;
use bitcoin::consensus::Decodable;
use bitcoin::consensus::Encodable;
use bitcoin::consensus::encode;
use bitcoin::p2p::message::CommandString;
use bitcoin::p2p::message::NetworkMessage;

#[derive(Debug)]
/// V2 message serialization error.
pub enum V2MessageError {
    /// Failed to deserialize a V2 message.
    Deserialize(encode::Error),

    /// Unknown V2 short message ID.
    UnknownShortID(u8),
}

impl fmt::Display for V2MessageError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deserialize(e) => write!(f, "unable to deserialize V2 message: {e}"),
            Self::UnknownShortID(b) => write!(f, "unrecognized short ID: {b}"),
        }
    }
}

impl core::error::Error for V2MessageError {}

impl From<encode::Error> for V2MessageError {
    fn from(e: encode::Error) -> Self {
        Self::Deserialize(e)
    }
}

/// Utilities to work with [`NetworkMessage`]s from/to p2p V2.
///
/// This code can be removed after we migrate to the enw `bitcoin-p2p` crate.
pub trait NetworkMessageExt: Sized {
    /// Serialize a [`NetworkMessage`] into a V2-encoded buffer.
    ///
    /// Uses BIP-324 short message IDs for known message types.
    fn serialize_v2(&self) -> Vec<u8>;

    /// Deserialize a [`NetworkMessage`] from a V2-encoded buffer.
    fn deserialize_v2(buffer: &[u8]) -> Result<Self, V2MessageError>;
}

impl NetworkMessageExt for NetworkMessage {
    /// Serialize a [`NetworkMessage`] into a V2-encoded buffer.
    ///
    /// Uses BIP-324 short message IDs for known message types.
    fn serialize_v2(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        // TODO: remove this once https://github.com/rust-bitcoin/rust-bitcoin/pull/5671 and
        // https://github.com/rust-bitcoin/rust-bitcoin/pull/5009 make it into a release
        if let Self::Unknown { command, payload } = self {
            /// P2PV2 BIP-0324 message type for `getuproof`.
            const P2PV2_GETUPROOF_MSG_TYPE: u8 = 30;
            // The legacy cmd string for get utreexo proof
            let get_utreexo_proof_cmd_string: CommandString =
                CommandString::try_from_static("getuproof")
                    .expect("`getuproof` is a valid command string");

            if *command == get_utreexo_proof_cmd_string {
                buffer.push(P2PV2_GETUPROOF_MSG_TYPE);
                buffer.extend(payload);
                return buffer;
            }

            // fall through, will handle this at the bottom of that match bellow.
        }

        match self {
            Self::Addr(_) => buffer.push(1u8),
            Self::Inv(_) => buffer.push(14u8),
            Self::GetData(_) => buffer.push(11u8),
            Self::NotFound(_) => buffer.push(17u8),
            Self::GetBlocks(_) => buffer.push(9u8),
            Self::GetHeaders(_) => buffer.push(12u8),
            Self::MemPool => buffer.push(15u8),
            Self::Tx(_) => buffer.push(21u8),
            Self::Block(_) => buffer.push(2u8),
            Self::Headers(_) => buffer.push(13u8),
            Self::Ping(_) => buffer.push(18u8),
            Self::Pong(_) => buffer.push(19u8),
            Self::MerkleBlock(_) => buffer.push(16u8),
            Self::FilterLoad(_) => buffer.push(8u8),
            Self::FilterAdd(_) => buffer.push(6u8),
            Self::FilterClear => buffer.push(7u8),
            Self::GetCFilters(_) => buffer.push(22u8),
            Self::CFilter(_) => buffer.push(23u8),
            Self::GetCFHeaders(_) => buffer.push(24u8),
            Self::CFHeaders(_) => buffer.push(25u8),
            Self::GetCFCheckpt(_) => buffer.push(26u8),
            Self::CFCheckpt(_) => buffer.push(27u8),
            Self::SendCmpct(_) => buffer.push(20u8),
            Self::CmpctBlock(_) => buffer.push(4u8),
            Self::GetBlockTxn(_) => buffer.push(10u8),
            Self::BlockTxn(_) => buffer.push(3u8),
            Self::FeeFilter(_) => buffer.push(5u8),
            Self::AddrV2(_) => buffer.push(28u8),
            Self::Version(_)
            | Self::Verack
            | Self::SendHeaders
            | Self::GetAddr
            | Self::WtxidRelay
            | Self::SendAddrV2
            | Self::Alert(_)
            | Self::Reject(_) => {
                buffer.push(0u8);
                self.command()
                    .consensus_encode(&mut buffer)
                    .expect("Encoding to Vec<u8> never fails");
            }
            Self::Unknown { command, payload } => {
                buffer.push(0u8);
                command
                    .consensus_encode(&mut buffer)
                    .expect("Encoding to Vec<u8> never fails");
                buffer.extend(payload);
                return buffer;
            }
        }

        self.consensus_encode(&mut buffer)
            .expect("Encoding to Vec<u8> never fails");

        buffer
    }

    /// Deserialize a [`NetworkMessage`] from a V2-encoded buffer.
    fn deserialize_v2(buffer: &[u8]) -> Result<Self, V2MessageError> {
        if buffer.is_empty() {
            return Err(V2MessageError::Deserialize(encode::Error::Io(
                bitcoin::io::Error::new(
                    bitcoin::io::ErrorKind::UnexpectedEof,
                    "Missing short_id for message",
                ),
            )));
        }

        let short_id = buffer[0];
        let mut payload_buffer = &buffer[1..];

        // TODO: remove this once rust-bitcoin supports these BIP-0324
        // Utreexo message types.
        const P2PV2_UPROOF_MSG_TYPE: u8 = 29;
        const P2PV2_UTREEXO_TX_MSG_TYPE: u8 = 34;
        let utreexo_command = match short_id {
            P2PV2_UPROOF_MSG_TYPE => Some("uproof"),
            P2PV2_UTREEXO_TX_MSG_TYPE => Some("utreexotx"),
            _ => None,
        };
        if let Some(command) = utreexo_command {
            return Ok(Self::Unknown {
                command: CommandString::try_from_static(command).expect("Utreexo command is valid"),
                payload: payload_buffer.to_vec(),
            });
        }

        match short_id {
            0u8 => {
                let Some(mut command_buffer) = buffer.get(1..13) else {
                    return Err(V2MessageError::Deserialize(encode::Error::Io(
                        bitcoin::io::Error::new(
                            bitcoin::io::ErrorKind::UnexpectedEof,
                            "Missing command for zero short_id message",
                        ),
                    )));
                };
                let command_end = command_buffer
                    .iter()
                    .position(|byte| *byte == 0)
                    .unwrap_or(command_buffer.len());
                let command_is_valid = command_buffer[..command_end]
                    .iter()
                    .all(|byte| (0x20..=0x7e).contains(byte))
                    && command_buffer[command_end..].iter().all(|byte| *byte == 0);
                if !command_is_valid {
                    return Err(V2MessageError::Deserialize(encode::Error::ParseFailed(
                        "Invalid command string",
                    )));
                }
                let command = CommandString::consensus_decode(&mut command_buffer)
                    .map_err(V2MessageError::Deserialize)?;
                payload_buffer = &buffer[13..];
                match command.as_ref() {
                    "version" => Ok(Self::Version(Decodable::consensus_decode(
                        &mut payload_buffer,
                    )?)),
                    "verack" => Ok(Self::Verack),
                    "sendheaders" => Ok(Self::SendHeaders),
                    "getaddr" => Ok(Self::GetAddr),
                    "wtxidrelay" => Ok(Self::WtxidRelay),
                    "sendaddrv2" => Ok(Self::SendAddrV2),
                    "alert" => Ok(Self::Alert(Decodable::consensus_decode(
                        &mut payload_buffer,
                    )?)),
                    "reject" => Ok(Self::Reject(Decodable::consensus_decode(
                        &mut payload_buffer,
                    )?)),
                    _ => Ok(Self::Unknown {
                        command,
                        payload: payload_buffer.to_vec(),
                    }),
                }
            }
            1u8 => Ok(Self::Addr(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            2u8 => Ok(Self::Block(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            3u8 => Ok(Self::BlockTxn(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            4u8 => Ok(Self::CmpctBlock(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            5u8 => Ok(Self::FeeFilter(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            6u8 => Ok(Self::FilterAdd(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            7u8 => Ok(Self::FilterClear),
            8u8 => Ok(Self::FilterLoad(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            9u8 => Ok(Self::GetBlocks(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            10u8 => Ok(Self::GetBlockTxn(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            11u8 => Ok(Self::GetData(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            12u8 => Ok(Self::GetHeaders(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            13u8 => {
                let len = VarInt::consensus_decode(&mut payload_buffer)?.0;
                let mut headers = Vec::with_capacity(core::cmp::min(1024 * 16, len as usize));
                for _ in 0..len {
                    headers.push(Decodable::consensus_decode(&mut payload_buffer)?);
                    if u8::consensus_decode(&mut payload_buffer)? != 0u8 {
                        return Err(V2MessageError::Deserialize(encode::Error::ParseFailed(
                            "Headers message should not contain transactions",
                        )));
                    }
                }
                Ok(Self::Headers(headers))
            }
            14u8 => Ok(Self::Inv(Decodable::consensus_decode(&mut payload_buffer)?)),
            15u8 => Ok(Self::MemPool),
            16u8 => Ok(Self::MerkleBlock(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            17u8 => Ok(Self::NotFound(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            18u8 => Ok(Self::Ping(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            19u8 => Ok(Self::Pong(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            20u8 => Ok(Self::SendCmpct(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            21u8 => Ok(Self::Tx(Decodable::consensus_decode(&mut payload_buffer)?)),
            22u8 => Ok(Self::GetCFilters(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            23u8 => Ok(Self::CFilter(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            24u8 => Ok(Self::GetCFHeaders(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            25u8 => Ok(Self::CFHeaders(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            26u8 => Ok(Self::GetCFCheckpt(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            27u8 => Ok(Self::CFCheckpt(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            28u8 => Ok(Self::AddrV2(Decodable::consensus_decode(
                &mut payload_buffer,
            )?)),
            unknown => Err(V2MessageError::UnknownShortID(unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use bitcoin::p2p::message::CommandString;
    use bitcoin::p2p::message::NetworkMessage;

    use super::NetworkMessageExt;
    use super::V2MessageError;

    #[test]
    fn deserialize_v2_zero_short_id() {
        assert!(matches!(
            NetworkMessage::deserialize_v2(&[0u8]),
            Err(V2MessageError::Deserialize(_))
        ));

        assert!(matches!(
            NetworkMessage::deserialize_v2(&[0u8; 12]),
            Err(V2MessageError::Deserialize(_))
        ));

        let mut verack = vec![0u8];
        verack.extend(b"verack\0\0\0\0\0\0");
        assert_eq!(
            NetworkMessage::deserialize_v2(&verack).expect("valid verack frame"),
            NetworkMessage::Verack
        );
    }

    #[test]
    fn unknown_v2_message_roundtrip() {
        let message = NetworkMessage::Unknown {
            command: CommandString::try_from_static("uproof").expect("valid command"),
            payload: vec![0xf8, 0x08],
        };

        let encoded = message.serialize_v2();
        let decoded =
            NetworkMessage::deserialize_v2(&encoded).expect("serialized message must decode");

        assert_eq!(decoded, message);
    }

    #[test]
    fn deserialize_v2_utreexo_transaction() {
        let message = NetworkMessage::deserialize_v2(&[34, 0xf8, 0x08])
            .expect("Utreexo transaction frame must decode");

        assert_eq!(
            message,
            NetworkMessage::Unknown {
                command: CommandString::try_from_static("utreexotx").expect("valid command"),
                payload: vec![0xf8, 0x08],
            }
        );
    }

    #[test]
    fn deserialize_v2_rejects_invalid_command() {
        let invalid_command = [
            0, 0x7e, 0xa9, 0xa9, 0xa9, 0xa9, 0x1c, 0xa9, 0xa9, 0xa9, 0xa9, 0xa9, 0,
        ];

        assert!(NetworkMessage::deserialize_v2(&invalid_command).is_err());
    }
}

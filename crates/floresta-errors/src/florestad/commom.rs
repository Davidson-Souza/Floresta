use thiserror::Error;
use bitcoin::consensus::encode;
#[cfg(feature = "cli-blockchain")]
use btcd_rpc::error::UtreexodError;

use super::slip123::Slip32Errors;



#[derive(Error, Debug)]
pub enum FlorestadError {
    #[cfg(feature = "cli-blockchain")]
    #[error(transparent)]
    UtreexodError(#[from] UtreexodError),
    #[error(transparent)]
    Encode(#[from] encode::Error),
    #[error(transparent)]
    Db(#[from] kv::Error),
    #[error(transparent)]
    ParseNum(#[from] std::num::ParseIntError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    ScriptValidation(#[from] bitcoin::blockdata::script::Error),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
    #[error(transparent)]
    TomlParsing(#[from] toml::de::Error),
    #[error(transparent)]
    AddressParsing(#[from] bitcoin::address::ParseError),
    #[error(transparent)]
    Address(#[from] bitcoin::address::Error),
    #[error(transparent)]
    Miniscript(#[from] miniscript::Error),
    #[error(transparent)]
    Slip132(#[from] Slip32Errors),
}

use bitcoin::base58;
use thiserror::Error;

#[derive(Error, Clone, PartialEq, Eq, Debug)]
pub enum Slip32Errors {
    /// error in BASE58 key encoding. Details: {0}
    #[error(transparent)]
    Base58(#[from] base58::Error),

    /// error in hex key encoding. Details: {0}
    #[error(transparent)]
    Hex(#[from] bitcoin::hashes::hex::HexToArrayError),

    /// pk->pk derivation was attempted on a hardened key.
    #[error("cannot derive a hardened key from a hardened key")]
    CannotDeriveFromHardenedKey,

    /// child number {0} is out of range.
    #[error("child number {0} is out of range")]
    InvalidChildNumber(u32),

    /// invalid child number format.
    #[error("invalid child number format")]
    InvalidChildNumberFormat,

    /// invalid derivation path format.
    #[error("invalid derivation path format")]
    InvalidDerivationPathFormat,

    /// unknown version magic bytes {0:#06X?}
    #[error("unknown version magic bytes {0:#06X?}")]
    UnknownVersion([u8; 4]),

    /// encoded extended key data has wrong length {0}
    #[error("encoded extended key data has wrong length {0}")]
    WrongExtendedKeyLength(usize),

    /// unrecognized or unsupported extended key prefix (please check SLIP 32
    /// for possible values)
    #[error("unrecognized or unsupported extended key prefix")]
    UnknownSlip32Prefix,
    /// Unknown string representation of KeyApplication enum
    #[error("invalid key data {0}")]
    UnknownKeyApplicationError(String),
    // failure in rust bitcoin library [docstring]
    // Todo: remake this error... 
    // a fatal error should not carry any information about the internal state of the library
    //#[error()]
    //InternalFailure,
    #[error(transparent)]
    Bip32(#[from] bitcoin::bip32::Error),
    //#[error(transparent)]
    //WOError(#[from]  WatchOnlyError<D::Error>),
}
// SPDX-License-Identifier: MIT OR Apache-2.0

use core::error;
use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;
use core::net::AddrParseError;
use std::path::PathBuf;

use bitcoin::Network;
use bitcoin::consensus::encode;
use floresta_chain::BlockValidationErrors;
use floresta_chain::BlockchainError;
#[cfg(feature = "compact-filters")]
use floresta_compact_filters::IterableFilterStoreError;
use floresta_watch_only::WatchOnlyError;
use floresta_watch_only::descriptor::DescriptorError;
use floresta_watch_only::kv_database::KvDatabaseError;
use tokio_rustls::rustls::pki_types;

#[derive(Debug)]
pub enum FlorestadError {
    /// Encoding/decoding error.
    Encode(encode::Error),

    /// Integer parsing error.
    ParseNum(core::num::ParseIntError),

    /// Proof validation failure.
    Rustreexo(String),

    /// Generic IO operation error.
    Io(std::io::Error),

    // Block validation error, such as a missing transaction or an invalid proof.
    BlockValidation(BlockValidationErrors),

    /// Script validation error, such as an invalid script or a failed evaluation.
    ScriptValidation(bitcoin::blockdata::script::Error),

    /// Blockchain backend error, such as a missing block.
    Blockchain(BlockchainError),

    /// Deserializing JSON error.
    SerdeJson(serde_json::Error),

    /// TOML parsing error.
    TomlParsing(toml::de::Error),

    /// A signet challenge was configured for another network.
    SignetChallengeOnNonSignet(Network),

    /// Parsing registered HD version bytes from slip132.
    WalletInput(DescriptorError),

    /// Parsing a bitcoin address.
    AddressParsing(bitcoin::address::ParseError),

    /// Parsing miniscript error.
    Miniscript(miniscript::Error),

    /// Parsing a private key in PEM format.
    InvalidPrivKey(pki_types::pem::Error),

    /// Parsing a certificate from PEM format.
    InvalidCert(pki_types::pem::Error),

    /// Configuring TLS settings.
    CouldNotConfigureTLS(tokio_rustls::rustls::Error),

    /// Generating a PKCS#8 keypair.
    CouldNotGenerateKeypair(rcgen::Error),

    /// Generating a certificate parameter.
    CouldNotGenerateCertParam(rcgen::Error),

    /// Generating a self-signed certificate.
    CouldNotGenerateSelfSignedCert(rcgen::Error),

    /// Writing a file to the filesystem.
    CouldNotWriteFile(PathBuf, std::io::Error),

    /// Data directory doesn't exist or is not writable.
    InvalidDataDir(PathBuf),

    /// Obtaining a lock on the data directory.
    CouldNotOpenKvDatabase(KvDatabaseError),

    /// Initializing the watch-only wallet.
    CouldNotInitializeWallet(WatchOnlyError<KvDatabaseError>),

    /// Setting up the watch-only wallet.
    CouldNotSetupWallet(String),

    #[cfg(feature = "compact-filters")]
    /// Loading the compact filters store.
    CouldNotLoadCompactFiltersStore(IterableFilterStoreError),

    /// Failed to create a chain provider.
    CouldNotCreateChainProvider(String),

    /// Failed to create an Electrum server.
    CouldNotCreateElectrumServer(Box<dyn error::Error>),

    /// Failed to bind the Electrum server to a socket.
    FailedToBindElectrumServer(std::io::Error),

    /// Failed to create the TLS data directory.
    CouldNotCreateTLSDataDir(PathBuf, std::io::Error),

    /// Failed to obtain the wallet cache.
    CouldNotObtainWalletCache(WatchOnlyError<KvDatabaseError>),

    /// Failed to push a descriptor to the wallet.
    CouldNotPushDescriptor(String),

    /// Invalid Ip address error.
    InvalidIpAddress(AddrParseError),

    /// Ip address not found error.
    NoIPAddressesFound(String),

    /// Resolve a hostname error.
    CouldNotResolveHostname(std::io::Error),

    /// Load a flat chain store error.
    CouldNotLoadFlatChainStore(BlockchainError),
}

impl Display for FlorestadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "Encode error: {err}"),
            Self::ParseNum(err) => write!(f, "int parse error: {err}"),
            Self::Rustreexo(err) => write!(f, "Rustreexo error: {err}"),
            Self::Io(err) => write!(f, "Io error {err}"),
            Self::ScriptValidation(err) => {
                write!(f, "Error during script evaluation: {err}")
            }
            Self::Blockchain(err) => {
                write!(f, "Error with our blockchain backend: {err:?}")
            }
            Self::SerdeJson(err) => write!(f, "Error serializing object {err}"),
            Self::WalletInput(err) => write!(f, "Error while parsing user input {err:?}"),
            Self::TomlParsing(err) => write!(f, "Error deserializing toml file {err}"),
            Self::SignetChallengeOnNonSignet(network) => {
                write!(f, "A signet challenge cannot be used with {network}")
            }
            Self::AddressParsing(err) => write!(f, "Invalid address {err}"),
            Self::Miniscript(err) => write!(f, "Miniscript error: {err}"),
            Self::BlockValidation(err) => {
                write!(f, "Error while validating block: {err:?}")
            }
            Self::CouldNotConfigureTLS(err) => {
                write!(f, "Error while configuring TLS: {err:?}")
            }
            Self::InvalidPrivKey(err) => {
                write!(f, "Error while reading PKCS#8 private key {err:?}")
            }
            Self::InvalidCert(err) => {
                write!(f, "Error while reading PKCS#8 certificate {err:?}")
            }
            Self::CouldNotGenerateKeypair(err) => {
                write!(f, "Error while generating PKCS#8 keypair: {err}")
            }
            Self::CouldNotGenerateCertParam(err) => {
                write!(f, "Error while generating certificate param: {err}")
            }
            Self::CouldNotGenerateSelfSignedCert(err) => {
                write!(f, "Error while generating self-signed certificate: {err}")
            }
            Self::CouldNotWriteFile(path, err) => {
                write!(
                    f,
                    "Error while creating file at path={}: {err}",
                    path.display()
                )
            }
            Self::InvalidDataDir(path) => {
                write!(
                    f,
                    "Data directory at path={} doesn't exist or is not writable",
                    path.display()
                )
            }
            Self::CouldNotOpenKvDatabase(err) => {
                write!(f, "Cannot open a key-value database: {err}")
            }
            Self::CouldNotInitializeWallet(err) => {
                write!(f, "Could not initialize wallet: {err}")
            }
            Self::CouldNotSetupWallet(err) => {
                write!(f, "Could not setup wallet: {err}")
            }

            #[cfg(feature = "compact-filters")]
            Self::CouldNotLoadCompactFiltersStore(err) => {
                write!(f, "Could not load compact filters store: {err}")
            }

            Self::CouldNotCreateChainProvider(err) => {
                write!(f, "Could not create chain provider: {err}")
            }
            Self::CouldNotCreateElectrumServer(err) => {
                write!(f, "Could not create Electrum server: {err}")
            }
            Self::FailedToBindElectrumServer(err) => {
                write!(f, "Failed to bind Electrum server: {err}")
            }
            Self::CouldNotCreateTLSDataDir(path, err) => {
                write!(
                    f,
                    "Could not create TLS data directory at path={}: {err}",
                    path.display()
                )
            }
            Self::CouldNotObtainWalletCache(err) => {
                write!(f, "Could not obtain wallet cache: {err}")
            }
            Self::CouldNotPushDescriptor(err) => {
                write!(f, "Could not push descriptor to wallet: {err}")
            }
            Self::InvalidIpAddress(err) => {
                write!(f, "Invalid IP address: {err}")
            }
            Self::NoIPAddressesFound(hostname) => {
                write!(f, "No IP Addresses found for {hostname}")
            }
            Self::CouldNotResolveHostname(host) => {
                write!(f, "Could not resolve hostname: {host}")
            }
            Self::CouldNotLoadFlatChainStore(err) => {
                write!(f, "Failure while loading flat chainstore: {err:?}")
            }
        }
    }
}

/// Implements `From<T>` where `T` is a possible error outcome in this crate, this macro only
/// takes `T` and builds [`FlorestadError`] with the right variant.
macro_rules! impl_from_error {
    ($field:ident, $error:ty) => {
        impl From<$error> for FlorestadError {
            fn from(err: $error) -> Self {
                FlorestadError::$field(err)
            }
        }
    };
}

impl_from_error!(Encode, encode::Error);
impl_from_error!(ParseNum, core::num::ParseIntError);
impl_from_error!(Rustreexo, String);
impl_from_error!(Io, std::io::Error);
impl_from_error!(ScriptValidation, bitcoin::blockdata::script::Error);
impl_from_error!(Blockchain, BlockchainError);
impl_from_error!(SerdeJson, serde_json::Error);
impl_from_error!(WalletInput, DescriptorError);
impl_from_error!(TomlParsing, toml::de::Error);
impl_from_error!(BlockValidation, BlockValidationErrors);
impl_from_error!(AddressParsing, bitcoin::address::ParseError);
impl_from_error!(Miniscript, miniscript::Error);
impl_from_error!(CouldNotObtainWalletCache, WatchOnlyError<KvDatabaseError>);

impl error::Error for FlorestadError {}

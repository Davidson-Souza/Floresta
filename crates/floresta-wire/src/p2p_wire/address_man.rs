//! Address manager is a module that keeps track of known peer addresses and associated
//! metadata. This module is very important in keeping our node protected against targeted
//! attacks, like eclipse attacks.

use std::collections::HashMap;
use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::net::Ipv6Addr;
use std::str::FromStr;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::p2p::address::AddrV2;
use bitcoin::p2p::address::AddrV2Message;
use bitcoin::p2p::ServiceFlags;
use floresta_chain::Network;
use log::info;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

const RETRY_TIME: u64 = 60 * 60; // 1 hour
type AddressToSend = Vec<(AddrV2, u64, ServiceFlags, u16)>;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum AddressState {
    /// We never tried this peer before, so we don't know what to expect. This variant
    /// also applies to peers that we tried to connect, but failed or we didn't connect
    /// to for a long time.
    NeverTried,
    /// We tried this peer before, and had success at least once, so we know what to expect
    Tried(u64),
    /// This peer misbehaved and we banned them
    Banned(u64),
    /// We are connected to this peer right now
    Connected,
    /// We tried connecting, but failed
    Failed(u64),
}
/// How do we store peers locally
#[derive(Debug, Clone, PartialEq)]
pub struct LocalAddress {
    /// An actual address
    address: AddrV2,
    /// Last time we successfully connected to this peer, only relevant is state == State::Tried
    last_connected: u64,
    /// Our local state for this peer, as defined in AddressState
    state: AddressState,
    /// Network services announced by this peer
    services: ServiceFlags,
    /// Network port this peers listens to
    port: u16,
    /// Random id for this peer
    pub id: usize,
}
#[derive(Debug, Error)]
pub enum AddrManError {}
impl From<AddrV2> for LocalAddress {
    fn from(value: AddrV2) -> Self {
        LocalAddress {
            address: value,
            last_connected: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            state: AddressState::NeverTried,
            services: ServiceFlags::NONE,
            port: 8333,
            id: rand::random::<usize>(),
        }
    }
}
impl From<AddrV2Message> for LocalAddress {
    fn from(value: AddrV2Message) -> Self {
        LocalAddress {
            address: value.addr,
            last_connected: value.time.into(),
            state: AddressState::NeverTried,
            services: value.services,
            port: value.port,
            id: rand::random::<usize>(),
        }
    }
}
impl FromStr for LocalAddress {
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LocalAddress::try_from(s)
    }
    type Err = std::net::AddrParseError;
}
impl TryFrom<&str> for LocalAddress {
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let split = value.split(':').collect::<Vec<_>>();
        let address = split[0].parse::<Ipv4Addr>()?;
        let port = if let Some(port) = split.get(1) {
            port.parse().unwrap_or(8333)
        } else {
            8333
        };
        Ok(LocalAddress::new(
            AddrV2::Ipv4(address),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            super::address_man::AddressState::NeverTried,
            ServiceFlags::NONE,
            port,
            rand::random::<usize>(),
        ))
    }

    type Error = std::net::AddrParseError;
}

impl LocalAddress {
    pub fn new(
        address: AddrV2,
        last_connected: u64,
        state: AddressState,
        services: ServiceFlags,
        port: u16,
        id: usize,
    ) -> LocalAddress {
        LocalAddress {
            address,
            last_connected,
            state,
            services,
            port,
            id,
        }
    }
    /// Returns this address's port
    pub fn get_port(&self) -> u16 {
        self.port
    }
    /// Return an IP address associated with this peer address
    pub fn get_net_address(&self) -> IpAddr {
        match self.address {
            // IPV4
            AddrV2::Ipv4(ipv4) => IpAddr::V4(ipv4),
            // IPV6
            AddrV2::Ipv6(ipv6) => IpAddr::V6(ipv6),
            _ => IpAddr::V4(Ipv4Addr::LOCALHOST),
        }
    }
    /// Returns the actual address, as defined in AddrV2. This is useful
    /// if we are trying a peer that needs a proxy like Tor.
    pub fn get_address(&self) -> AddrV2 {
        self.address.clone()
    }
}
/// A module that keeps track of know addresses and serve them to our node to connect
#[derive(Default)]
pub struct AddressMan {
    addresses: HashMap<usize, LocalAddress>,
    good_addresses: Vec<usize>,
    utreexo_addresses: Vec<usize>,
}
impl AddressMan {
    /// Add a new address to our list of known address
    pub fn push_addresses(&mut self, addresses: &[LocalAddress]) {
        for address in addresses {
            let id = address.id;
            if let std::collections::hash_map::Entry::Vacant(e) = self.addresses.entry(id) {
                // For now we assume that all addresses are valid, until proven otherwise.
                self.good_addresses.push(id);
                if address.services.has(ServiceFlags::from(1 << 24)) {
                    self.utreexo_addresses.push(id);
                }
                e.insert(address.to_owned());
            }
        }
    }
    pub fn get_addresses_to_send(&self) -> AddressToSend {
        let addresses = self
            .addresses
            .iter()
            .flat_map(|(time, v)| match v.state {
                AddressState::Tried(time) => {
                    let timeout = time + RETRY_TIME;
                    let now_as_sec = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    if timeout < now_as_sec {
                        return Some((v.address.clone(), time, v.services, v.port));
                    }
                    None
                }
                AddressState::Connected => {
                    Some((v.address.clone(), *time as u64, v.services, v.port))
                }
                _ => None,
            })
            .collect();
        addresses
    }
    pub fn get_seeds_from_dns(
        &mut self,
        seed: &str,
        default_port: u16,
    ) -> Result<usize, std::io::Error> {
        let mut addresses = Vec::new();
        let utreexo_seed = seed.contains("x1000000.");
        for ip in dns_lookup::lookup_host(seed)? {
            if let Ok(mut ip) = LocalAddress::try_from(format!("{}:{}", ip, default_port).as_str())
            {
                // This seed returns utreexo nodes
                if utreexo_seed {
                    ip.services |= ServiceFlags::from(1 << 24);
                }
                addresses.push(ip);
            }
        }
        self.push_addresses(&addresses);
        Ok(addresses.len())
    }
    /// Returns a new random address to open a new connection, we try to get addresses with
    /// a set of features supported for our peers
    pub fn get_address_to_connect(
        &mut self,
        flags: ServiceFlags,
        feeler: bool,
    ) -> Option<(usize, LocalAddress)> {
        if self.addresses.is_empty() {
            return None;
        }
        // Feeler connection are used to test if a peer is still alive, we don't care about
        // the features it supports or even if it's a valid peer. The only thing we care about
        // is that we haven't banned it.
        let (id, peer) = if feeler {
            let idx = rand::random::<usize>() % self.addresses.len();
            let peer = self.addresses.keys().nth(idx)?;
            let address = self.addresses.get(peer)?.to_owned();
            if let AddressState::Banned(_) = address.state {
                return None;
            }
            (*peer, address)
        } else if flags.has(ServiceFlags::from(1 << 24)) {
            // if we don't have an utreexo address, we fallback to a normal good address
            self.get_random_utreexo_address()
                .or_else(|| self.get_random_good_address())?
        } else {
            self.get_random_good_address()?
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        match peer.state {
            AddressState::Banned(_) => None,
            AddressState::Connected => self.get_random_good_address(),
            AddressState::NeverTried | AddressState::Tried(_) => Some((id, peer)),
            AddressState::Failed(time) => {
                if now - time > RETRY_TIME {
                    Some((id, peer))
                } else {
                    None
                }
            }
        }
    }
    pub fn dump_peers(&self, datadir: &str) -> std::io::Result<()> {
        let peers: Vec<_> = self
            .addresses
            .values()
            .cloned()
            .map(Into::<DiskLocalAddress>::into)
            .collect::<Vec<_>>();
        let peers = serde_json::to_string(&peers);
        if let Ok(peers) = peers {
            std::fs::write(datadir.to_owned() + "/peers.json", peers)?;
        }
        Ok(())
    }
    fn get_random_good_address(&self) -> Option<(usize, LocalAddress)> {
        if self.good_addresses.is_empty() {
            return None;
        }
        let idx = rand::random::<usize>() % self.good_addresses.len();
        let good_peer = self.good_addresses.get(idx)?;
        Some((*good_peer, self.addresses.get(good_peer)?.to_owned()))
    }
    fn get_random_utreexo_address(&self) -> Option<(usize, LocalAddress)> {
        if self.utreexo_addresses.is_empty() {
            return None;
        }
        let idx = rand::random::<usize>() % self.utreexo_addresses.len();
        let utreexo_peer = self.utreexo_addresses.get(idx)?;
        Some((*utreexo_peer, self.addresses.get(utreexo_peer)?.to_owned()))
    }
    fn get_net_seeds(network: Network) -> &'static str {
        match network {
            Network::Bitcoin => include_str!("seeds/mainnet_seeds.json"),
            Network::Testnet => include_str!("seeds/testnet_seeds.json"),
            Network::Signet => include_str!("seeds/signet_seeds.json"),
            Network::Regtest => include_str!("seeds/regtest_seeds.json"),
        }
    }
    pub fn start_addr_man(
        &mut self,
        datadir: String,
        default_port: u16,
        network: Network,
        dns_seeds: &[&'static str],
    ) -> Result<Vec<LocalAddress>, std::io::Error> {
        let local_db = std::fs::read_to_string(format!("{datadir}/peers.json"));
        let peers = if let Ok(peers) = local_db {
            info!("Peers database found, using it");

            serde_json::from_str::<Vec<DiskLocalAddress>>(&peers)
        } else {
            info!("No peers available, using fixed peers");
            let mut peers_from_dns = 0;
            for seed in dns_seeds {
                peers_from_dns += self.get_seeds_from_dns(seed, default_port)?;
            }
            info!("Got {peers_from_dns} peers from DNS Seeds",);
            let addresses = Self::get_net_seeds(network);
            serde_json::from_str(addresses)
        };
        if let Ok(peers) = peers {
            let peers = peers
                .iter()
                .cloned()
                .map(Into::<LocalAddress>::into)
                .collect::<Vec<_>>();
            self.push_addresses(&peers);
        }
        let anchors = std::fs::read_to_string(format!("{datadir}/anchors.json"));
        if anchors.is_err() {
            return Ok(Vec::new());
        }
        if let Ok(anchors) = serde_json::from_str::<Vec<DiskLocalAddress>>(&anchors.unwrap()) {
            let anchors = anchors
                .iter()
                .cloned()
                .map(Into::<LocalAddress>::into)
                .collect::<Vec<_>>();
            return Ok(anchors);
        }
        Ok(Vec::new())
    }
    /// This function moves addresses between buckets, like if the ban time of a peer expired,
    /// or if we tried to connect to a peer and it failed in the past, but now it might be online
    /// again.
    pub fn rearrange_buckets(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        for (_, address) in self.addresses.iter_mut() {
            match address.state {
                AddressState::Banned(ban_time) => {
                    if ban_time < now {
                        address.state = AddressState::NeverTried;
                    }
                }
                AddressState::Tried(tried_time) => {
                    if tried_time + RETRY_TIME < now {
                        address.state = AddressState::NeverTried;
                    }
                }
                AddressState::Failed(failed_time) => {
                    if failed_time + RETRY_TIME < now {
                        address.state = AddressState::NeverTried;
                    }
                }
                AddressState::Connected | AddressState::NeverTried => {}
            }
        }
    }
    /// Updates the state of an address
    pub fn update_set_state(&mut self, idx: usize, state: AddressState) -> &mut Self {
        if matches!(state, AddressState::Banned(_)) {
            return self;
        }

        match state {
            AddressState::Banned(_) => {
                self.good_addresses.retain(|&x| x != idx);
            }
            AddressState::Tried(_) => {
                if !self.good_addresses.contains(&idx) {
                    self.good_addresses.push(idx);
                }
            }
            AddressState::NeverTried => {
                self.good_addresses.retain(|&x| x != idx);
            }
            AddressState::Connected => {
                if !self.good_addresses.contains(&idx) {
                    self.good_addresses.push(idx);
                }
            }
            AddressState::Failed(_) => {
                self.good_addresses.retain(|&x| x != idx);
            }
        }
        if let Some(address) = self.addresses.get_mut(&idx) {
            address.state = state;
        };
        self
    }
    /// Updates the service flags after we receive a version message
    pub fn update_set_service_flag(&mut self, idx: usize, flags: ServiceFlags) -> &mut Self {
        if let Some(address) = self.addresses.get_mut(&idx) {
            address.services = flags;
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskLocalAddress {
    /// An actual address
    address: Address,
    /// Last time we successfully connected to this peer, only relevant is state == State::Tried
    last_connected: u64,
    /// Our local state for this peer, as defined in AddressState
    state: AddressState,
    /// Network services announced by this peer
    services: u64,
    /// Network port this peers listens to
    port: u16,
    /// An id to identify this address
    id: Option<usize>,
}
impl From<LocalAddress> for DiskLocalAddress {
    fn from(value: LocalAddress) -> Self {
        let address = match value.address {
            AddrV2::Ipv4(ip) => Address::V4(ip),
            AddrV2::Ipv6(ip) => Address::V6(ip),
            AddrV2::Cjdns(ip) => Address::Cjdns(ip),
            AddrV2::I2p(ip) => Address::I2p(ip),
            AddrV2::TorV2(ip) => Address::OnionV2(ip),
            AddrV2::TorV3(ip) => Address::OnionV3(ip),
            AddrV2::Unknown(_, _) => Address::V4(Ipv4Addr::LOCALHOST),
        };

        DiskLocalAddress {
            address,
            last_connected: value.last_connected,
            state: if value.state == AddressState::Connected {
                AddressState::Tried(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                )
            } else {
                value.state
            },
            services: value.services.to_u64(),
            port: value.port,
            id: Some(value.id),
        }
    }
}
impl From<DiskLocalAddress> for LocalAddress {
    fn from(value: DiskLocalAddress) -> Self {
        let address = match value.address {
            Address::V4(ip) => AddrV2::Ipv4(ip),
            Address::V6(ip) => AddrV2::Ipv6(ip),
            Address::Cjdns(ip) => AddrV2::Cjdns(ip),
            Address::I2p(ip) => AddrV2::I2p(ip),
            Address::OnionV2(ip) => AddrV2::TorV2(ip),
            Address::OnionV3(ip) => AddrV2::TorV3(ip),
        };
        let services = ServiceFlags::from(value.services);
        LocalAddress {
            address,
            last_connected: value.last_connected,
            state: value.state,
            services,
            port: value.port,
            id: value.id.unwrap_or_else(rand::random::<usize>),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Address {
    /// Regular ipv4 address
    V4(Ipv4Addr),
    /// Regular ipv6 address
    V6(Ipv6Addr),
    /// Tor v2 address, this may never be used, as OnionV2 is deprecated
    /// but we'll keep it here for completeness sake
    OnionV2([u8; 10]),
    /// Tor v3 address. This is the preferred way to connect to a tor node
    OnionV3([u8; 32]),
    /// Cjdns ipv6 address
    Cjdns(Ipv6Addr),
    /// I2p address, a 32 byte node key
    I2p([u8; 32]),
}

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
use floresta_chain::DnsSeed;
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
    peers_by_service: HashMap<ServiceFlags, Vec<usize>>,
}

impl AddressMan {
    /// Add a new address to our list of known address
    pub fn push_addresses(&mut self, addresses: &[LocalAddress]) {
        for address in addresses {
            let id = address.id;

            // don't add addresses that don't have the minimum required services
            if !address.services.has(ServiceFlags::WITNESS)
                | !address.services.has(ServiceFlags::NETWORK)
            {
                continue;
            }

            // don't add private addresses
            if Self::is_localhost(address) || Self::is_private(address) {
                continue;
            }

            // don't add duplicate addresses
            if self
                .addresses
                .values()
                .any(|x| x.address == address.address)
            {
                continue;
            }

            if let std::collections::hash_map::Entry::Vacant(e) = self.addresses.entry(id) {
                e.insert(address.to_owned());
                // For now we assume that all addresses are valid, until proven otherwise.
                self.good_addresses.push(id);

                self.push_if_has_service(address, ServiceFlags::UTREEXO);
                self.push_if_has_service(address, ServiceFlags::from(1 << 25)); // UTREEXO_FILTER
                self.push_if_has_service(address, ServiceFlags::NONE); // this means any peer
                self.push_if_has_service(address, ServiceFlags::COMPACT_FILTERS);
            }
        }
    }

    fn is_private(address: &LocalAddress) -> bool {
        match address.address {
            AddrV2::Ipv4(ip) => ip.is_private(),
            AddrV2::Ipv6(ip) => ip.octets()[0] == 0xfd || ip.octets()[0] == 0xfe,
            _ => false,
        }
    }

    fn is_localhost(address: &LocalAddress) -> bool {
        match address.address {
            AddrV2::Ipv4(ip) => ip.is_loopback(),
            AddrV2::Ipv6(ip) => ip.is_loopback(),
            _ => false,
        }
    }

    fn push_if_has_service(&mut self, address: &LocalAddress, service: ServiceFlags) {
        if address.services.has(service) {
            self.peers_by_service
                .entry(service)
                .or_default()
                .push(address.id);
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

    fn do_lookup(
        address: &str,
        default_port: u16,
    ) -> Result<Vec<LocalAddress>, dns_lookup::LookupError> {
        let mut addresses = Vec::new();
        for ip in dns_lookup::lookup_host(address)? {
            if let Ok(ip) = LocalAddress::try_from(format!("{}:{}", ip, default_port).as_str()) {
                addresses.push(ip);
            }
        }

        Ok(addresses)
    }

    fn get_seeds_from_dns(
        &mut self,
        seed: &DnsSeed,
        default_port: u16,
    ) -> Result<usize, std::io::Error> {
        let mut seed_address_count = 0;

        // ask for utreexo peers (if filtering is available)
        if seed.filters.has(ServiceFlags::UTREEXO) {
            let address = format!("x1000000.{}", seed.seed);
            let _addresses = Self::do_lookup(&address, default_port).unwrap_or_default();
            seed_address_count += _addresses.len();
            _addresses
                .into_iter()
                .map(|mut x| {
                    x.services =
                        ServiceFlags::UTREEXO | ServiceFlags::NETWORK | ServiceFlags::WITNESS;
                    x
                })
                .for_each(|x| {
                    self.push_addresses(&[x]);
                });
        }

        // ask for compact filter peers (if filtering is available)
        if seed.filters.has(ServiceFlags::COMPACT_FILTERS) {
            let address = format!("x49.{}", seed.seed);
            let _addresses = Self::do_lookup(&address, default_port).unwrap_or_default();
            seed_address_count += _addresses.len();
            _addresses
                .into_iter()
                .map(|mut x| {
                    x.services = ServiceFlags::COMPACT_FILTERS
                        | ServiceFlags::NETWORK
                        | ServiceFlags::WITNESS;
                    x
                })
                .for_each(|x| {
                    self.push_addresses(&[x]);
                });
        }

        // ask for any peer (if filtering is available)
        if seed.filters.has(ServiceFlags::WITNESS) {
            let address = format!("x9.{}", seed.seed);
            let _addresses = Self::do_lookup(&address, default_port).unwrap_or_default();
            seed_address_count += _addresses.len();
            _addresses
                .into_iter()
                .map(|mut x| {
                    x.services = ServiceFlags::NETWORK | ServiceFlags::WITNESS;
                    x
                })
                .for_each(|x| {
                    self.push_addresses(&[x]);
                });
        }

        Ok(seed_address_count)
    }

    /// Returns a new random address to open a new connection, we try to get addresses with
    /// a set of features supported for our peers
    pub fn get_address_to_connect(
        &mut self,
        required_service: ServiceFlags,
        feeler: bool,
    ) -> Option<(usize, LocalAddress)> {
        if self.addresses.is_empty() {
            return None;
        }

        // Feeler connection are used to test if a peer is still alive, we don't care about
        // the features it supports or even if it's a valid peer. The only thing we care about
        // is that we haven't banned it.
        if feeler {
            let idx = rand::random::<usize>() % self.addresses.len();
            let peer = self.addresses.keys().nth(idx)?;
            let address = self.addresses.get(peer)?.to_owned();
            if let AddressState::Banned(_) = address.state {
                return None;
            }
            return Some((*peer, address));
        };

        let (id, peer) = self.get_address_by_service(required_service)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match peer.state {
            AddressState::Banned(_) | AddressState::Connected => None,
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
    fn get_address_by_service(&self, service: ServiceFlags) -> Option<(usize, LocalAddress)> {
        let peers = self.peers_by_service.get(&service)?;
        if peers.is_empty() {
            return None;
        }

        let idx = rand::random::<usize>() % peers.len();
        let utreexo_peer = peers.get(idx)?;
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
        dns_seeds: &[DnsSeed],
    ) -> Result<Vec<LocalAddress>, std::io::Error> {
        let persisted_peers = std::fs::read_to_string(format!("{datadir}/peers.json"))
            .map(|seeds| serde_json::from_str::<Vec<DiskLocalAddress>>(&seeds));

        let persisted_peers = match persisted_peers {
            Ok(Ok(peers)) => peers,
            _ => {
                let addresses = Self::get_net_seeds(network);
                serde_json::from_str(addresses).expect("BUG: fixed peers are invalid")
            }
        };

        let persisted_peers = persisted_peers
            .iter()
            .cloned()
            .map(Into::<LocalAddress>::into)
            .collect::<Vec<_>>();
        self.push_addresses(&persisted_peers);

        let mut peers_from_dns = 0;
        for seed in dns_seeds {
            match self.get_seeds_from_dns(seed, default_port) {
                Ok(peers) => peers_from_dns += peers,
                Err(e) => {
                    info!("Error getting peers from DNS seed {}: {e:?}", seed.seed);
                }
            }
        }

        info!("Got {peers_from_dns} peers from DNS Seeds",);

        let anchors = std::fs::read_to_string(format!("{datadir}/anchors.json"))?;
        let anchors = serde_json::from_str::<Vec<DiskLocalAddress>>(&anchors)?;
        let anchors = anchors
            .iter()
            .cloned()
            .map(Into::<LocalAddress>::into)
            .collect::<Vec<_>>();

        Ok(anchors)
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

#[cfg(test)]
mod test {
    use std::fs::File;
    use std::io::Read;
    use std::io::{self};
    use std::net::Ipv4Addr;

    use bitcoin::p2p::address::AddrV2;
    use bitcoin::p2p::ServiceFlags;
    use floresta_chain::get_chain_dns_seeds;
    use floresta_chain::Network;
    use rand::Rng;
    use serde::Deserialize;
    use serde::Serialize;

    use super::AddressState;
    use super::LocalAddress;
    use crate::address_man::AddressMan;

    /// Seed Data for paesing in tests.
    #[derive(Debug, Clone, PartialEq, Deserialize)]
    pub struct SeedData {
        /// An actual address
        address: SeedAddress,
        /// Last time we successfully connected to this peer, only relevant is state == State::Tried
        last_connected: u64,
        /// Our local state for this peer, as defined in AddressState
        state: AddressState,
        /// Network services announced by this peer
        pub services: u64,
        /// Network port this peers listens to
        port: u16,
    }

    #[allow(non_snake_case)]
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    struct SeedAddress {
        v4: Ipv4Addr,
    }

    fn load_addresses_from_json(file_path: &str) -> io::Result<Vec<LocalAddress>> {
        let mut contents = String::new();
        File::open(file_path)?.read_to_string(&mut contents)?;

        let seeds: Vec<SeedData> =
            serde_json::from_str(&contents).expect("JSON not well-formatted");
        let mut addresses = Vec::new();
        let mut rng = rand::thread_rng();

        for seed in seeds {
            let state = match seed.state {
                AddressState::Tried(time) => AddressState::Tried(time),
                _ => continue,
            };

            let _address = AddrV2::Ipv4(seed.address.v4);

            let local_address = LocalAddress {
                address: _address,
                last_connected: seed.last_connected,
                state,
                services: ServiceFlags::from(seed.services),
                port: seed.port,
                id: rng.gen(),
            };
            addresses.push(local_address);
        }

        Ok(addresses)
    }
    #[test]
    fn test_parse() {
        let signet_address =
            load_addresses_from_json("./src/p2p_wire/seeds/signet_seeds.json").unwrap();

        assert!(!signet_address.is_empty());
        let random = rand::thread_rng().gen_range(1..=13);
        let loc_adr_1 = LocalAddress::from(signet_address[random].address.clone());
        assert_eq!(loc_adr_1.address, signet_address[random].address);
    }
    #[test]
    fn test_address_man() {
        let mut address_man = AddressMan::default();

        let signet_address =
            load_addresses_from_json("./src/p2p_wire/seeds/signet_seeds.json").unwrap();

        address_man.push_addresses(&signet_address);

        assert!(!address_man.good_addresses.is_empty());

        assert!(!address_man.peers_by_service.is_empty());

        assert!(!address_man.get_addresses_to_send().is_empty());

        assert!(address_man
            .get_address_to_connect(ServiceFlags::default(), true)
            .is_some());

        assert!(address_man
            .get_address_to_connect(ServiceFlags::default(), false)
            .is_some());

        assert!(address_man
            .get_address_to_connect(ServiceFlags::NONE, false)
            .is_some());

        assert!(address_man
            .get_address_to_connect(ServiceFlags::UTREEXO, false)
            .is_some());

        assert!(!AddressMan::get_net_seeds(Network::Signet).is_empty());
        assert!(!AddressMan::get_net_seeds(Network::Bitcoin).is_empty());
        assert!(!AddressMan::get_net_seeds(Network::Regtest).is_empty());
        assert!(!AddressMan::get_net_seeds(Network::Testnet).is_empty());

        assert!(address_man
            .get_seeds_from_dns(&get_chain_dns_seeds(Network::Signet)[0], 8333)
            .is_ok());

        address_man.rearrange_buckets();
    }
}

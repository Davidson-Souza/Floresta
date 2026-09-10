// SPDX-License-Identifier: MIT OR Apache-2.0

//! Address manager is a module that keeps track of known peer addresses and associated
//! metadata. This module is very important in keeping our node protected against targeted
//! attacks, like eclipse attacks.

use core::fmt::Display;
use core::net::IpAddr;
use core::net::Ipv4Addr;
use core::net::Ipv6Addr;
use core::net::SocketAddr;
use core::str::FromStr;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::read_to_string;
use std::path::Path;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use bitcoin::Network;
use bitcoin::p2p::ServiceFlags;
use bitcoin::p2p::address::AddrV2;
use bitcoin::p2p::address::AddrV2Message;
use floresta_chain::DnsSeed;
use floresta_common::service_flags;
use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::seq::IteratorRandom;
use serde::Deserialize;
use serde::Serialize;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::bitcoin_socket_addr::BitcoinSocketAddr;
use crate::bitcoin_socket_addr::InvalidAddressError;
use crate::onion::OnionV3Addr;

/// Time constant used to reduce the weight of addresses that were attempted recently.
const RETRY_TAU: f64 = 10.0 * 60.0; // 10 minutes

/// Time constant used to restore the weight of addresses after a failed attempt.
const FAILURE_TAU: f64 = 6.0 * 60.0 * 60.0; // 6 hours

/// Minimum retry score assigned to an address attempted just now.
const RETRY_FLOOR: f64 = 0.01;

/// Minimum result score assigned to an address whose latest attempt failed.
const FAILURE_FLOOR: f64 = 0.05;

/// Reliability multiplier for an address with a previously successful connection.
const TRIED_PEER_BONUS: f64 = 10.0;

/// Multiplier applied for each connected or in-flight peer in an address's network group.
const NETWORK_GROUP_PENALTY: f64 = 0.05;

/// Probability mass reserved for compact-filter-capable peers.
const COMPACT_FILTERS_WEIGHT: f64 = 0.20;

/// Probability mass reserved for Utreexo-capable peers.
const UTREEXO_WEIGHT: f64 = 0.20;

/// Probability mass kept for the unconditioned peer distribution.
const MIXED_PEERS_WEIGHT: f64 = 0.60;

/// The minimum amount of addresses we need to have on the [`AddressMan`].
const MIN_ADDRESSES: usize = 15;

/// The minimum amount of CBF-capable addresses we need to have on the [`AddressMan`].
const MIN_ADDRESSES_CBF: usize = 5;

/// The minimum amount of Utreexo-capable addresses we need to have on the [`AddressMan`].
const MIN_ADDRESSES_UTREEXO: usize = 2;

/// If we haven't heard from a peer in this amount of time, we consider its info stale
/// and add it to the NeverTried bucket
const ASSUME_STALE: u64 = 24 * 60 * 60; // 24 hours

/// How many addresses we keep in our address manager
const MAX_ADDRESSES: usize = 50_000;

/// A type alias for a list of addresses to send to our peers
type AddressToSend = Vec<(AddrV2, u64, ServiceFlags, u16)>;

/// A network group used to keep one overrepresented network from dominating peer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AddressGroup {
    /// IPv4 addresses are grouped by their /16 prefix.
    Ipv4(u16),
    /// IPv6 addresses are grouped by their /32 prefix, the IPv6 analogue used here for /16s.
    Ipv6(u32),
    /// Overlay-network addresses have no meaningful IP subnet and form individual groups.
    Overlay(usize),
}

#[derive(Debug, Clone, Copy)]
struct SelectionCandidate {
    id: usize,
    group: AddressGroup,
    health: f64,
    services: ServiceFlags,
}

#[derive(Debug, Default)]
struct AddressGroupStats {
    represented: usize,
    total_health: f64,
    maximum_health: f64,
    selection_weight: f64,
}

#[derive(Debug, Copy, Clone, PartialEq, Deserialize, Serialize)]
/// A local state for how we see this peer. It helps us during peer selection,
/// by keeping track of our past encounters with this node (if any),
/// helping us to find live peers more easily, and avoid troublesome peers.
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

/// All the networks we might receive addresses for
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReachableNetworks {
    IPv4,
    IPv6,
    TorV3,
    I2P,
    Cjdns,
}

impl ReachableNetworks {
    /// All networks Floresta is aware of.
    pub const ALL: [Self; 5] = [Self::IPv4, Self::IPv6, Self::TorV3, Self::I2P, Self::Cjdns];

    /// Networks this implementation currently supports.
    pub const SUPPORTED: [Self; 3] = [Self::IPv4, Self::IPv6, Self::TorV3];
}

impl Display for ReachableNetworks {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IPv4 => write!(f, "ipv4"),
            Self::IPv6 => write!(f, "ipv6"),
            Self::TorV3 => write!(f, "onion"),
            Self::I2P => write!(f, "i2p"),
            Self::Cjdns => write!(f, "cjdns"),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
/// Per-network address manager statistics.
pub struct NetworkStats {
    /// Number of new (untried) addresses.
    pub new: u64,

    /// Number of tried addresses.
    pub tried: u64,
}

impl NetworkStats {
    pub const fn total(&self) -> u64 {
        self.new + self.tried
    }
}

#[derive(Debug, Default, Clone, Copy)]
/// Address manager statistics broken down by network.
///
/// Differently from
pub struct ConnectionStats {
    pub ipv4: NetworkStats,
    pub ipv6: NetworkStats,
    pub onion: NetworkStats,
    pub i2p: NetworkStats,
    pub cjdns: NetworkStats,
}

#[derive(Debug, Clone, PartialEq)]
/// How do we store peers locally
pub struct LocalAddress {
    /// An actual address
    address: BitcoinSocketAddr,

    /// Last time we successfully connected to this peer, only relevant is state == State::Tried
    last_connected: u64,

    /// Our local state for this peer, as defined in AddressState
    state: AddressState,

    /// Network services announced by this peer
    services: ServiceFlags,

    /// Random id for this peer
    pub id: usize,
}

impl Display for LocalAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr: String = self
            .get_net_address()
            .map(|ip| ip.to_string())
            .unwrap_or(String::from("unknown"));

        write!(f, "{}:{}", addr, self.address.get_port())
    }
}

impl From<BitcoinSocketAddr> for LocalAddress {
    fn from(value: BitcoinSocketAddr) -> Self {
        Self {
            address: value,
            last_connected: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            state: AddressState::NeverTried,
            services: ServiceFlags::NONE,

            // TODO(Davidson): Refactor `id` to be a `u64`?
            id: rand::random::<u64>() as usize,
        }
    }
}

impl From<AddrV2Message> for LocalAddress {
    fn from(value: AddrV2Message) -> Self {
        let socket_addr = BitcoinSocketAddr::new(value.addr, value.port);
        Self {
            address: socket_addr,
            last_connected: value.time.into(),
            state: AddressState::NeverTried,
            services: value.services,
            // TODO(Davidson): Refactor `id` to be a `u64`?
            id: rand::random::<u64>() as usize,
        }
    }
}

impl FromStr for LocalAddress {
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let address = value.parse::<BitcoinSocketAddr>()?;

        Ok(Self::new(
            address,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            super::address_man::AddressState::NeverTried,
            ServiceFlags::NONE,
            // TODO(Davidson): Refactor `id` to be a `u64`?
            rand::random::<u64>() as usize,
        ))
    }

    type Err = InvalidAddressError;
}

impl LocalAddress {
    pub fn new(
        address: BitcoinSocketAddr,
        last_connected: u64,
        state: AddressState,
        services: ServiceFlags,
        id: usize,
    ) -> Self {
        Self {
            address,
            last_connected,
            state,
            services,
            id,
        }
    }

    /// Get the [`SocketAddr`] for this [`LocalAddress`].
    pub fn as_bitcoin_socket_addr(&self) -> &BitcoinSocketAddr {
        &self.address
    }

    /// Get the `port` for this [`LocalAddress`].
    pub fn get_port(&self) -> u16 {
        self.address.get_port()
    }

    /// Set the `port` for this [`LocalAddress`].
    pub fn set_port(&mut self, port: u16) {
        self.address.set_port(port);
    }

    /// Get the [`ServiceFlags`] for this [`LocalAddress`].
    pub fn get_services(&self) -> ServiceFlags {
        self.services
    }

    /// Set the [`ServiceFlags`] for this [`LocalAddress`].
    pub fn set_services(&mut self, services: ServiceFlags) {
        self.services = services;
    }

    pub fn get_socket_addr(&self) -> Option<SocketAddr> {
        let ip = match self.as_addrv2() {
            AddrV2::Ipv6(ipv6) => IpAddr::V6(*ipv6),
            AddrV2::Ipv4(ipv4) => IpAddr::V4(*ipv4),

            _ => return None,
        };

        let port = self.get_port();
        Some(SocketAddr::new(ip, port))
    }

    /// Returns the inner [`OnionV3Addr`] if this is an [`AddrV2::TorV3`].
    ///
    /// Returns [`None`] if this is another address kind.
    pub fn get_onion_addr(&self) -> Option<OnionV3Addr> {
        if let AddrV2::TorV3(addr) = self.as_addrv2() {
            return Some(OnionV3Addr::from(*addr));
        }

        None
    }

    /// Return an IP address associated with this peer address, if any
    ///
    /// Note: For domain-based addresses such as Tor and I2P this function will return [`None`]
    pub const fn get_net_address(&self) -> Option<IpAddr> {
        match *self.address.as_addrv2() {
            // IPV4
            AddrV2::Ipv4(ipv4) => Some(IpAddr::V4(ipv4)),

            // IPV6
            AddrV2::Ipv6(ipv6) => Some(IpAddr::V6(ipv6)),

            // CJDNS
            AddrV2::Cjdns(cjdns) => Some(IpAddr::V6(cjdns)),

            // Others - These are domain-based, and can't be encoded as an IP address
            _ => None,
        }
    }

    /// Returns the network group used for peer diversity.
    fn address_group(&self) -> AddressGroup {
        match self.address.as_addrv2() {
            AddrV2::Ipv4(ip) => {
                let octets = ip.octets();
                AddressGroup::Ipv4(u16::from_be_bytes([octets[0], octets[1]]))
            }
            AddrV2::Ipv6(ip) => {
                let octets = ip.octets();
                AddressGroup::Ipv6(u32::from_be_bytes([
                    octets[0], octets[1], octets[2], octets[3],
                ]))
            }
            AddrV2::Cjdns(_) | AddrV2::I2p(_) | AddrV2::TorV2(_) | AddrV2::TorV3(_) => {
                AddressGroup::Overlay(self.id)
            }
            AddrV2::Unknown(_, _) => AddressGroup::Overlay(self.id),
        }
    }

    /// Computes this address's health score at `now`.
    ///
    /// Connected and banned addresses are ineligible. Never-tried addresses start at the
    /// baseline weight. A previously successful address receives a reliability bonus after its
    /// retry cooldown, while a recent failure receives both the retry and failure penalties.
    fn selection_health(&self, now: u64) -> Option<f64> {
        fn recovery_score(age: u64, time_constant: f64, floor: f64) -> f64 {
            floor + (1.0 - floor) * (1.0 - (-(age as f64) / time_constant).exp())
        }

        match self.state {
            AddressState::Banned(_) | AddressState::Connected => None,
            AddressState::NeverTried => Some(1.0),
            AddressState::Tried(last_attempt) => Some(
                TRIED_PEER_BONUS
                    * recovery_score(now.saturating_sub(last_attempt), RETRY_TAU, RETRY_FLOOR),
            ),
            AddressState::Failed(last_attempt) => {
                let age = now.saturating_sub(last_attempt);
                let retry_score = recovery_score(age, RETRY_TAU, RETRY_FLOOR);
                let failure_score = recovery_score(age, FAILURE_TAU, FAILURE_FLOOR);
                Some(retry_score * failure_score)
            }
        }
    }

    /// Returns a reference to the inner [`BitcoinSocketAddr`].
    pub fn as_addrv2(&self) -> &AddrV2 {
        self.address.as_addrv2()
    }

    /// Converts this [`LocalAddress`] into an [`AddrV2`]
    pub fn into_addrv2(self) -> AddrV2 {
        self.address.into_addrv2()
    }

    /// Return whether the address can be reached from our node
    ///
    /// Some addresses are not reachable from the global internet,
    /// those includes documentation, reserved and private ones ranges.
    /// Since we can't connect with them, there's no point into keeping them
    const fn is_routable(&self) -> bool {
        match self.address.as_addrv2() {
            AddrV2::Ipv4(ipv4) => Self::is_routable_ipv4(ipv4),
            AddrV2::Ipv6(ipv6) => Self::is_routable_ipv6(ipv6),
            AddrV2::Cjdns(address) => {
                let octets = address.octets();
                // CJDNS addresses use a special range for local addresses (FC00::/8)
                // See: https://github.com/cjdelisle/cjdns/tree/master/doc#what-is-notable-about-cjdns-why-should-i-use-it
                octets[0] == 0xFC
            }
            _ => true,
        }
    }

    /// Returns whether an ipv4 address is publicly routable
    const fn is_routable_ipv4(ip: &Ipv4Addr) -> bool {
        // Code taken from bitcoinfuzz commit: 7619d400bbd8078b8dc51d077c900f0b54f9cfcf/
        let octets = ip.octets();

        // 0.0.0.0/8 - "This" network
        if octets[0] == 0 {
            return false;
        }

        // Loopback, broadcast, private (RFC 1918)
        if ip.is_loopback() || ip.is_broadcast() || ip.is_private() {
            return false;
        }

        // RFC 2544 - Benchmarking - 198.18.0.0/15
        if octets[0] == 198 && (octets[1] == 18 || octets[1] == 19) {
            return false;
        }

        // RFC 3927 - Link-Local - 169.254.0.0/16
        if ip.is_link_local() {
            return false;
        }

        // RFC 6598 - Shared Address Space (CGNAT) - 100.64.0.0/10
        if octets[0] == 100 && (octets[1] >= 64 && octets[1] <= 127) {
            return false;
        }

        // RFC 5737 - Documentation (TEST-NET-1, TEST-NET-2, TEST-NET-3)
        if ip.is_documentation() {
            return false;
        }

        true
    }

    /// Returns whether an ipv6 address is publicly routable
    #[rustfmt::skip]
    const fn is_routable_ipv6(ip: &Ipv6Addr) -> bool {
        let octets = ip.octets();

        // Unspecified, loopback, unique local (RFC 4193 - fc00::/7)
        if ip.is_unspecified() || ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00 {
            return false;
        }

        // RFC 4843 - ORCHID - 2001:10::/28
        if octets[0] == 0x20 && octets[1] == 0x01 && octets[2] == 0x00 && (octets[3] & 0xF0) == 0x10 {
            return false;
        }

        // RFC 4862 - Link-local - fe80::/64
        if octets[0] == 0xFE && (octets[1] & 0xC0) == 0x80 {
            return false;
        }

        // RFC 7343 - ORCHIDv2 - 2001:20::/28
        if octets[0] == 0x20 && octets[1] == 0x01 && octets[2] == 0x00 && (octets[3] & 0xf0) == 0x20 {
            return false;
        }

        true
    }

    /// Return whether an address is good to connect to
    pub fn is_good_address(&self) -> bool {
        if !self.is_routable() {
            return false;
        }

        matches!(self.state, AddressState::Connected)
            || matches!(self.state, AddressState::Tried(_))
    }
}

#[derive(Clone)]
/// A module that keeps track of known addresses and chooses addresses that our node can connect
pub struct AddressMan {
    /// A map of all peers we know, mapping the address id to the actual address.
    addresses: HashMap<usize, LocalAddress>,

    /// All indexes of "good" addresses
    ///
    /// Good peers are those which we think are live, and haven't banned yet.
    /// If we try to connect with one peer, and the connection doesn't succeed,
    /// this peer is assumed to be down and removed from good addresses for some time.
    good_addresses: Vec<usize>,

    /// A map of a set of good peers indexes by their [`ServiceFlags`]
    ///
    /// We use this to make peer selection, if we are looking for a specific kind of peer (like utreexo or CBF peers)
    good_peers_by_service: HashMap<ServiceFlags, Vec<usize>>,

    /// A map of a set of peers indexes by their [`ServiceFlags`]
    ///
    /// This works similarly to `good_peers_by_service`. However, we keep all peers here, not only good peers
    peers_by_service: HashMap<ServiceFlags, Vec<usize>>,

    /// Addresses with an in-flight connection attempt.
    ///
    /// These are excluded from subsequent draws, which makes sequential selection operate
    /// without replacement. Their network groups still count toward the diversity penalty.
    attempting: HashSet<usize>,

    /// The maximum number of entries this address manager can hold
    max_size: usize,

    /// The networks we can reach
    reachable_networks: HashSet<ReachableNetworks>,
}

impl AddressMan {
    /// Creates a new address manager
    ///
    /// `max_size` is the maximum number of addresses to keep in memory. If None is provided,
    /// a default of 50,000 addresses is used.
    pub fn new(max_size: Option<usize>, reachable_networks: &[ReachableNetworks]) -> Self {
        let reachable_networks: HashSet<ReachableNetworks> =
            reachable_networks.iter().cloned().collect();

        Self {
            addresses: HashMap::new(),
            good_addresses: Vec::new(),
            good_peers_by_service: HashMap::new(),
            peers_by_service: HashMap::new(),
            attempting: HashSet::new(),
            max_size: max_size.unwrap_or(MAX_ADDRESSES),
            reachable_networks,
        }
    }

    /// Returns the current timestamp since the epoch
    fn time_since_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Add a new address to our list of known address
    pub fn push_addresses(&mut self, addresses: &[LocalAddress]) {
        for address in addresses {
            let id = address.id;
            // don't add addresses that don't have the minimum required services
            if !address.services.has(ServiceFlags::WITNESS)
                || !address.services.has(ServiceFlags::NETWORK_LIMITED)
            {
                continue;
            }

            // don't add addresses from networks we can't reach
            if !self.is_net_reachable(address) {
                continue;
            }

            if !address.is_routable() {
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
                e.insert(address.clone());
                if address.is_good_address() {
                    self.good_addresses.push(id);
                }

                self.push_if_has_service(address, service_flags::UTREEXO.into());
                self.push_if_has_service(address, ServiceFlags::NONE); // this means any peer
                self.push_if_has_service(address, ServiceFlags::COMPACT_FILTERS);
            }
        }

        // Open up space by pruning old addresses
        self.prune_addresses();
    }

    /// Check if we can reach this address based on our reachable networks
    fn is_net_reachable(&self, address: &LocalAddress) -> bool {
        match address.address.as_addrv2() {
            AddrV2::Ipv4(_) => self.reachable_networks.contains(&ReachableNetworks::IPv4),
            AddrV2::Ipv6(_) => self.reachable_networks.contains(&ReachableNetworks::IPv6),
            AddrV2::TorV3(_) => self.reachable_networks.contains(&ReachableNetworks::TorV3),
            AddrV2::I2p(_) => self.reachable_networks.contains(&ReachableNetworks::I2P),
            AddrV2::Cjdns(_) => self.reachable_networks.contains(&ReachableNetworks::Cjdns),
            _ => false,
        }
    }

    /// Remove addresses that we last heard of, until we are under the limit
    /// of addresses to keep.
    fn prune_addresses(&mut self) {
        let excess = self.addresses.len().saturating_sub(self.max_size);
        if excess == 0 {
            return;
        }

        let mut oldest_ids: Vec<_> = self
            .addresses
            .iter()
            .map(|(&id, addr)| (id, addr.last_connected))
            .collect();

        oldest_ids.sort_by_key(|&(_, last_connected)| last_connected);

        for (oldest_id, _) in oldest_ids.into_iter().take(excess) {
            self.addresses.remove(&oldest_id);
            self.attempting.remove(&oldest_id);
            self.good_addresses.retain(|&x| x != oldest_id);
            for peers in self.good_peers_by_service.values_mut() {
                peers.retain(|&x| x != oldest_id);
            }
            for peers in self.peers_by_service.values_mut() {
                peers.retain(|&x| x != oldest_id);
            }
        }
    }

    /// Return addresses from the [`AddressMan`] filtered by their [`ServiceFlags`].
    fn get_addresses_by_service(&self, service: ServiceFlags) -> Vec<LocalAddress> {
        self.good_peers_by_service
            .get(&service)
            .map(|peer_ids| {
                peer_ids
                    .iter()
                    .filter_map(|id| self.addresses.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns address manager statistics broken down by network type.
    pub(crate) fn get_connection_stats(&self) -> ConnectionStats {
        let mut stats = ConnectionStats::default();

        for addr in self.addresses.values() {
            let bucket = match &addr.address.as_addrv2() {
                AddrV2::Ipv4(_) => &mut stats.ipv4,
                AddrV2::Ipv6(_) => &mut stats.ipv6,
                AddrV2::TorV3(_) => &mut stats.onion,
                AddrV2::I2p(_) => &mut stats.i2p,
                AddrV2::Cjdns(_) => &mut stats.cjdns,
                _ => continue,
            };

            match addr.state {
                AddressState::Banned(_) => continue,
                AddressState::Tried(_) | AddressState::Connected => bucket.tried += 1,
                AddressState::NeverTried | AddressState::Failed(_) => bucket.new += 1,
            }
        }

        stats
    }

    /// Check if we have enough addresses on the address manager.
    #[rustfmt::skip]
    pub fn enough_addresses(&self) -> bool {
        if self.good_addresses.len() < MIN_ADDRESSES {
            return false;
        }

        if self.get_addresses_by_service(ServiceFlags::COMPACT_FILTERS).len() < MIN_ADDRESSES_CBF {
            return false;
        }

        if self.get_addresses_by_service(service_flags::UTREEXO.into()).len() < MIN_ADDRESSES_UTREEXO {
            return false;
        }

        true
    }

    fn push_if_has_service(&mut self, address: &LocalAddress, service: ServiceFlags) {
        if !address.services.has(service) {
            return;
        }

        let addresses = self.peers_by_service.entry(service).or_default();
        if !addresses.contains(&address.id) {
            self.peers_by_service
                .entry(service)
                .or_default()
                .push(address.id);
        }

        let addresses = self.good_peers_by_service.entry(service).or_default();
        if !addresses.contains(&address.id) && address.is_good_address() {
            self.good_peers_by_service
                .entry(service)
                .or_default()
                .push(address.id);
        }
    }

    pub fn get_addresses_to_send(&self) -> AddressToSend {
        self.good_addresses
            .iter()
            .filter_map(|id| {
                let address = self.addresses.get(id)?;
                Some((
                    address.address.as_addrv2().clone(),
                    address.last_connected,
                    address.services,
                    address.get_port(),
                ))
            })
            .collect()
    }

    fn dns_seed_address(
        ip: IpAddr,
        default_port: u16,
        services: ServiceFlags,
        now: u64,
    ) -> LocalAddress {
        let address = match ip {
            IpAddr::V4(ip) => AddrV2::Ipv4(ip),
            IpAddr::V6(ip) => AddrV2::Ipv6(ip),
        };
        let mut address = LocalAddress::from(BitcoinSocketAddr::new(address, default_port));
        address.services = services;
        address.state = AddressState::Tried(now);
        address
    }

    fn do_lookup(
        host: &str,
        default_port: u16,
        socks5: Option<SocketAddr>,
        services: ServiceFlags,
    ) -> Vec<LocalAddress> {
        let ips = match socks5 {
            Some(proxy) => {
                debug!("Performing DNS lookup for host: {host}, using SOCKS5 proxy: {proxy}");
                // SOCKS5 proxy lookup (proxied DNS-over-HTTPS).
                dns_proxy::lookup_host_via_proxy(host, proxy).unwrap_or_else(|e| {
                    error!("DNS lookup via SOCKS5 proxy failed: {e}");
                    Vec::new()
                })
            }
            None => {
                debug!("Performing DNS lookup for host: {host}, using the system resolver");
                // System lookup (usually unencrypted, resolver sees both query and our IP).
                dns_lookup::lookup_host(host).unwrap_or_else(|e| {
                    error!("DNS lookup failed: {e}");
                    Vec::new()
                })
            }
        };

        if ips.is_empty() {
            warn!("No peer addresses read from DNS host: {host}");
        } else {
            info!("Fetched {} peer addresses from DNS host: {host}", ips.len());
        }

        let now = Self::time_since_unix();
        ips.into_iter()
            .map(|ip| Self::dns_seed_address(ip, default_port, services, now))
            .collect()
    }

    pub fn get_seeds_from_dns(
        seed: &DnsSeed,
        default_port: u16,
        socks5: Option<SocketAddr>,
    ) -> Result<Vec<LocalAddress>, std::io::Error> {
        let mut addresses = Vec::new();

        // ask for utreexo peers (if filtering is available)
        if seed.filters.has(service_flags::UTREEXO.into()) {
            let host = format!("x1000.{}", seed.seed);
            let services = ServiceFlags::NETWORK_LIMITED
                | service_flags::UTREEXO.into()
                | ServiceFlags::WITNESS;
            addresses.extend(Self::do_lookup(&host, default_port, socks5, services));
        }

        // ask for compact filter peers (if filtering is available)
        if seed.filters.has(ServiceFlags::COMPACT_FILTERS) {
            let host = format!("x49.{}", seed.seed);
            let services = ServiceFlags::COMPACT_FILTERS
                | ServiceFlags::NETWORK_LIMITED
                | ServiceFlags::WITNESS;
            addresses.extend(Self::do_lookup(&host, default_port, socks5, services));
        }

        // ask for any peer (if filtering is available)
        if seed.filters.has(ServiceFlags::WITNESS) {
            let host = format!("x9.{}", seed.seed);
            let services = ServiceFlags::NETWORK_LIMITED | ServiceFlags::WITNESS;
            addresses.extend(Self::do_lookup(&host, default_port, socks5, services));
        }

        // ask for any peer (if filtering isn't available)
        if seed.filters == ServiceFlags::NONE {
            let services = ServiceFlags::NETWORK_LIMITED | ServiceFlags::WITNESS;
            addresses.extend(Self::do_lookup(seed.seed, default_port, socks5, services));
        }

        Ok(addresses)
    }

    /// Builds the weighted distribution used for a regular outgoing connection.
    ///
    /// The base distribution samples network groups before addresses. IPv4 /16s and IPv6 /32s
    /// therefore do not gain probability merely by contributing more addresses. Address health
    /// favors peers that have not been attempted recently and heavily penalizes recent failures.
    /// Connected and in-flight peers reduce their network group's probability exponentially.
    fn build_selection_weights(
        &self,
        required_service: ServiceFlags,
        now: u64,
    ) -> Vec<(usize, f64)> {
        let mut candidates = Vec::with_capacity(self.addresses.len());
        let mut groups = HashMap::<AddressGroup, AddressGroupStats>::new();

        for (&id, address) in &self.addresses {
            let group = address.address_group();
            let stats = groups.entry(group).or_default();

            if address.state == AddressState::Connected || self.attempting.contains(&id) {
                stats.represented += 1;
                continue;
            }

            let Some(health) = address.selection_health(now) else {
                continue;
            };

            stats.total_health += health;
            stats.maximum_health = stats.maximum_health.max(health);
            candidates.push(SelectionCandidate {
                id,
                group,
                health,
                services: address.services,
            });
        }

        let mut total_group_weight = 0.0;
        for stats in groups.values_mut().filter(|stats| stats.total_health > 0.0) {
            let represented = i32::try_from(stats.represented).unwrap_or(i32::MAX);
            stats.selection_weight = NETWORK_GROUP_PENALTY.powi(represented) * stats.maximum_health;
            total_group_weight += stats.selection_weight;
        }

        if total_group_weight <= 0.0 || !total_group_weight.is_finite() {
            return Vec::new();
        }

        let mut probabilities = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            let stats = groups
                .get(&candidate.group)
                .expect("candidate network group must have statistics");
            let group_probability = stats.selection_weight / total_group_weight;
            probabilities.push(group_probability * candidate.health / stats.total_health);
        }

        if required_service != ServiceFlags::NONE {
            let service_mass: f64 = candidates
                .iter()
                .zip(&probabilities)
                .filter(|(candidate, _)| candidate.services.has(required_service))
                .map(|(_, probability)| probability)
                .sum();

            if service_mass > 0.0 {
                for (candidate, probability) in candidates.iter().zip(&mut probabilities) {
                    if candidate.services.has(required_service) {
                        *probability /= service_mass;
                    } else {
                        *probability = 0.0;
                    }
                }
            } else {
                debug!(
                    "No eligible address advertises required services {required_service}; using base peer distribution"
                );
            }
        } else {
            let utreexo = service_flags::UTREEXO.into();
            let utreexo_mass: f64 = candidates
                .iter()
                .zip(&probabilities)
                .filter(|(candidate, _)| candidate.services.has(utreexo))
                .map(|(_, probability)| probability)
                .sum();
            let compact_filters_mass: f64 = candidates
                .iter()
                .zip(&probabilities)
                .filter(|(candidate, _)| candidate.services.has(ServiceFlags::COMPACT_FILTERS))
                .map(|(_, probability)| probability)
                .sum();

            let mut base_weight = MIXED_PEERS_WEIGHT;
            if utreexo_mass == 0.0 {
                base_weight += UTREEXO_WEIGHT;
                debug!(
                    "No eligible Utreexo address; assigning its reserved probability to the base peer distribution"
                );
            }
            if compact_filters_mass == 0.0 {
                base_weight += COMPACT_FILTERS_WEIGHT;
                debug!(
                    "No eligible compact-filter address; assigning its reserved probability to the base peer distribution"
                );
            }

            for (candidate, probability) in candidates.iter().zip(&mut probabilities) {
                let base_probability = *probability;
                *probability = base_weight * base_probability;

                if utreexo_mass > 0.0 && candidate.services.has(utreexo) {
                    *probability += UTREEXO_WEIGHT * base_probability / utreexo_mass;
                }
                if compact_filters_mass > 0.0
                    && candidate.services.has(ServiceFlags::COMPACT_FILTERS)
                {
                    *probability +=
                        COMPACT_FILTERS_WEIGHT * base_probability / compact_filters_mass;
                }
            }
        }

        candidates
            .into_iter()
            .zip(probabilities)
            .map(|(candidate, probability)| (candidate.id, probability))
            .collect()
    }

    /// Returns a weighted address for a new connection and reserves it as in-flight.
    ///
    /// Calling [`AddressMan::update_set_state`] completes the attempt and releases the
    /// reservation. If no address advertises `required_service`, selection falls back to the
    /// unconditioned distribution because service advertisements may be stale.
    pub fn get_address_to_connect(
        &mut self,
        required_service: ServiceFlags,
        feeler: bool,
    ) -> Option<(usize, LocalAddress)> {
        let selected_id = if feeler {
            self.addresses
                .iter()
                .filter(|(id, address)| {
                    !self.attempting.contains(id)
                        && !matches!(
                            address.state,
                            AddressState::Banned(_) | AddressState::Connected
                        )
                })
                .map(|(&id, _)| id)
                .choose(&mut rand::rng())?
        } else {
            let weighted_addresses =
                self.build_selection_weights(required_service, Self::time_since_unix());
            let distribution =
                WeightedIndex::new(weighted_addresses.iter().map(|(_, weight)| *weight)).ok()?;
            weighted_addresses
                .get(distribution.sample(&mut rand::rng()))?
                .0
        };

        let address = self.addresses.get(&selected_id)?.clone();
        self.attempting.insert(selected_id);
        Some((selected_id, address))
    }

    /// Releases a reservation when a connection could not be started.
    pub(crate) fn cancel_connection_attempt(&mut self, id: usize) {
        self.attempting.remove(&id);
    }

    pub fn dump_peers(&self, datadir: impl AsRef<Path>) -> std::io::Result<()> {
        let datadir = datadir.as_ref();

        let peers: Vec<_> = self
            .addresses
            .values()
            .cloned()
            .map(Into::<DiskLocalAddress>::into)
            .collect::<Vec<_>>();
        let peers = serde_json::to_string(&peers);
        if let Ok(peers) = peers {
            std::fs::write(datadir.join("peers.json"), peers)?;
        }
        Ok(())
    }

    /// Dumps the connected utreexo peers to a file on dir `datadir/anchors.json` in json format `
    /// inputs are the directory to save the file and the list of ids of the connected utreexo peers
    pub fn dump_utreexo_peers(
        &self,
        datadir: impl AsRef<Path>,
        peers_id: &[usize],
    ) -> std::io::Result<()> {
        let datadir = datadir.as_ref();

        let addresses: Vec<DiskLocalAddress> = peers_id
            .iter()
            .filter_map(|id| Some(self.addresses.get(id)?.to_owned().into()))
            .collect();
        let addresses: Result<String, serde_json::Error> = serde_json::to_string(&addresses);
        if let Ok(addresses) = addresses {
            std::fs::write(datadir.join("anchors.json"), addresses)?;
        }
        Ok(())
    }

    pub fn start_addr_man(&mut self, datadir: impl AsRef<Path>) -> Vec<LocalAddress> {
        let datadir = datadir.as_ref();

        let persisted_peers = read_to_string(datadir.join("peers.json"))
            .map(|seeds| serde_json::from_str::<Vec<DiskLocalAddress>>(&seeds));

        if let Ok(Ok(peers)) = persisted_peers {
            let peers = peers
                .into_iter()
                .map(Into::<LocalAddress>::into)
                .collect::<Vec<_>>();

            self.push_addresses(&peers);
        }

        let anchors = read_to_string(datadir.join("anchors.json")).and_then(|anchors| {
            let anchors = serde_json::from_str::<Vec<DiskLocalAddress>>(&anchors)?;
            Ok(anchors
                .into_iter()
                .map(Into::<LocalAddress>::into)
                .collect::<Vec<_>>())
        });

        if anchors.is_err() {
            warn!("Failed to init Utreexo peers: anchors.json does not exist yet, or is invalid");
        }

        anchors.unwrap_or_default()
    }

    /// This function moves addresses between buckets, like if the ban time of a peer expired,
    /// or if we tried to connect to a peer and it failed in the past, but now it might be online
    /// again.
    pub fn rearrange_buckets(&mut self) {
        let now = Self::time_since_unix();

        for address in self.addresses.values_mut() {
            match address.state {
                AddressState::Banned(ban_time) => {
                    if ban_time < now {
                        address.state = AddressState::NeverTried;
                    }
                }
                AddressState::Tried(tried_time) => {
                    if tried_time + ASSUME_STALE < now {
                        address.state = AddressState::NeverTried;
                    }
                }
                AddressState::Failed(failed_time) => {
                    if failed_time + ASSUME_STALE < now {
                        address.state = AddressState::NeverTried;
                    }
                }
                AddressState::Connected | AddressState::NeverTried => {}
            }
        }
    }

    /// Updates the state of an address
    pub fn update_set_state(&mut self, idx: usize, state: AddressState) -> &mut Self {
        self.attempting.remove(&idx);
        if let Some(address) = self.addresses.get_mut(&idx) {
            address.state = state;
        }

        match state {
            AddressState::Banned(_) => {
                self.good_addresses.retain(|&x| x != idx);
            }
            AddressState::Tried(_) => {
                if !self.good_addresses.contains(&idx) {
                    self.good_addresses.push(idx);
                }

                if let Some(address) = self.addresses.get(&idx).cloned() {
                    self.push_if_has_service(&address, service_flags::UTREEXO.into());
                    self.push_if_has_service(&address, service_flags::UTREEXO_ARCHIVE.into());
                    self.push_if_has_service(&address, ServiceFlags::NONE); // this means any peer
                    self.push_if_has_service(&address, ServiceFlags::COMPACT_FILTERS);
                }
            }
            AddressState::NeverTried => {
                self.good_addresses.retain(|&x| x != idx);
            }
            AddressState::Connected => {
                self.addresses.entry(idx).and_modify(|addr| {
                    addr.last_connected = Self::time_since_unix();
                });

                if !self.good_addresses.contains(&idx) {
                    self.good_addresses.push(idx);
                }

                // push to the good peers by service
                if let Some(address) = self.addresses.get(&idx).cloned() {
                    self.push_if_has_service(&address, service_flags::UTREEXO.into());
                    self.push_if_has_service(&address, ServiceFlags::NONE); // this means any peer
                    self.push_if_has_service(&address, ServiceFlags::COMPACT_FILTERS);
                }
            }
            AddressState::Failed(_) => {
                self.good_addresses.retain(|&x| x != idx);
                for peers in self.good_peers_by_service.values_mut() {
                    peers.retain(|&x| x != idx);
                }
            }
        }

        self
    }

    /// Adds a peer to the list of peers known to have some service
    fn add_peer_to_service(&mut self, idx: usize, service: ServiceFlags) {
        if let Some(peers) = self.peers_by_service.get_mut(&service) {
            if peers.contains(&idx) {
                return;
            }

            peers.push(idx);
        } else {
            self.peers_by_service.insert(service, vec![idx]);
        }
    }

    /// Removes a peer from the list of peers known to have some service
    fn remove_peer_from_service(&mut self, idx: usize, service: ServiceFlags) {
        if let Some(peers) = self.peers_by_service.get_mut(&service) {
            peers.retain(|&x| x != idx);
        }
    }

    /// Updates the list of peers that have a service
    ///
    /// If a peer used to advertise a service, but now it doesn't, we remove it from the list
    /// of peers that have that service. If a peer didn't advertise a service, but now it does,
    /// we add it to the list of peers that have that service.
    fn update_peer_for_service(&mut self, id: usize, service: ServiceFlags) {
        let Some(peer) = self.addresses.get(&id) else {
            return;
        };

        match peer.services.has(service) {
            true => self.add_peer_to_service(id, service),
            false => self.remove_peer_from_service(id, service),
        }
    }

    /// Updates `peers_by_service` buckets with the latest service flags info about a peer
    ///
    /// This function is called when we receive a version message from a peer, telling which
    /// services it advertises.
    ///
    /// We only index for Compact Filters and Utreexo. For NODE_NETWORK and NODE_WITNESS we already
    /// filter them out when we add them to the address manager, therefore, all peers in this list
    /// is already known for having those. And we don't care about the rest of the services,
    /// like NODE_BLOOM.
    fn update_peer_services_buckets(&mut self, idx: usize) {
        self.update_peer_for_service(idx, service_flags::UTREEXO.into());
        self.update_peer_for_service(idx, ServiceFlags::COMPACT_FILTERS);
    }

    /// Updates the service flags after we receive a version message
    pub fn update_set_service_flag(&mut self, idx: usize, flags: ServiceFlags) -> &mut Self {
        // if this peer turns out to not have the minimum required services, we remove it
        if !flags.has(ServiceFlags::NETWORK_LIMITED) || !flags.has(ServiceFlags::WITNESS) {
            self.addresses.remove(&idx);
            self.attempting.remove(&idx);
            for peers in self.peers_by_service.values_mut() {
                peers.retain(|&x| x != idx);
            }

            self.good_addresses.retain(|&x| x != idx);
            self.good_peers_by_service
                .values_mut()
                .for_each(|peers| peers.retain(|&x| x != idx));

            return self;
        }

        if let Some(address) = self.addresses.get_mut(&idx) {
            address.services = flags;
        }

        self.update_peer_services_buckets(idx);
        self
    }

    /// Returns the file path to the seeds file for the given network
    const fn get_net_seeds(network: Network) -> &'static str {
        match network {
            Network::Bitcoin => include_str!("../../seeds/mainnet_seeds.json"),
            Network::Signet => include_str!("../../seeds/signet_seeds.json"),
            Network::Testnet => include_str!("../../seeds/testnet_seeds.json"),
            Network::Testnet4 => include_str!("../../seeds/testnet4_seeds.json"),
            Network::Regtest => include_str!("../../seeds/regtest_seeds.json"),
        }
    }

    /// Reads the hard-coded addresses from the seeds file and adds them to the address manager
    ///
    /// This is a last-resort method to try to connect to a peer, if we don't have any other
    /// addresses to connect to.
    pub(crate) fn add_fixed_addresses(&mut self, network: Network) {
        let addresses = Self::get_net_seeds(network);
        let peers: Vec<DiskLocalAddress> =
            serde_json::from_str(addresses).expect("BUG: fixed peers are invalid");

        let peers = peers
            .into_iter()
            .map(Into::<LocalAddress>::into)
            .collect::<Vec<_>>();

        self.push_addresses(&peers);
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
        let address = match *value.address.as_addrv2() {
            AddrV2::Ipv4(ip) => Address::V4(ip),
            AddrV2::Ipv6(ip) => Address::V6(ip),
            AddrV2::Cjdns(ip) => Address::Cjdns(ip),
            AddrV2::I2p(ip) => Address::I2p(ip),
            AddrV2::TorV2(ip) => Address::OnionV2(ip),
            AddrV2::TorV3(ip) => Address::OnionV3(ip),
            AddrV2::Unknown(_, _) => Address::V4(Ipv4Addr::LOCALHOST),
        };

        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            address,
            last_connected: value.last_connected,
            state: if value.state == AddressState::Connected {
                AddressState::Tried(time)
            } else {
                value.state
            },
            services: value.services.to_u64(),
            port: value.get_port(),
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
        let sock_addr = BitcoinSocketAddr::new(address, value.port);

        Self {
            address: sock_addr,
            last_connected: value.last_connected,
            state: value.state,
            services,
            id: value
                .id
                .unwrap_or_else(|| rand::rng().random_range(0..usize::MAX)),
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

impl From<Address> for AddrV2 {
    fn from(value: Address) -> Self {
        match value {
            Address::V4(addr) => Self::Ipv4(addr),
            Address::V6(addr) => Self::Ipv6(addr),
            Address::I2p(addr) => Self::I2p(addr),
            Address::Cjdns(addr) => Self::Cjdns(addr),
            Address::OnionV2(addr) => Self::TorV2(addr),
            Address::OnionV3(addr) => Self::TorV3(addr),
        }
    }
}

/// Simple implementation of a DNS-over-HTTPS (DoH) lookup routed through the SOCKS5 proxy
pub mod dns_proxy {
    use core::net::IpAddr;
    use core::net::SocketAddr;
    use std::sync::Arc;
    use std::time::Duration;

    use rustls::crypto;
    use serde::Deserialize;
    use ureq::Agent;
    use ureq::Proxy;
    use ureq::tls::TlsConfig;
    use ureq::tls::TlsProvider;

    #[derive(Deserialize)]
    /// JSON format from [Google's DoH API](https://developers.google.com/speed/public-dns/docs/doh/json#dns_response_in_json)
    struct DnsResponse {
        /// We only care about the "Answer" array
        #[serde(rename = "Answer")]
        answers: Option<Vec<AnswerEntry>>,
    }

    #[derive(Deserialize)]
    struct AnswerEntry {
        /// The IP address as a string
        data: String,

        /// Record type; 1=A, 28=AAAA
        #[serde(rename = "type")]
        record_type: u8,
    }

    /// Lookup `host` by DNS-over-HTTPS (DoH) through a SOCKS5 proxy. Returns both A (IPv4)
    /// and AAAA (IPv6) records. Only Google sees the actual DNS query but doesn't learn our IP.
    pub fn lookup_host_via_proxy(
        host: &str,
        proxy_addr: SocketAddr,
    ) -> Result<Vec<IpAddr>, ureq::Error> {
        // Note: ureq does not implement "socks5h://", so this will resolve "dns.google" locally,
        // but the Bitcoin DNS query remains encrypted. Only Google can see the query contents.
        let proxy = Proxy::new(&format!("socks5://{proxy_addr}"))?;

        let crypto = Arc::new(crypto::ring::default_provider());
        let tls_config = TlsConfig::builder()
            .provider(TlsProvider::Rustls)
            .unversioned_rustls_crypto_provider(crypto)
            .build();

        let agent: Agent = Agent::config_builder()
            .tls_config(tls_config)
            .timeout_global(Some(Duration::from_secs(30)))
            .proxy(Some(proxy))
            .build()
            .into();

        // We will perform two queries in sequence: type=1 (A) and type=28 (AAAA).
        let mut all_ips = Vec::new();
        for record_type in [1u8, 28u8] {
            let mut ips = query(&agent, host, record_type)?;
            all_ips.append(&mut ips);
        }

        Ok(all_ips)
    }

    // Helper function that performs a single DoH query for the given record_type.
    fn query(agent: &Agent, host: &str, record_type: u8) -> Result<Vec<IpAddr>, ureq::Error> {
        // Construct the DoH URL for the JSON API:
        // https://developers.google.com/speed/public-dns/docs/secure-transports
        let url = format!("https://dns.google/resolve?name={host}&type={record_type}");

        // Send a GET over HTTPS. The proxy will only see Google's address and the TLS handshake.
        let mut response = agent.get(&url).call()?;
        let dns_response: DnsResponse = response.body_mut().read_json()?;

        let answers = dns_response.answers.unwrap_or_default();

        // Filter by record_type (sanity) and parse each "data" field into an IpAddr.
        let mut result = Vec::new();
        for entry in answers.into_iter().filter(|e| e.record_type == record_type) {
            if let Ok(ip) = entry.data.parse() {
                result.push(ip);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use std::fs::File;
    use std::io::Read;
    use std::io::{self};

    use bitcoin::Network;
    use bitcoin::p2p::ServiceFlags;
    use bitcoin::p2p::address::AddrV2;
    use floresta_chain::get_chain_dns_seeds;
    use floresta_common::assert_ok;
    use floresta_common::service_flags;
    use rand::RngExt;

    use super::AddressState;
    use super::LocalAddress;
    use super::*;
    use crate::address_man::AddressMan;
    use crate::address_man::DiskLocalAddress;
    use crate::address_man::ReachableNetworks;
    use crate::bitcoin_socket_addr::BitcoinSocketAddr;

    fn load_addresses_from_json(file_path: impl AsRef<Path>) -> io::Result<Vec<LocalAddress>> {
        let mut contents = String::new();
        File::open(file_path)?.read_to_string(&mut contents)?;

        let seeds: Vec<DiskLocalAddress> =
            serde_json::from_str(&contents).expect("JSON not well-formatted");
        let mut addresses = Vec::new();
        let mut rng = rand::rng();

        for seed in seeds {
            let state = match seed.state {
                AddressState::Tried(time) => AddressState::Tried(time),
                _ => continue,
            };

            let local_address = LocalAddress {
                address: BitcoinSocketAddr::new(seed.address.into(), seed.port),
                last_connected: seed.last_connected,
                state,
                services: ServiceFlags::from(seed.services),
                id: rng.random::<u64>() as usize,
            };
            addresses.push(local_address);
        }

        Ok(addresses)
    }

    fn selection_test_address(
        ip: &str,
        id: usize,
        state: AddressState,
        services: ServiceFlags,
    ) -> LocalAddress {
        LocalAddress {
            address: format!("{ip}:8333").parse().unwrap(),
            last_connected: 0,
            state,
            services,
            id,
        }
    }

    fn assert_probability(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected probability {expected}, got {actual}"
        );
    }

    #[test]
    fn test_dns_seed_addresses_are_marked_tried() {
        let now = 1_000_000;
        let services = ServiceFlags::NETWORK_LIMITED
            | ServiceFlags::WITNESS
            | ServiceFlags::COMPACT_FILTERS
            | service_flags::UTREEXO.into();

        for ip in ["8.8.8.8", "2001:4860:4860::8888"] {
            let address = AddressMan::dns_seed_address(ip.parse().unwrap(), 8333, services, now);
            assert_eq!(address.state, AddressState::Tried(now));
            assert_eq!(address.services, services);
        }
    }

    #[test]
    fn test_selection_health_penalizes_and_recovers_failed_addresses() {
        let now = 1_000_000;
        let recent_failure =
            selection_test_address("11.1.1.1", 1, AddressState::Failed(now), ServiceFlags::NONE);
        let old_failure = selection_test_address(
            "12.1.1.1",
            2,
            AddressState::Failed(now - FAILURE_TAU as u64),
            ServiceFlags::NONE,
        );
        let never_tried =
            selection_test_address("13.1.1.1", 3, AddressState::NeverTried, ServiceFlags::NONE);

        let recent_health = recent_failure.selection_health(now).unwrap();
        let old_health = old_failure.selection_health(now).unwrap();
        let never_tried_health = never_tried.selection_health(now).unwrap();

        assert_probability(recent_health, RETRY_FLOOR * FAILURE_FLOOR);
        assert!(recent_health < old_health);
        assert!(old_health < never_tried_health);
        assert_probability(never_tried_health, 1.0);
    }

    #[test]
    fn test_selection_prefers_tried_addresses() {
        let now = 1_000_000;
        let mut address_man = AddressMan::new(None, &[]);
        address_man.addresses.insert(
            1,
            selection_test_address(
                "11.1.0.1",
                1,
                AddressState::Tried(now - 10 * RETRY_TAU as u64),
                ServiceFlags::NONE,
            ),
        );
        address_man.addresses.insert(
            2,
            selection_test_address("12.1.0.1", 2, AddressState::NeverTried, ServiceFlags::NONE),
        );

        let weights = address_man.build_selection_weights(ServiceFlags::NONE, now);
        let tried_probability = weights
            .iter()
            .find(|(id, _)| *id == 1)
            .expect("tried address is eligible")
            .1;
        let never_tried_probability = weights
            .iter()
            .find(|(id, _)| *id == 2)
            .expect("never-tried address is eligible")
            .1;

        assert!(tried_probability > 0.9);
        assert!(never_tried_probability < 0.1);
    }

    #[test]
    fn test_selection_equalizes_subnets_and_penalizes_connected_subnets() {
        let mut address_man = AddressMan::new(None, &[]);

        for id in 0..20 {
            let address = selection_test_address(
                &format!("11.1.0.{}", id + 1),
                id,
                AddressState::NeverTried,
                ServiceFlags::NONE,
            );
            address_man.addresses.insert(id, address);
        }
        address_man.addresses.insert(
            20,
            selection_test_address("12.1.0.1", 20, AddressState::NeverTried, ServiceFlags::NONE),
        );

        let crowded_group = address_man.addresses[&0].address_group();
        let probability_for_crowded_group = |address_man: &AddressMan, weights: &[(usize, f64)]| {
            weights
                .iter()
                .filter(|(id, _)| address_man.addresses[id].address_group() == crowded_group)
                .map(|(_, probability)| probability)
                .sum::<f64>()
        };

        let weights = address_man.build_selection_weights(ServiceFlags::NONE, 1_000_000);
        assert_probability(probability_for_crowded_group(&address_man, &weights), 0.5);

        address_man.addresses.get_mut(&0).unwrap().state = AddressState::Connected;
        let weights = address_man.build_selection_weights(ServiceFlags::NONE, 1_000_000);
        assert_probability(
            probability_for_crowded_group(&address_man, &weights),
            NETWORK_GROUP_PENALTY / (1.0 + NETWORK_GROUP_PENALTY),
        );

        address_man.addresses.get_mut(&0).unwrap().state = AddressState::NeverTried;
        address_man.attempting.insert(0);
        let weights = address_man.build_selection_weights(ServiceFlags::NONE, 1_000_000);
        assert_probability(
            probability_for_crowded_group(&address_man, &weights),
            NETWORK_GROUP_PENALTY / (1.0 + NETWORK_GROUP_PENALTY),
        );
    }

    #[test]
    fn test_selection_reserves_twenty_percent_for_each_service() {
        let mut address_man = AddressMan::new(None, &[]);
        let utreexo: ServiceFlags = service_flags::UTREEXO.into();

        for id in 0..100 {
            let services = match id {
                98 => ServiceFlags::COMPACT_FILTERS,
                99 => utreexo,
                _ => ServiceFlags::NONE,
            };
            let address = selection_test_address(
                &format!("20.{}.0.1", id + 1),
                id,
                AddressState::NeverTried,
                services,
            );
            address_man.addresses.insert(id, address);
        }

        let weights = address_man.build_selection_weights(ServiceFlags::NONE, 1_000_000);
        let total_probability: f64 = weights.iter().map(|(_, probability)| probability).sum();
        let utreexo_probability: f64 = weights
            .iter()
            .filter(|(id, _)| address_man.addresses[id].services.has(utreexo))
            .map(|(_, probability)| probability)
            .sum();
        let compact_filters_probability: f64 = weights
            .iter()
            .filter(|(id, _)| {
                address_man.addresses[id]
                    .services
                    .has(ServiceFlags::COMPACT_FILTERS)
            })
            .map(|(_, probability)| probability)
            .sum();

        assert_probability(total_probability, 1.0);
        assert_probability(
            utreexo_probability,
            UTREEXO_WEIGHT + MIXED_PEERS_WEIGHT / 100.0,
        );
        assert_probability(
            compact_filters_probability,
            COMPACT_FILTERS_WEIGHT + MIXED_PEERS_WEIGHT / 100.0,
        );

        let required_weights = address_man.build_selection_weights(utreexo, 1_000_000);
        assert_probability(
            required_weights.iter().find(|(id, _)| *id == 99).unwrap().1,
            1.0,
        );
        assert!(
            required_weights
                .iter()
                .filter(|(id, _)| *id != 99)
                .all(|(_, probability)| *probability == 0.0)
        );
    }

    #[test]
    fn test_selection_service_weights_are_additive() {
        let mut address_man = AddressMan::new(None, &[]);
        let utreexo: ServiceFlags = service_flags::UTREEXO.into();
        let compact_filters = ServiceFlags::COMPACT_FILTERS;
        let services = [
            ServiceFlags::NONE,
            compact_filters,
            utreexo,
            compact_filters | utreexo,
        ];

        for (id, services) in services.into_iter().enumerate() {
            address_man.addresses.insert(
                id,
                selection_test_address(
                    &format!("30.{}.0.1", id + 1),
                    id,
                    AddressState::NeverTried,
                    services,
                ),
            );
        }

        let weights = address_man.build_selection_weights(ServiceFlags::NONE, 1_000_000);
        let probability = |id| {
            weights
                .iter()
                .find(|(candidate_id, _)| *candidate_id == id)
                .unwrap()
                .1
        };

        assert_probability(probability(0), 0.15);
        assert_probability(probability(1), 0.25);
        assert_probability(probability(2), 0.25);
        assert_probability(probability(3), 0.35);
    }

    #[test]
    fn test_selection_is_without_replacement_until_attempt_finishes() {
        let mut address_man = AddressMan::new(None, &[]);
        address_man.addresses.insert(
            1,
            selection_test_address("11.1.0.1", 1, AddressState::NeverTried, ServiceFlags::NONE),
        );
        address_man.addresses.insert(
            2,
            selection_test_address("12.1.0.1", 2, AddressState::NeverTried, ServiceFlags::NONE),
        );

        let first = address_man
            .get_address_to_connect(ServiceFlags::NONE, false)
            .unwrap()
            .0;
        let second = address_man
            .get_address_to_connect(ServiceFlags::NONE, false)
            .unwrap()
            .0;

        assert_ne!(first, second);
        assert!(
            address_man
                .get_address_to_connect(ServiceFlags::NONE, false)
                .is_none()
        );

        address_man.update_set_state(first, AddressState::Failed(AddressMan::time_since_unix()));
        assert_eq!(
            address_man
                .get_address_to_connect(ServiceFlags::NONE, false)
                .unwrap()
                .0,
            first
        );
    }

    #[test]
    fn test_selection_excludes_connected_addresses() {
        let mut address_man = AddressMan::new(None, &[]);
        let preferred_services =
            ServiceFlags::COMPACT_FILTERS | ServiceFlags::from(service_flags::UTREEXO);
        address_man.addresses.insert(
            1,
            selection_test_address("11.1.0.1", 1, AddressState::Connected, preferred_services),
        );
        address_man.addresses.insert(
            2,
            selection_test_address("12.1.0.1", 2, AddressState::NeverTried, ServiceFlags::NONE),
        );

        assert_eq!(
            address_man
                .get_address_to_connect(ServiceFlags::NONE, false)
                .unwrap()
                .0,
            2
        );
        address_man.update_set_state(2, AddressState::Connected);
        assert!(
            address_man
                .get_address_to_connect(ServiceFlags::NONE, false)
                .is_none()
        );
    }

    #[test]
    fn test_local_addr_from_str() {
        // v4
        let ips = [
            "127.146.182.45",
            "2.212.31.248",
            "6.108.160.10",
            "151.43.223.99",
            "216.20.167.190",
            "188.33.163.249",
            "227.237.60.84",
            "8.104.121.145",
            "100.119.250.124",
        ];

        for addr_str in ips {
            let local_address = format!("{addr_str}:8333")
                .parse::<LocalAddress>()
                .unwrap_or_else(|_| panic!("failed to parse {addr_str}"));

            assert_eq!(
                *local_address.address.as_addrv2(),
                AddrV2::Ipv4(addr_str.parse().unwrap())
            );

            assert_eq!(local_address.get_port(), 8333);
        }

        // v6
        let ips = [
            "67db:3727:f145:5c59:718f:d3b9:6e56:d937",
            "7813:70c7:ea5d:f78a:7920:33d8:1da0:f9d7",
            "4a08:75e4:893f:d5a1:e2e2:3c99:8d20:22cf",
            "8da0:6b59:1494:bc7f:b217:51eb:c5fb:29c6",
            "cb1a:5104:57a9:0616:f6e0:191f:9224:4f35",
            "259a:ddc7:44a2:b5ec:f1ff:6024:50e8:928d",
            "46eb:cab1:bd48:c461:1775:c64e:c11b:3e77",
            "142b:a452:dff7:a41c:6cc6:317e:cc94:bb10",
            "0f8d:6d08:de58:017a:cd92:c868:023a:86e6",
            "8a80:5cfd:ccac:3e63:a243:d89f:d5e1:8e4c",
        ];

        for addr_str in ips {
            let local_address = format!("[{addr_str}]:8333")
                .parse::<LocalAddress>()
                .unwrap_or_else(|_| panic!("failed to parse {addr_str}"));

            assert_eq!(
                *local_address.address.as_addrv2(),
                AddrV2::Ipv6(addr_str.parse().unwrap())
            );

            assert_eq!(local_address.get_port(), 8333);
        }
    }

    #[test]
    fn test_adding_fixed_peer() {
        let signet_addresses = load_addresses_from_json("./seeds/signet_seeds.json").unwrap();

        let mut addr_man =
            AddressMan::new(None, &[ReachableNetworks::IPv4, ReachableNetworks::IPv6]);
        addr_man.add_fixed_addresses(Network::Signet);

        assert_eq!(addr_man.good_addresses.len(), signet_addresses.len());

        let utreexo_addresses = signet_addresses
            .iter()
            .filter(|address| address.services.has(service_flags::UTREEXO.into()))
            .collect::<Vec<_>>();

        assert_eq!(
            addr_man
                .good_peers_by_service
                .get(&service_flags::UTREEXO.into())
                .unwrap()
                .len(),
            utreexo_addresses.len()
        );

        assert_eq!(
            addr_man
                .peers_by_service
                .get(&service_flags::UTREEXO.into())
                .unwrap()
                .len(),
            utreexo_addresses.len()
        );
    }

    #[test]
    fn test_parse() {
        let signet_address = load_addresses_from_json("./seeds/signet_seeds.json").unwrap();

        assert!(!signet_address.is_empty());
        let random = rand::rng().random_range(1..=13);
        let loc_adr_1 = LocalAddress::from(signet_address[random].address.clone());
        assert_eq!(loc_adr_1.address, signet_address[random].address);
    }

    #[test]
    fn test_fixed_peers() {
        let _ = load_addresses_from_json("./seeds/signet_seeds.json").unwrap();
        let _ = load_addresses_from_json("./seeds/mainnet_seeds.json").unwrap();
        let _ = load_addresses_from_json("./seeds/testnet_seeds.json").unwrap();
        let _ = load_addresses_from_json("./seeds/regtest_seeds.json").unwrap();
    }

    #[test]
    fn test_address_man() {
        let mut address_man =
            AddressMan::new(None, &[ReachableNetworks::IPv4, ReachableNetworks::IPv6]);

        let signet_address = load_addresses_from_json("./seeds/signet_seeds.json").unwrap();

        address_man.push_addresses(&signet_address);

        assert!(!address_man.good_addresses.is_empty());

        assert!(!address_man.peers_by_service.is_empty());

        assert!(!address_man.get_addresses_to_send().is_empty());

        assert!(
            address_man
                .get_address_to_connect(ServiceFlags::default(), true)
                .is_some()
        );

        assert!(
            address_man
                .get_address_to_connect(ServiceFlags::default(), false)
                .is_some()
        );

        assert!(
            address_man
                .get_address_to_connect(ServiceFlags::NONE, false)
                .is_some()
        );

        assert!(
            address_man
                .get_address_to_connect(service_flags::UTREEXO.into(), false)
                .is_some()
        );

        assert!(!AddressMan::get_net_seeds(Network::Signet).is_empty());
        assert!(!AddressMan::get_net_seeds(Network::Bitcoin).is_empty());
        assert!(!AddressMan::get_net_seeds(Network::Regtest).is_empty());
        assert!(!AddressMan::get_net_seeds(Network::Testnet).is_empty());

        assert_ok!(AddressMan::get_seeds_from_dns(
            &get_chain_dns_seeds(Network::Signet)[0],
            8333,
            None, // No proxy
        ));

        address_man.rearrange_buckets();
    }

    #[test]
    fn test_is_routable() {
        // random addresses that are private
        let addresses = vec![
            "10.42.187.23:8333",
            "10.0.254.199:8333",
            "172.16.88.4:8333",
            "172.31.201.77:8333",
            "192.168.1.14:8333",
            "192.168.203.250:8333",
            "0.9.85.249:8333",
            "[fd3a:9f2b:4c10:1a2b::1]:8333",
            "[fd12:3456:789a:1::dead]:8333",
            "[fdff:ab23:9012:beef::42]:8333",
            "[fd7c:2e91:aa10:ff01:1234:5678:9abc:def0]:8333",
            "[fd00:1111:2222:3333:4444:5555:6666:7777]:8333",
        ]
        .into_iter()
        .map(|s| {
            s.parse::<LocalAddress>()
                .unwrap_or_else(|_| panic!("Failed to parse address: {s}"))
        })
        .collect::<Vec<_>>();

        for address in addresses {
            assert!(!address.is_routable(), "{address:?}");
        }

        // now load the signet seeds and ensure none are private
        let signet_address = load_addresses_from_json("./seeds/signet_seeds.json").unwrap();

        for address in signet_address {
            assert!(address.is_routable(), "{address:?}");
        }
    }

    fn get_addresses_and_random_times() -> Vec<LocalAddress> {
        let signet_address = load_addresses_from_json("./seeds/signet_seeds.json").unwrap();

        // modify some addresses to have failed connections in the past
        let now = AddressMan::time_since_unix();

        let mut modified_addresses = signet_address.clone();
        let addresses = modified_addresses.len();
        for (i, item) in modified_addresses.iter_mut().enumerate().take(addresses) {
            if i % 3 == 0 {
                item.last_connected = now - 5000;
            } else if i % 3 == 1 {
                item.last_connected = now - 6000;
            } else {
                item.last_connected = now - 2000;
            }
        }

        modified_addresses
    }

    #[test]
    fn test_rearrange_buckets() {
        let mut address_man = AddressMan::new(None, &[]);
        let addresses = get_addresses_and_random_times();
        address_man.addresses.extend(
            addresses
                .iter()
                .map(|addr| (addr.id, addr.clone()))
                .collect::<std::collections::HashMap<usize, LocalAddress>>(),
        );

        assert_eq!(address_man.addresses.len(), addresses.len());
        address_man.rearrange_buckets();

        assert!(address_man.addresses.iter().all(|(_, addr)| {
            matches!(
                addr.state,
                AddressState::NeverTried | AddressState::Tried(_)
            )
        }));
    }

    #[test]
    fn test_is_net_reachable() {
        let v4 = "127.146.182.45";
        let v6 = "142b:a452:dff7:a41c:6cc6:317e:cc94:bb10";

        let addr_v4 = AddrV2::Ipv4(v4.parse().unwrap());
        let addr_v6 = AddrV2::Ipv6(v6.parse().unwrap());
        let addr_onionv3 = AddrV2::TorV3([
            0x89, 0x6c, 0x6a, 0x71, 0x70, 0x6b, 0x67, 0x61, 0x62, 0x67, 0x34, 0x68, 0x72, 0x63,
            0x68, 0x62, 0x6f, 0x7a, 0x77, 0x6f, 0x76, 0x66, 0x66, 0x79, 0x6b, 0x37, 0x66, 0x6f,
            0x62, 0x70, 0x6f, 0x76,
        ]);
        let addr_i2p = AddrV2::I2p([
            0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e,
            0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x66, 0x6f,
            0x62, 0x70, 0x6f, 0x76,
        ]);

        let address_man =
            AddressMan::new(None, &[ReachableNetworks::IPv4, ReachableNetworks::IPv6]);

        assert!(address_man.is_net_reachable(&LocalAddress {
            address: BitcoinSocketAddr::new(addr_v4, 8333),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::default(),
            id: 0,
        }));

        assert!(address_man.is_net_reachable(&LocalAddress {
            address: BitcoinSocketAddr::new(addr_v6, 8333),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::default(),
            id: 0,
        }));

        assert!(!address_man.is_net_reachable(&LocalAddress {
            address: BitcoinSocketAddr::new(addr_onionv3, 8333),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::default(),
            id: 0,
        }));

        assert!(!address_man.is_net_reachable(&LocalAddress {
            address: BitcoinSocketAddr::new(addr_i2p, 8333),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::default(),
            id: 0,
        }));
    }

    #[test]
    fn test_push_address() {
        let mut address_man = AddressMan::new(None, &[ReachableNetworks::IPv4]);
        let v4_no_witness = LocalAddress {
            address: "12.146.182.45:8333".parse().unwrap(),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::NETWORK | ServiceFlags::NETWORK_LIMITED,
            id: 0,
        };

        let v4_with_witness = LocalAddress {
            address: "12.146.182.45:8333".parse().unwrap(),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::NETWORK | ServiceFlags::NETWORK_LIMITED | ServiceFlags::WITNESS,
            id: 1,
        };

        let v6_with_witness = LocalAddress {
            address: "[fd3a:9f2b:4c10:1a2b::1]:8333".parse().unwrap(),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::NETWORK_LIMITED | ServiceFlags::NETWORK | ServiceFlags::WITNESS,
            id: 2,
        };

        let v4_not_routable = LocalAddress {
            address: "127.0.0.1:8333".parse().unwrap(),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::NETWORK_LIMITED | ServiceFlags::NETWORK | ServiceFlags::WITNESS,
            id: 3,
        };

        let v6_not_routable = LocalAddress {
            address: "[::1]:8333".parse().unwrap(),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::NETWORK_LIMITED | ServiceFlags::NETWORK | ServiceFlags::WITNESS,
            id: 4,
        };

        let onion = LocalAddress {
            address: BitcoinSocketAddr::new(
                AddrV2::TorV3([
                    0x89, 0x6c, 0x6a, 0x71, 0x70, 0x6b, 0x67, 0x61, 0x62, 0x67, 0x34, 0x68, 0x72,
                    0x63, 0x68, 0x62, 0x6f, 0x7a, 0x77, 0x6f, 0x76, 0x66, 0x66, 0x79, 0x6b, 0x37,
                    0x66, 0x6f, 0x62, 0x70, 0x6f, 0x76,
                ]),
                8333,
            ),
            last_connected: 0,
            state: AddressState::NeverTried,
            services: ServiceFlags::NETWORK_LIMITED | ServiceFlags::NETWORK | ServiceFlags::WITNESS,
            id: 5,
        };

        let addresses = vec![
            v4_no_witness,
            v4_with_witness.clone(),
            v6_with_witness,
            v4_not_routable,
            v6_not_routable,
            onion,
        ];

        address_man.push_addresses(&addresses);

        // only the v4 with witness
        assert_eq!(address_man.addresses.len(), 1);
        assert_eq!(
            *address_man.addresses.values().next().unwrap(),
            v4_with_witness
        );
    }

    #[test]
    fn test_prune_addresses() {
        let mut address_man = AddressMan::new(Some(10), &[]);
        let addresses = get_addresses_and_random_times();
        address_man.addresses.extend(
            addresses
                .iter()
                .map(|addr| (addr.id, addr.clone()))
                .collect::<std::collections::HashMap<usize, LocalAddress>>(),
        );

        assert_eq!(address_man.addresses.len(), addresses.len(),);

        address_man.prune_addresses();

        assert_ne!(address_man.addresses.len(), addresses.len());
    }

    #[test]
    fn test_update_address_state() {
        let mut address_man = AddressMan::new(None, &[]);
        let addresses = get_addresses_and_random_times();
        address_man.addresses.extend(
            addresses
                .iter()
                .map(|addr| (addr.id, addr.clone()))
                .collect::<std::collections::HashMap<usize, LocalAddress>>(),
        );

        for addr in addresses {
            address_man.update_set_state(addr.id, AddressState::Banned(0));
        }

        assert!(
            address_man
                .addresses
                .values()
                .all(|addr| matches!(addr.state, AddressState::Banned(_)))
        );
    }

    #[test]
    fn test_update_service_flags() {
        let mut address_man = AddressMan::new(None, &[]);
        let addresses = get_addresses_and_random_times();

        address_man.addresses.extend(
            addresses
                .iter()
                .map(|addr| (addr.id, addr.clone()))
                .collect::<std::collections::HashMap<usize, LocalAddress>>(),
        );

        for addr in addresses {
            address_man.update_set_service_flag(addr.id, service_flags::UTREEXO.into());
        }

        assert!(
            address_man
                .addresses
                .values()
                .all(|addr| addr.services.has(service_flags::UTREEXO.into()))
        );
    }

    #[test]
    fn test_add_fixed_addresses() {
        let mut address_man = AddressMan::new(None, &ReachableNetworks::SUPPORTED);
        address_man.add_fixed_addresses(Network::Signet);
        assert!(!address_man.addresses.is_empty());
    }
}

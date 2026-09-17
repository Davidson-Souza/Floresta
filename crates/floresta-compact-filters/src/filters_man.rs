// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact block-filter synchronization and rescan service.
//!
//! [`FiltersMan`] persists filter headers, validates every downloaded filter against that chain,
//! caches recent filters, builds filters for newly connected local blocks, and serves paginated
//! rescans through cloneable [`FilterManHandle`] values.

use core::fmt;
use core::fmt::Display;
use core::fmt::Formatter;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::time::Duration;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::FilterHeader;
use bitcoin::OutPoint;
use bitcoin::ScriptBuf;
use bitcoin::bip158::BlockFilter;
use bitcoin::hashes::Hash;
use bitcoin::p2p::message_filter::CFHeaders;
use floresta_chain::BlockConsumer;
use floresta_chain::BlockchainInterface;
use floresta_chain::UtxoData;
pub use floresta_common::ChainMethods;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::MissedTickBehavior;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::FilterHeadersStore;
use crate::FlatFilterStoreError;

const BASIC_FILTER_TYPE: u8 = 0;
const FILTER_HEADER_BATCH_SIZE: u32 = 1_000;
const FILTER_BATCH_SIZE: u32 = 1_000;
const FILTER_REQUEST_SIZE: usize = 100;
const CHECKPOINT_INTERVAL: u32 = 1_000;
const CONNECTED_BLOCK_BUFFER: usize = 16;
const DEFAULT_RESCAN_PAGE_SIZE: usize = 50;
const MAX_RESCAN_PAGE_SIZE: usize = 10_000;
const SYNC_INTERVAL: Duration = Duration::from_secs(30);
const RESCAN_RETRY_INTERVAL: Duration = Duration::from_secs(1);

/// Minimal blockchain view required by [`FiltersMan`].
pub trait FilterChain: Clone + Send + Sync + 'static {
    /// Error returned by the chain backend.
    type Error: Error + Send + Sync + 'static;

    /// Returns the current best-chain height.
    fn get_height(&self) -> Result<u32, Self::Error>;

    /// Returns the best-chain block hash at `height`.
    fn get_block_hash(&self, height: u32) -> Result<BlockHash, Self::Error>;
}

impl<T> FilterChain for T
where
    T: BlockchainInterface + Clone + Send + Sync + 'static,
    T::Error: Error + Send + Sync + 'static,
{
    type Error = T::Error;

    fn get_height(&self) -> Result<u32, Self::Error> {
        BlockchainInterface::get_height(self)
    }

    fn get_block_hash(&self, height: u32) -> Result<BlockHash, Self::Error> {
        BlockchainInterface::get_block_hash(self, height)
    }
}

/// Parameters for a compact-filter rescan.
#[derive(Debug, Clone)]
pub struct RescanRequest {
    /// Scripts to match against each filter.
    pub scripts: Vec<ScriptBuf>,

    /// First height to scan. Genesis is used when omitted.
    pub start_height: Option<u32>,

    /// Last height to scan, inclusively. The current tip is used when omitted.
    pub end_height: Option<u32>,

    /// Maximum number of matched blocks buffered until the caller consumes a page.
    pub max_blocks_per_page: Option<usize>,
}

impl RescanRequest {
    /// Creates a full-chain rescan for `scripts`.
    pub fn new(scripts: Vec<ScriptBuf>) -> Self {
        Self {
            scripts,
            start_height: None,
            end_height: None,
            max_blocks_per_page: None,
        }
    }

    /// Restricts this rescan to an inclusive height range.
    pub fn with_range(mut self, start_height: Option<u32>, end_height: Option<u32>) -> Self {
        self.start_height = start_height;
        self.end_height = end_height;
        self
    }

    /// Sets the maximum number of matched blocks buffered for this rescan.
    pub fn with_max_blocks_per_page(mut self, max_blocks_per_page: usize) -> Self {
        self.max_blocks_per_page = Some(max_blocks_per_page);
        self
    }
}

/// Current state of an asynchronous rescan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RescanStatus {
    /// The rescan was accepted but has not produced a result yet.
    Started,

    /// At least one matched block is ready to consume.
    Available,

    /// Filters are still being scanned and no matched block is currently ready.
    Waiting,

    /// Every matched block has been consumed.
    Finished,
}

impl Display for RescanStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Started => write!(f, "rescan started"),
            Self::Available => write!(f, "rescan blocks are available"),
            Self::Waiting => write!(f, "rescan is still scanning filters"),
            Self::Finished => write!(f, "rescan finished"),
        }
    }
}

/// Stable identifier for an in-flight rescan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RescanTicket(u64);

/// Errors returned by the compact-filter service or its handles.
#[derive(Debug)]
pub enum FilterManError {
    /// The chain backend rejected a lookup.
    Chain(Box<dyn Error + Send + Sync>),

    /// The node interface failed while fetching network data.
    Node(Box<dyn Error + Send + Sync>),
    /// A concurrent filter request task failed.
    Task(tokio::task::JoinError),

    /// Persistent filter-header storage failed.
    Store(FlatFilterStoreError),

    /// BIP158 filter construction or matching failed.
    Bip158(bitcoin::bip158::Error),

    /// A peer returned filter headers or checkpoints that violate BIP157.
    InvalidHeaders(String),

    /// A downloaded filter does not commit to the stored header.
    InvalidFilter(u32),
    /// A peer returned a different number of filters than requested.
    InvalidFilterCount {
        /// Number of requested filters.
        expected: usize,

        /// Number of returned filters.
        received: usize,
    },

    /// A rescan request contained no scripts.
    EmptyRescan,

    /// A rescan range is outside the current best chain or is reversed.
    InvalidRescanRange {
        /// Requested first height.
        start: u32,

        /// Requested last height.
        end: u32,

        /// Best-chain height when the request was accepted.
        tip: u32,
    },

    /// A rescan page size was zero or exceeded the service limit.
    InvalidPageSize(usize),

    /// The requested rescan ticket does not exist.
    RescanNotFound(RescanTicket),

    /// A background rescan failed.
    RescanFailed(String),

    /// The manager task has stopped.
    ManagerStopped,

    /// Internal shared state was poisoned.
    PoisonedLock,
}

impl FilterManError {
    fn chain<E>(error: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::Chain(Box::new(error))
    }

    fn node<E>(error: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::Node(Box::new(error))
    }
}

impl Display for FilterManError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chain(error) => write!(f, "chain error: {error}"),
            Self::Node(error) => write!(f, "node error: {error}"),
            Self::Task(error) => write!(f, "filter request task failed: {error}"),
            Self::Store(error) => write!(f, "filter store error: {error}"),
            Self::Bip158(error) => write!(f, "BIP158 error: {error}"),
            Self::InvalidHeaders(reason) => write!(f, "invalid filter headers: {reason}"),
            Self::InvalidFilter(height) => {
                write!(f, "filter at height {height} does not match its header")
            }
            Self::InvalidFilterCount { expected, received } => write!(
                f,
                "peer returned {received} compact filters; expected {expected}"
            ),
            Self::EmptyRescan => write!(f, "a rescan requires at least one script"),
            Self::InvalidRescanRange { start, end, tip } => write!(
                f,
                "invalid rescan range {start}..={end}; current tip is {tip}"
            ),
            Self::InvalidPageSize(size) => write!(
                f,
                "invalid rescan page size {size}; expected 1..={MAX_RESCAN_PAGE_SIZE}"
            ),
            Self::RescanNotFound(ticket) => write!(f, "rescan ticket {ticket:?} was not found"),
            Self::RescanFailed(error) => write!(f, "rescan failed: {error}"),
            Self::ManagerStopped => write!(f, "compact-filter manager stopped"),
            Self::PoisonedLock => write!(f, "compact-filter store lock was poisoned"),
        }
    }
}

impl Error for FilterManError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Chain(error) | Self::Node(error) => Some(error.as_ref()),
            Self::Store(error) => Some(error),
            Self::Bip158(error) => Some(error),
            Self::Task(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FlatFilterStoreError> for FilterManError {
    fn from(error: FlatFilterStoreError) -> Self {
        Self::Store(error)
    }
}

impl From<bitcoin::bip158::Error> for FilterManError {
    fn from(error: bitcoin::bip158::Error) -> Self {
        Self::Bip158(error)
    }
}

impl<Store> From<PoisonError<MutexGuard<'_, Store>>> for FilterManError {
    fn from(_: PoisonError<MutexGuard<'_, Store>>) -> Self {
        Self::PoisonedLock
    }
}

#[derive(Debug)]
struct ConnectedBlock {
    block: Block,
    height: u32,
    spent_utxos: HashMap<OutPoint, UtxoData>,
}

#[derive(Clone)]
struct FilterBlockConsumer {
    sender: mpsc::Sender<ConnectedBlock>,
}

impl BlockConsumer for FilterBlockConsumer {
    fn wants_spent_utxos(&self) -> bool {
        true
    }

    fn on_block(
        &self,
        block: &Block,
        height: u32,
        spent_utxos: Option<&HashMap<OutPoint, UtxoData>>,
    ) {
        let Some(spent_utxos) = spent_utxos else {
            return;
        };
        let Ok(permit) = self.sender.try_reserve() else {
            return;
        };

        permit.send(ConnectedBlock {
            block: block.clone(),
            height,
            spent_utxos: spent_utxos.clone(),
        });
    }
}

enum ManagerRequest {
    Rescan {
        request: RescanRequest,
        response: oneshot::Sender<Result<RescanTicket, FilterManError>>,
    },
    RescanStatus {
        ticket: RescanTicket,
        response: oneshot::Sender<Result<RescanStatus, FilterManError>>,
    },
    RescanBlocks {
        ticket: RescanTicket,
        response: oneshot::Sender<Result<Vec<Block>, FilterManError>>,
    },
    Filter {
        height: u32,
        response: oneshot::Sender<Result<BlockFilter, FilterManError>>,
    },
}

struct RescanState {
    blocks: mpsc::Receiver<Block>,
    failure: oneshot::Receiver<String>,
    failure_message: Option<String>,
    page_size: usize,
    first_status: bool,
}

/// Cloneable interface to a running [`FiltersMan`].
#[derive(Clone)]
pub struct FilterManHandle {
    sender: mpsc::Sender<ManagerRequest>,
}

impl FilterManHandle {
    /// Starts a rescan and returns its ticket without waiting for completion.
    pub async fn rescan(&self, request: RescanRequest) -> Result<RescanTicket, FilterManError> {
        let (response, receiver) = oneshot::channel();
        self.send(ManagerRequest::Rescan { request, response }, receiver)
            .await
    }

    /// Returns the current state of `ticket`.
    pub async fn get_info(&self, ticket: RescanTicket) -> Result<RescanStatus, FilterManError> {
        let (response, receiver) = oneshot::channel();
        self.send(ManagerRequest::RescanStatus { ticket, response }, receiver)
            .await
    }

    /// Consumes at most one page of currently available matched blocks.
    pub async fn get_blocks(&self, ticket: RescanTicket) -> Result<Vec<Block>, FilterManError> {
        let (response, receiver) = oneshot::channel();
        self.send(ManagerRequest::RescanBlocks { ticket, response }, receiver)
            .await
    }

    /// Returns a validated filter, fetching and caching it when necessary.
    pub async fn get_filter(&self, height: u32) -> Result<BlockFilter, FilterManError> {
        let (response, receiver) = oneshot::channel();
        self.send(ManagerRequest::Filter { height, response }, receiver)
            .await
    }

    async fn send<T>(
        &self,
        request: ManagerRequest,
        receiver: oneshot::Receiver<Result<T, FilterManError>>,
    ) -> Result<T, FilterManError> {
        self.sender
            .send(request)
            .await
            .map_err(|_| FilterManError::ManagerStopped)?;
        receiver.await.map_err(|_| FilterManError::ManagerStopped)?
    }
}

/// Compact-filter synchronization and rescan service.
pub struct FiltersMan<Store, Chain, Node> {
    store: Arc<Mutex<Store>>,
    node: Node,
    chain: Chain,
    requests: mpsc::Receiver<ManagerRequest>,
    request_sender: mpsc::Sender<ManagerRequest>,
    connected_blocks: mpsc::Receiver<ConnectedBlock>,
    block_sender: mpsc::Sender<ConnectedBlock>,
    rescans: HashMap<RescanTicket, RescanState>,
    next_ticket: u64,
    checkpoints: Vec<FilterHeader>,
}

impl<Store, Chain, Node> FiltersMan<Store, Chain, Node>
where
    Store: FilterHeadersStore,
    Chain: FilterChain,
    Node: ChainMethods + Clone + Send + Sync + 'static,
    Node::Error: Error + Send + Sync + 'static,
{
    /// Creates a compact-filter manager.
    pub fn new(store: Store, node: Node, chain: Chain) -> Self {
        let (request_sender, requests) = mpsc::channel(128);
        let (block_sender, connected_blocks) = mpsc::channel(CONNECTED_BLOCK_BUFFER);

        Self {
            store: Arc::new(Mutex::new(store)),
            node,
            chain,
            requests,
            request_sender,
            connected_blocks,
            block_sender,
            rescans: HashMap::new(),
            next_ticket: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Returns a cloneable service handle.
    pub fn get_handle(&self) -> FilterManHandle {
        FilterManHandle {
            sender: self.request_sender.clone(),
        }
    }

    /// Returns a chain subscriber that builds filters for newly connected blocks.
    pub fn block_consumer(&self) -> Arc<dyn BlockConsumer> {
        Arc::new(FilterBlockConsumer {
            sender: self.block_sender.clone(),
        })
    }

    /// Runs synchronization and serves requests until every producer is dropped.
    pub async fn main_loop(mut self) -> Result<(), FilterManError> {
        if let Err(error) = self.sync().await {
            warn!(%error, "initial compact-filter synchronization failed; will retry");
        }

        let mut sync_interval = tokio::time::interval(SYNC_INTERVAL);
        sync_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        sync_interval.tick().await;

        loop {
            tokio::select! {
                request = self.requests.recv() => {
                    let Some(request) = request else {
                        return Ok(());
                    };
                    self.handle_request(request).await;
                }
                connected = self.connected_blocks.recv() => {
                    if let Some(connected) = connected {
                        if let Err(error) = self.process_connected_block(connected).await {
                            warn!(%error, "failed to build compact filter for connected block");
                        }
                    }
                }
                _ = sync_interval.tick() => {
                    if let Err(error) = self.sync().await {
                        warn!(%error, "periodic compact-filter synchronization failed");
                    }
                }
            }
        }
    }

    async fn handle_request(&mut self, request: ManagerRequest) {
        match request {
            ManagerRequest::Rescan { request, response } => {
                let _ = response.send(self.start_rescan(request));
            }
            ManagerRequest::RescanStatus { ticket, response } => {
                let _ = response.send(self.rescan_status(ticket));
            }
            ManagerRequest::RescanBlocks { ticket, response } => {
                let _ = response.send(self.rescan_blocks(ticket));
            }
            ManagerRequest::Filter { height, response } => {
                let result = Self::fetch_filter(
                    self.store.clone(),
                    self.node.clone(),
                    self.chain.clone(),
                    height,
                )
                .await;
                let _ = response.send(result);
            }
        }
    }

    fn start_rescan(&mut self, request: RescanRequest) -> Result<RescanTicket, FilterManError> {
        if request.scripts.is_empty() {
            return Err(FilterManError::EmptyRescan);
        }

        let tip = self.chain.get_height().map_err(FilterManError::chain)?;
        let start = request.start_height.unwrap_or(0);
        let end = request.end_height.unwrap_or(tip);
        if start > end || end > tip {
            return Err(FilterManError::InvalidRescanRange { start, end, tip });
        }

        let page_size = request
            .max_blocks_per_page
            .unwrap_or(DEFAULT_RESCAN_PAGE_SIZE);
        if page_size == 0 || page_size > MAX_RESCAN_PAGE_SIZE {
            return Err(FilterManError::InvalidPageSize(page_size));
        }

        let ticket = RescanTicket(self.next_ticket);
        self.next_ticket = self.next_ticket.wrapping_add(1);
        let (sender, blocks) = mpsc::channel(page_size);
        let (failure_sender, failure) = oneshot::channel();
        self.rescans.insert(
            ticket,
            RescanState {
                blocks,
                failure,
                failure_message: None,
                page_size,
                first_status: true,
            },
        );

        let store = self.store.clone();
        let node = self.node.clone();
        let chain = self.chain.clone();
        tokio::spawn(async move {
            if let Err(error) =
                Self::run_rescan(store, node, chain, request, start, end, &sender).await
            {
                let _ = failure_sender.send(error.to_string());
            }
        });

        Ok(ticket)
    }

    fn rescan_status(&mut self, ticket: RescanTicket) -> Result<RescanStatus, FilterManError> {
        let state = self
            .rescans
            .get_mut(&ticket)
            .ok_or(FilterManError::RescanNotFound(ticket))?;

        if let Some(error) = Self::rescan_failure(state) {
            return Err(FilterManError::RescanFailed(error));
        }
        if !state.blocks.is_empty() {
            return Ok(RescanStatus::Available);
        }
        if state.blocks.is_closed() {
            return Ok(RescanStatus::Finished);
        }
        if state.first_status {
            state.first_status = false;
            return Ok(RescanStatus::Started);
        }

        Ok(RescanStatus::Waiting)
    }

    fn rescan_blocks(&mut self, ticket: RescanTicket) -> Result<Vec<Block>, FilterManError> {
        let state = self
            .rescans
            .get_mut(&ticket)
            .ok_or(FilterManError::RescanNotFound(ticket))?;
        let mut blocks = Vec::with_capacity(state.page_size);

        if let Some(error) = Self::rescan_failure(state) {
            return Err(FilterManError::RescanFailed(error));
        }

        for _ in 0..state.page_size {
            match state.blocks.try_recv() {
                Ok(block) => blocks.push(block),
                Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                    break;
                }
            }
        }

        Ok(blocks)
    }

    fn rescan_failure(state: &mut RescanState) -> Option<String> {
        if state.failure_message.is_none() {
            if let Ok(error) = state.failure.try_recv() {
                state.failure_message = Some(error);
            }
        }

        state.failure_message.clone()
    }

    async fn run_rescan(
        store: Arc<Mutex<Store>>,
        node: Node,
        chain: Chain,
        request: RescanRequest,
        start: u32,
        end: u32,
        blocks: &mpsc::Sender<Block>,
    ) -> Result<(), FilterManError> {
        let mut batch_start = start;
        let mut matching_blocks = tracing::enabled!(tracing::Level::DEBUG).then(Vec::new);
        loop {
            let batch_end = batch_start.saturating_add(FILTER_BATCH_SIZE - 1).min(end);
            let filters = loop {
                if blocks.is_closed() {
                    return Err(FilterManError::ManagerStopped);
                }

                match Self::fetch_filters(
                    store.clone(),
                    node.clone(),
                    chain.clone(),
                    batch_start,
                    batch_end,
                )
                .await
                {
                    Ok(filters) => break filters,
                    Err(FilterManError::Node(error)) => {
                        warn!(
                            %error,
                            start = batch_start,
                            end = batch_end,
                            "compact-filter rescan batch failed; retrying"
                        );
                        tokio::time::sleep(RESCAN_RETRY_INTERVAL).await;
                    }
                    Err(error) => return Err(error),
                }
            };

            for (block_hash, filter) in filters {
                let matches = filter.match_any(
                    &block_hash,
                    request.scripts.iter().map(|script| script.as_bytes()),
                )?;
                if !matches {
                    continue;
                }

                let block = loop {
                    match node.get_block(block_hash).await {
                        Ok(Some(block)) => break block,
                        Ok(None) => {
                            warn!(%block_hash, "rescan block unavailable; retrying");
                        }
                        Err(error) => {
                            warn!(%error, %block_hash, "rescan block request failed; retrying");
                        }
                    }
                    tokio::time::sleep(RESCAN_RETRY_INTERVAL).await;
                };
                blocks
                    .send(block)
                    .await
                    .map_err(|_| FilterManError::ManagerStopped)?;
                if let Some(matching_blocks) = &mut matching_blocks {
                    matching_blocks.push(block_hash);
                }
            }

            if batch_end == end {
                if let Some(matching_blocks) = matching_blocks {
                    debug!(
                        start,
                        end,
                        ?matching_blocks,
                        "compact-filter rescan finished"
                    );
                }
                return Ok(());
            }
            batch_start = batch_end + 1;
        }
    }

    async fn fetch_filter(
        store: Arc<Mutex<Store>>,
        node: Node,
        chain: Chain,
        height: u32,
    ) -> Result<BlockFilter, FilterManError> {
        let mut filters = Self::fetch_filters(store, node, chain, height, height).await?;
        Ok(filters
            .pop()
            .expect("a one-height filter range returns one filter")
            .1)
    }

    async fn fetch_filters(
        store: Arc<Mutex<Store>>,
        node: Node,
        chain: Chain,
        start: u32,
        end: u32,
    ) -> Result<Vec<(BlockHash, BlockFilter)>, FilterManError> {
        let heights = start..=end;
        let mut block_hashes = Vec::with_capacity(heights.size_hint().0);
        for height in heights.clone() {
            block_hashes.push(
                chain
                    .get_block_hash(height)
                    .map_err(FilterManError::chain)?,
            );
        }

        let mut filters = {
            let store = store.lock()?;
            heights
                .clone()
                .map(|height| store.get_filter(height))
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut requests = tokio::task::JoinSet::new();
        for first in (0..filters.len()).step_by(FILTER_REQUEST_SIZE) {
            let last = (first + FILTER_REQUEST_SIZE).min(filters.len());
            if filters[first..last].iter().all(Option::is_some) {
                continue;
            }

            let node = node.clone();
            let requested_hashes = block_hashes[first..last].to_vec();
            let request_height = start + first as u32;
            requests.spawn(async move {
                let expected = requested_hashes.len();
                let filters = node.get_cfilter(request_height, requested_hashes).await?;
                Ok::<_, Node::Error>((first, expected, filters))
            });
        }

        while let Some(response) = requests.join_next().await {
            let (first, expected, received_filters) = response
                .map_err(FilterManError::Task)?
                .map_err(FilterManError::node)?;
            if received_filters.len() != expected {
                return Err(FilterManError::InvalidFilterCount {
                    expected,
                    received: received_filters.len(),
                });
            }

            let mut store = store.lock()?;
            for (offset, filter) in received_filters.into_iter().enumerate() {
                let index = first + offset;
                let height = start + index as u32;
                Self::validate_filter(&mut *store, height, &filter)?;
                store.put_filter(height, filter.clone())?;
                filters[index] = Some(filter);
            }
        }

        Ok(block_hashes
            .into_iter()
            .zip(filters)
            .map(|(block_hash, filter)| {
                (
                    block_hash,
                    filter.expect("every requested compact filter is available"),
                )
            })
            .collect())
    }

    fn validate_filter(
        store: &mut Store,
        height: u32,
        filter: &BlockFilter,
    ) -> Result<(), FilterManError> {
        let previous_header = if height == 0 {
            FilterHeader::all_zeros()
        } else {
            store.get_filter_header(height - 1)?
        };
        let expected_header = store.get_filter_header(height)?;

        if filter.filter_header(&previous_header) != expected_header {
            return Err(FilterManError::InvalidFilter(height));
        }

        Ok(())
    }

    async fn process_connected_block(
        &mut self,
        connected: ConnectedBlock,
    ) -> Result<(), FilterManError> {
        let filter = BlockFilter::new_script_filter(&connected.block, |outpoint| {
            connected
                .spent_utxos
                .get(outpoint)
                .map(|utxo| utxo.txout.script_pubkey.as_script())
                .ok_or(bitcoin::bip158::Error::UtxoMissing(*outpoint))
        })?;

        let height = connected.height;
        let needs_sync = match self.store.lock()?.get_height()? {
            Some(stored_height) => stored_height.saturating_add(1) < height,
            None => height > 0,
        };
        if needs_sync {
            self.sync().await?;
        }

        let block_hash = connected.block.block_hash();
        let mut store = self.store.lock()?;
        let previous_header = if height == 0 {
            FilterHeader::all_zeros()
        } else {
            store.get_filter_header(height - 1)?
        };
        let filter_header = filter.filter_header(&previous_header);

        let expected_checkpoint = if height > 0 && height % CHECKPOINT_INTERVAL == 0 {
            self.checkpoints
                .get((height / CHECKPOINT_INTERVAL - 1) as usize)
                .copied()
        } else {
            None
        };
        match store.get_height()? {
            Some(stored_height) if stored_height >= height => {
                let stored_hash = store.get_block_hash(height)?;
                let stored_header = store.get_filter_header(height)?;
                if stored_hash == block_hash {
                    if stored_header != filter_header {
                        return Err(FilterManError::InvalidFilter(height));
                    }
                } else {
                    store.truncate(height.checked_sub(1))?;
                    store.put_filter_header(block_hash, filter_header)?;
                    self.checkpoints
                        .truncate((height.saturating_sub(1) / CHECKPOINT_INTERVAL) as usize);
                }
            }
            Some(stored_height) if stored_height.saturating_add(1) == height => {
                if expected_checkpoint.is_some_and(|expected| expected != filter_header) {
                    return Err(FilterManError::InvalidHeaders(format!(
                        "locally built filter disagrees with checkpoint at height {height}"
                    )));
                }
                store.put_filter_header(block_hash, filter_header)?;
            }
            None if height == 0 => {
                store.put_filter_header(block_hash, filter_header)?;
            }
            _ => {
                return Err(FilterManError::InvalidHeaders(format!(
                    "cannot connect locally built filter at height {height}"
                )));
            }
        }

        store.put_filter(height, filter)?;
        store.flush()?;
        Ok(())
    }

    /// Synchronizes filter headers and BIP157 checkpoints to the current best-chain tip.
    pub async fn sync(&mut self) -> Result<(), FilterManError> {
        self.reconcile_store()?;
        let tip = self.chain.get_height().map_err(FilterManError::chain)?;
        self.sync_checkpoints(tip).await?;
        self.verify_stored_checkpoints()?;

        loop {
            let Some((start, stop, stop_hash)) = self.next_header_range()? else {
                break;
            };
            debug!(start, stop, %stop_hash, "requesting compact filter headers");
            let headers = self
                .node
                .get_cfilters_headers(start, stop_hash)
                .await
                .map_err(FilterManError::node)?;
            self.apply_headers(start, stop, stop_hash, headers)?;
        }

        let height = self.store.lock()?.get_height()?;
        info!(?height, "compact filter headers synchronized");
        Ok(())
    }

    fn reconcile_store(&mut self) -> Result<(), FilterManError> {
        let tip = self.chain.get_height().map_err(FilterManError::chain)?;
        let Some(stored_tip) = self.store.lock()?.get_height()? else {
            return Ok(());
        };
        let mut height = stored_tip.min(tip);

        loop {
            let stored_hash = self.store.lock()?.get_block_hash(height)?;
            let chain_hash = self
                .chain
                .get_block_hash(height)
                .map_err(FilterManError::chain)?;
            if stored_hash == chain_hash {
                if height != stored_tip {
                    self.store.lock()?.truncate(Some(height))?;
                    self.checkpoints
                        .truncate((height / CHECKPOINT_INTERVAL) as usize);
                }
                return Ok(());
            }

            if height == 0 {
                self.store.lock()?.truncate(None)?;
                self.checkpoints.clear();
                return Ok(());
            }
            height -= 1;
        }
    }

    async fn sync_checkpoints(&mut self, tip: u32) -> Result<(), FilterManError> {
        let expected_count = (tip / CHECKPOINT_INTERVAL) as usize;
        if expected_count == 0 {
            self.checkpoints.clear();
            return Ok(());
        }
        if self.checkpoints.len() == expected_count {
            return Ok(());
        }

        let stop_hash = self
            .chain
            .get_block_hash(tip)
            .map_err(FilterManError::chain)?;
        let response = self
            .node
            .get_cfcheckpt(stop_hash)
            .await
            .map_err(FilterManError::node)?;

        if response.filter_type != BASIC_FILTER_TYPE {
            return Err(FilterManError::InvalidHeaders(format!(
                "unexpected checkpoint filter type {}",
                response.filter_type
            )));
        }
        if response.stop_hash != stop_hash {
            return Err(FilterManError::InvalidHeaders(format!(
                "checkpoint stop hash {} does not match requested {stop_hash}",
                response.stop_hash
            )));
        }
        if response.filter_headers.len() != expected_count {
            return Err(FilterManError::InvalidHeaders(format!(
                "received {} checkpoints, expected {expected_count}",
                response.filter_headers.len()
            )));
        }

        self.checkpoints = response.filter_headers;
        Ok(())
    }

    fn verify_stored_checkpoints(&mut self) -> Result<(), FilterManError> {
        let Some(stored_tip) = self.store.lock()?.get_height()? else {
            return Ok(());
        };

        for (index, checkpoint) in self.checkpoints.iter().enumerate() {
            let height = u32::try_from(index + 1)
                .ok()
                .and_then(|index| index.checked_mul(CHECKPOINT_INTERVAL))
                .ok_or_else(|| {
                    FilterManError::InvalidHeaders("checkpoint height overflow".to_owned())
                })?;
            if height > stored_tip {
                break;
            }
            if self.store.lock()?.get_filter_header(height)? != *checkpoint {
                return Err(FilterManError::InvalidHeaders(format!(
                    "stored filter header disagrees with checkpoint at height {height}"
                )));
            }
        }

        Ok(())
    }

    fn next_header_range(&self) -> Result<Option<(u32, u32, BlockHash)>, FilterManError> {
        let tip = self.chain.get_height().map_err(FilterManError::chain)?;
        let start = self
            .store
            .lock()?
            .get_height()?
            .map_or(0, |height| height.saturating_add(1));
        if start > tip {
            return Ok(None);
        }

        let stop = start.saturating_add(FILTER_HEADER_BATCH_SIZE - 1).min(tip);
        let stop_hash = self
            .chain
            .get_block_hash(stop)
            .map_err(FilterManError::chain)?;
        Ok(Some((start, stop, stop_hash)))
    }

    fn apply_headers(
        &mut self,
        start: u32,
        stop: u32,
        stop_hash: BlockHash,
        response: CFHeaders,
    ) -> Result<(), FilterManError> {
        if response.filter_type != BASIC_FILTER_TYPE {
            return Err(FilterManError::InvalidHeaders(format!(
                "unexpected filter type {}",
                response.filter_type
            )));
        }
        if response.stop_hash != stop_hash {
            return Err(FilterManError::InvalidHeaders(format!(
                "response stop hash {} does not match requested {stop_hash}",
                response.stop_hash
            )));
        }

        let expected_count = usize::try_from(stop - start + 1).map_err(|_| {
            FilterManError::InvalidHeaders("filter-header count overflow".to_owned())
        })?;
        if response.filter_hashes.len() != expected_count {
            return Err(FilterManError::InvalidHeaders(format!(
                "received {} filter hashes, expected {expected_count}",
                response.filter_hashes.len()
            )));
        }

        let previous_header = if start == 0 {
            FilterHeader::all_zeros()
        } else {
            self.store.lock()?.get_filter_header(start - 1)?
        };
        if response.previous_filter_header != previous_header {
            return Err(FilterManError::InvalidHeaders(format!(
                "filter headers do not connect at height {start}"
            )));
        }

        let mut entries = Vec::with_capacity(expected_count);
        let mut current_header = previous_header;
        for (offset, filter_hash) in response.filter_hashes.into_iter().enumerate() {
            let height = start
                + u32::try_from(offset).map_err(|_| {
                    FilterManError::InvalidHeaders("filter-header height overflow".to_owned())
                })?;
            current_header = filter_hash.filter_header(&current_header);

            if height > 0 && height % CHECKPOINT_INTERVAL == 0 {
                let checkpoint_index = (height / CHECKPOINT_INTERVAL - 1) as usize;
                if self
                    .checkpoints
                    .get(checkpoint_index)
                    .is_some_and(|checkpoint| *checkpoint != current_header)
                {
                    return Err(FilterManError::InvalidHeaders(format!(
                        "filter header disagrees with checkpoint at height {height}"
                    )));
                }
            }

            let block_hash = self
                .chain
                .get_block_hash(height)
                .map_err(FilterManError::chain)?;
            entries.push((block_hash, current_header));
        }

        let mut store = self.store.lock()?;
        for (block_hash, filter_header) in entries {
            store.put_filter_header(block_hash, filter_header)?;
        }
        store.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use bitcoin::Network;
    use bitcoin::blockdata::constants::genesis_block;
    use bitcoin::p2p::message_filter::CFCheckpt;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::FlatFilterStore;

    #[derive(Debug, Clone, Copy)]
    struct MockError(&'static str);

    impl Display for MockError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for MockError {}

    #[derive(Clone)]
    struct MockChain {
        hashes: Arc<Vec<BlockHash>>,
    }

    impl FilterChain for MockChain {
        type Error = MockError;

        fn get_height(&self) -> Result<u32, Self::Error> {
            self.hashes
                .len()
                .checked_sub(1)
                .and_then(|height| u32::try_from(height).ok())
                .ok_or(MockError("empty chain"))
        }

        fn get_block_hash(&self, height: u32) -> Result<BlockHash, Self::Error> {
            self.hashes
                .get(height as usize)
                .copied()
                .ok_or(MockError("unknown height"))
        }
    }

    fn mock_block_hash(height: u32) -> BlockHash {
        let mut bytes = [0_u8; 32];
        bytes[..4].copy_from_slice(&height.to_le_bytes());
        BlockHash::from_byte_array(bytes)
    }

    #[derive(Clone)]
    struct MockNode {
        blocks: Arc<HashMap<BlockHash, Block>>,
        filters: Arc<HashMap<BlockHash, BlockFilter>>,
        header_responses: Arc<HashMap<(u32, BlockHash), CFHeaders>>,
        checkpoint_responses: Arc<HashMap<BlockHash, CFCheckpt>>,
        block_requests: Arc<AtomicUsize>,
        filter_requests: Arc<AtomicUsize>,
        filter_failures: Arc<AtomicUsize>,
        active_filter_requests: Arc<AtomicUsize>,
        max_filter_requests: Arc<AtomicUsize>,
        header_requests: Arc<AtomicUsize>,
        checkpoint_requests: Arc<AtomicUsize>,
    }

    impl MockNode {
        fn new(
            blocks: HashMap<BlockHash, Block>,
            filters: HashMap<BlockHash, BlockFilter>,
        ) -> Self {
            Self {
                blocks: Arc::new(blocks),
                filters: Arc::new(filters),
                header_responses: Arc::new(HashMap::new()),
                checkpoint_responses: Arc::new(HashMap::new()),
                block_requests: Arc::new(AtomicUsize::new(0)),
                filter_requests: Arc::new(AtomicUsize::new(0)),
                header_requests: Arc::new(AtomicUsize::new(0)),
                filter_failures: Arc::new(AtomicUsize::new(0)),
                active_filter_requests: Arc::new(AtomicUsize::new(0)),
                max_filter_requests: Arc::new(AtomicUsize::new(0)),
                checkpoint_requests: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl ChainMethods for MockNode {
        type Error = MockError;

        async fn get_block(&self, block_hash: BlockHash) -> Result<Option<Block>, Self::Error> {
            self.block_requests.fetch_add(1, Ordering::Relaxed);
            Ok(self.blocks.get(&block_hash).cloned())
        }

        async fn get_cfilters_headers(
            &self,
            start_height: u32,
            stop_hash: BlockHash,
        ) -> Result<CFHeaders, Self::Error> {
            self.header_requests.fetch_add(1, Ordering::Relaxed);
            self.header_responses
                .get(&(start_height, stop_hash))
                .cloned()
                .ok_or(MockError("unexpected filter-header request"))
        }

        async fn get_cfilter(
            &self,
            _start_height: u32,
            block_hashes: Vec<BlockHash>,
        ) -> Result<Vec<BlockFilter>, Self::Error> {
            self.filter_requests.fetch_add(1, Ordering::Relaxed);
            let active = self.active_filter_requests.fetch_add(1, Ordering::Relaxed) + 1;
            self.max_filter_requests
                .fetch_max(active, Ordering::Relaxed);
            tokio::task::yield_now().await;

            let result = if self
                .filter_failures
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |failures| {
                    failures.checked_sub(1)
                })
                .is_ok()
            {
                Err(MockError("filter request failed"))
            } else {
                block_hashes
                    .into_iter()
                    .map(|block_hash| {
                        self.filters
                            .get(&block_hash)
                            .cloned()
                            .ok_or(MockError("unknown filter"))
                    })
                    .collect()
            };
            self.active_filter_requests.fetch_sub(1, Ordering::Relaxed);
            result
        }

        async fn get_cfcheckpt(&self, stop_hash: BlockHash) -> Result<CFCheckpt, Self::Error> {
            self.checkpoint_requests.fetch_add(1, Ordering::Relaxed);
            self.checkpoint_responses
                .get(&stop_hash)
                .cloned()
                .ok_or(MockError("unexpected filter-checkpoint request"))
        }
    }

    fn setup() -> (
        NamedTempFile,
        FlatFilterStore,
        MockChain,
        MockNode,
        Block,
        BlockFilter,
    ) {
        let file = NamedTempFile::new().unwrap();
        let mut store = FlatFilterStore::new(file.path()).unwrap();
        let block = genesis_block(Network::Regtest);
        let block_hash = block.block_hash();
        let filter = BlockFilter::new_script_filter(&block, |outpoint| {
            Err::<ScriptBuf, _>(bitcoin::bip158::Error::UtxoMissing(*outpoint))
        })
        .unwrap();
        store
            .put_filter_header(block_hash, filter.filter_header(&FilterHeader::all_zeros()))
            .unwrap();
        store.flush().unwrap();

        let chain = MockChain {
            hashes: Arc::new(vec![block_hash]),
        };
        let node = MockNode::new(
            HashMap::from([(block_hash, block.clone())]),
            HashMap::from([(block_hash, filter.clone())]),
        );
        (file, store, chain, node, block, filter)
    }

    #[tokio::test]
    async fn handle_fetches_validates_and_caches_filter() {
        let (_file, store, chain, node, _block, expected) = setup();
        let filter_requests = node.filter_requests.clone();
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());

        assert_eq!(handle.get_filter(0).await.unwrap(), expected);
        assert_eq!(handle.get_filter(0).await.unwrap(), expected);
        assert_eq!(filter_requests.load(Ordering::Relaxed), 1);

        task.abort();
    }

    #[tokio::test]
    async fn rescan_returns_matching_block_and_finishes() {
        let (_file, store, chain, node, block, _filter) = setup();
        let script = block.txdata[0].output[0].script_pubkey.clone();
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());
        let ticket = handle
            .rescan(RescanRequest::new(vec![script]).with_max_blocks_per_page(1))
            .await
            .unwrap();

        let matched = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let blocks = handle.get_blocks(ticket).await.unwrap();
                if !blocks.is_empty() {
                    break blocks;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(matched, vec![block]);

        let status = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let status = handle.get_info(ticket).await.unwrap();
                if status == RescanStatus::Finished {
                    break status;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(status, RescanStatus::Finished);

        task.abort();
    }

    #[tokio::test]
    async fn rescan_fetches_filters_in_batches() {
        let (_file, mut store, mut chain, mut node, block, first_filter) = setup();
        let mut second_block = block.clone();
        second_block.header.nonce = second_block.header.nonce.wrapping_add(1);
        let second_hash = second_block.block_hash();
        let second_filter = BlockFilter::new_script_filter(&second_block, |outpoint| {
            Err::<ScriptBuf, _>(bitcoin::bip158::Error::UtxoMissing(*outpoint))
        })
        .unwrap();
        let first_header = first_filter.filter_header(&FilterHeader::all_zeros());
        store
            .put_filter_header(second_hash, second_filter.filter_header(&first_header))
            .unwrap();
        store.flush().unwrap();
        Arc::make_mut(&mut chain.hashes).push(second_hash);
        Arc::make_mut(&mut node.blocks).insert(second_hash, second_block);
        Arc::make_mut(&mut node.filters).insert(second_hash, second_filter);
        let filter_requests = node.filter_requests.clone();

        let script = block.txdata[0].output[0].script_pubkey.clone();
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());
        let ticket = handle
            .rescan(RescanRequest::new(vec![script]))
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let _ = handle.get_blocks(ticket).await.unwrap();
                if handle.get_info(ticket).await.unwrap() == RescanStatus::Finished {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert_eq!(filter_requests.load(Ordering::Relaxed), 1);
        task.abort();
    }

    #[tokio::test]
    async fn fetches_filter_chunks_concurrently() {
        let file = NamedTempFile::new().unwrap();
        let mut store = FlatFilterStore::new(file.path()).unwrap();
        let count = FILTER_REQUEST_SIZE + 1;
        let mut hashes = Vec::with_capacity(count);
        let mut filters = HashMap::with_capacity(count);
        let mut previous_header = FilterHeader::all_zeros();

        for height in 0..count {
            let height = height as u32;
            let block_hash = mock_block_hash(height);
            let filter = BlockFilter::new(&height.to_le_bytes());
            previous_header = filter.filter_header(&previous_header);
            store
                .put_filter_header(block_hash, previous_header)
                .unwrap();
            hashes.push(block_hash);
            filters.insert(block_hash, filter);
        }
        store.flush().unwrap();

        let chain = MockChain {
            hashes: Arc::new(hashes),
        };
        let node = MockNode::new(HashMap::new(), filters);
        let filter_requests = node.filter_requests.clone();
        let max_filter_requests = node.max_filter_requests.clone();
        let received = FiltersMan::<FlatFilterStore, MockChain, MockNode>::fetch_filters(
            Arc::new(Mutex::new(store)),
            node,
            chain,
            0,
            count as u32 - 1,
        )
        .await
        .unwrap();

        assert_eq!(received.len(), count);
        assert_eq!(filter_requests.load(Ordering::Relaxed), 2);
        assert_eq!(max_filter_requests.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn rescan_retries_filter_request_errors() {
        let (_file, store, chain, node, block, _filter) = setup();
        node.filter_failures.store(1, Ordering::Relaxed);
        let filter_requests = node.filter_requests.clone();
        let script = block.txdata[0].output[0].script_pubkey.clone();
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());
        let ticket = handle
            .rescan(RescanRequest::new(vec![script]))
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let _ = handle.get_blocks(ticket).await.unwrap();
                if handle.get_info(ticket).await.unwrap() == RescanStatus::Finished {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert_eq!(filter_requests.load(Ordering::Relaxed), 2);
        task.abort();
    }

    #[tokio::test]
    async fn rejects_filter_that_does_not_match_header() {
        let (_file, mut store, chain, mut node, _block, _filter) = setup();
        let block_hash = chain.get_block_hash(0).unwrap();
        store
            .update_filter_header(0, block_hash, FilterHeader::all_zeros())
            .unwrap();
        Arc::make_mut(&mut node.filters).insert(block_hash, BlockFilter::new(&[1, 2, 3]));
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());

        assert!(matches!(
            handle.get_filter(0).await,
            Err(FilterManError::InvalidFilter(0))
        ));

        task.abort();
    }

    #[tokio::test]
    async fn reports_mock_node_filter_failure() {
        let (_file, store, chain, mut node, _block, _filter) = setup();
        Arc::make_mut(&mut node.filters).clear();
        let filter_requests = node.filter_requests.clone();
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());

        assert!(matches!(
            handle.get_filter(0).await,
            Err(FilterManError::Node(_))
        ));
        assert_eq!(filter_requests.load(Ordering::Relaxed), 1);

        task.abort();
    }

    #[tokio::test]
    async fn retries_missing_blocks_during_rescan() {
        let (_file, store, chain, mut node, block, _filter) = setup();
        Arc::make_mut(&mut node.blocks).clear();
        let block_requests = node.block_requests.clone();
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());
        let script = block.txdata[0].output[0].script_pubkey.clone();
        let ticket = handle
            .rescan(RescanRequest::new(vec![script]))
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            while block_requests.load(Ordering::Relaxed) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert!(matches!(
            handle.get_info(ticket).await,
            Ok(RescanStatus::Started | RescanStatus::Waiting)
        ));
        assert!(block_requests.load(Ordering::Relaxed) >= 2);

        task.abort();
    }

    #[tokio::test]
    async fn rejects_invalid_rescans_before_calling_mock_node() {
        let (_file, store, chain, node, _block, _filter) = setup();
        let filter_requests = node.filter_requests.clone();
        let block_requests = node.block_requests.clone();
        let manager = FiltersMan::new(store, node, chain);
        let handle = manager.get_handle();
        let task = tokio::spawn(manager.main_loop());

        assert!(matches!(
            handle.rescan(RescanRequest::new(Vec::new())).await,
            Err(FilterManError::EmptyRescan)
        ));
        assert!(matches!(
            handle
                .rescan(
                    RescanRequest::new(vec![ScriptBuf::new()])
                        .with_max_blocks_per_page(MAX_RESCAN_PAGE_SIZE + 1)
                )
                .await,
            Err(FilterManError::InvalidPageSize(_))
        ));
        assert!(matches!(
            handle
                .rescan(RescanRequest::new(vec![ScriptBuf::new()]).with_range(Some(1), Some(1)))
                .await,
            Err(FilterManError::InvalidRescanRange { .. })
        ));
        assert_eq!(filter_requests.load(Ordering::Relaxed), 0);
        assert_eq!(block_requests.load(Ordering::Relaxed), 0);

        task.abort();
    }
    #[tokio::test]
    async fn connected_block_builds_and_caches_filter() {
        let file = NamedTempFile::new().unwrap();
        let store = FlatFilterStore::new(file.path()).unwrap();
        let block = genesis_block(Network::Regtest);
        let block_hash = block.block_hash();
        let chain = MockChain {
            hashes: Arc::new(vec![block_hash]),
        };
        let node = MockNode::new(HashMap::new(), HashMap::new());
        let mut manager = FiltersMan::new(store, node, chain);

        manager
            .process_connected_block(ConnectedBlock {
                block: block.clone(),
                height: 0,
                spent_utxos: HashMap::new(),
            })
            .await
            .unwrap();

        let expected = BlockFilter::new_script_filter(&block, |outpoint| {
            Err::<ScriptBuf, _>(bitcoin::bip158::Error::UtxoMissing(*outpoint))
        })
        .unwrap();
        let mut store = manager.store.lock().unwrap();
        assert_eq!(store.get_filter(0).unwrap(), Some(expected.clone()));
        assert_eq!(store.get_block_hash(0).unwrap(), block_hash);
        assert_eq!(
            store.get_filter_header(0).unwrap(),
            expected.filter_header(&FilterHeader::all_zeros())
        );
    }

    #[test]
    fn applying_headers_persists_best_chain_mapping() {
        let file = NamedTempFile::new().unwrap();
        let store = FlatFilterStore::new(file.path()).unwrap();
        let block = genesis_block(Network::Regtest);
        let block_hash = block.block_hash();
        let filter = BlockFilter::new(&[1, 2, 3]);
        let filter_hash = bitcoin::FilterHash::hash(&filter.content);
        let chain = MockChain {
            hashes: Arc::new(vec![block_hash]),
        };
        let node = MockNode::new(HashMap::new(), HashMap::new());
        let mut manager = FiltersMan::new(store, node, chain);

        manager
            .apply_headers(
                0,
                0,
                block_hash,
                CFHeaders {
                    filter_type: BASIC_FILTER_TYPE,
                    stop_hash: block_hash,
                    previous_filter_header: FilterHeader::all_zeros(),
                    filter_hashes: vec![filter_hash],
                },
            )
            .unwrap();

        let mut store = manager.store.lock().unwrap();
        assert_eq!(store.get_block_hash(0).unwrap(), block_hash);
        assert_eq!(
            store.get_filter_header(0).unwrap(),
            filter.filter_header(&FilterHeader::all_zeros())
        );
    }

    #[tokio::test]
    async fn sync_requests_headers_and_checkpoints_from_mock_node() {
        let file = NamedTempFile::new().unwrap();
        let store = FlatFilterStore::new(file.path()).unwrap();
        let hashes = (0_u32..=CHECKPOINT_INTERVAL)
            .map(mock_block_hash)
            .collect::<Vec<_>>();
        let filter_hashes = (0_u32..=CHECKPOINT_INTERVAL)
            .map(|height| {
                let mut bytes = [1_u8; 32];
                bytes[..4].copy_from_slice(&height.to_le_bytes());
                bitcoin::FilterHash::from_byte_array(bytes)
            })
            .collect::<Vec<_>>();
        let mut previous_header = FilterHeader::all_zeros();
        let filter_headers = filter_hashes
            .iter()
            .map(|filter_hash| {
                previous_header = filter_hash.filter_header(&previous_header);
                previous_header
            })
            .collect::<Vec<_>>();
        let checkpoint = filter_headers[CHECKPOINT_INTERVAL as usize];
        let stop_999 = hashes[(CHECKPOINT_INTERVAL - 1) as usize];
        let stop_1000 = hashes[CHECKPOINT_INTERVAL as usize];
        let chain = MockChain {
            hashes: Arc::new(hashes),
        };
        let mut node = MockNode::new(HashMap::new(), HashMap::new());
        node.header_responses = Arc::new(HashMap::from([
            (
                (0, stop_999),
                CFHeaders {
                    filter_type: BASIC_FILTER_TYPE,
                    stop_hash: stop_999,
                    previous_filter_header: FilterHeader::all_zeros(),
                    filter_hashes: filter_hashes[..CHECKPOINT_INTERVAL as usize].to_vec(),
                },
            ),
            (
                (CHECKPOINT_INTERVAL, stop_1000),
                CFHeaders {
                    filter_type: BASIC_FILTER_TYPE,
                    stop_hash: stop_1000,
                    previous_filter_header: filter_headers[(CHECKPOINT_INTERVAL - 1) as usize],
                    filter_hashes: vec![filter_hashes[CHECKPOINT_INTERVAL as usize]],
                },
            ),
        ]));
        node.checkpoint_responses = Arc::new(HashMap::from([(
            stop_1000,
            CFCheckpt {
                filter_type: BASIC_FILTER_TYPE,
                stop_hash: stop_1000,
                filter_headers: vec![checkpoint],
            },
        )]));
        let header_requests = node.header_requests.clone();
        let checkpoint_requests = node.checkpoint_requests.clone();
        let mut manager = FiltersMan::new(store, node, chain);

        manager.sync().await.unwrap();

        assert_eq!(header_requests.load(Ordering::Relaxed), 2);
        assert_eq!(checkpoint_requests.load(Ordering::Relaxed), 1);
        let mut store = manager.store.lock().unwrap();
        assert_eq!(store.get_height().unwrap(), Some(CHECKPOINT_INTERVAL));
        assert_eq!(
            store.get_filter_header(CHECKPOINT_INTERVAL).unwrap(),
            checkpoint
        );
        assert_eq!(
            store.get_block_hash(CHECKPOINT_INTERVAL).unwrap(),
            stop_1000
        );
    }
    #[tokio::test]
    async fn rejects_missing_filter_header_checkpoint() {
        let file = NamedTempFile::new().unwrap();
        let store = FlatFilterStore::new(file.path()).unwrap();
        let hashes = (0_u32..=CHECKPOINT_INTERVAL)
            .map(|height| {
                let mut bytes = [0_u8; 32];
                bytes[..4].copy_from_slice(&height.to_le_bytes());
                BlockHash::from_byte_array(bytes)
            })
            .collect::<Vec<_>>();
        let stop_hash = hashes[CHECKPOINT_INTERVAL as usize];
        let chain = MockChain {
            hashes: Arc::new(hashes),
        };
        let mut node = MockNode::new(HashMap::new(), HashMap::new());
        node.checkpoint_responses = Arc::new(HashMap::from([(
            stop_hash,
            CFCheckpt {
                filter_type: BASIC_FILTER_TYPE,
                stop_hash,
                filter_headers: Vec::new(),
            },
        )]));
        let mut manager = FiltersMan::new(store, node, chain);

        assert!(matches!(
            manager.sync_checkpoints(CHECKPOINT_INTERVAL).await,
            Err(FilterManError::InvalidHeaders(_))
        ));
    }
}

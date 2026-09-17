// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use core::fmt;
use std::error::Error;

use bitcoin::Block;
use bitcoin::BlockHash;
use bitcoin::FilterHeader;
use bitcoin::Network;
use bitcoin::ScriptBuf;
use bitcoin::bip158::BlockFilter;
use bitcoin::blockdata::constants::genesis_block;
use bitcoin::hashes::Hash;
use bitcoin::p2p::message_filter::CFCheckpt;
use bitcoin::p2p::message_filter::CFHeaders;
use floresta_compact_filters::FilterHeadersStore;
use floresta_compact_filters::FlatFilterStoreError;
use floresta_compact_filters::filters_man::FilterChain;
use floresta_compact_filters::filters_man::FilterManError;
use floresta_compact_filters::filters_man::FiltersMan;
use floresta_compact_filters::filters_man::RescanRequest;
use floresta_compact_filters::filters_man::RescanStatus;
use floresta_wire::node_interface::ChainMethods;
use libfuzzer_sys::arbitrary::Arbitrary;
use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

#[derive(Debug)]
struct FuzzInput {
    filter_bytes: Vec<u8>,
    use_well_formed_filter: bool,
    missing_filter: bool,
    invalid_header: bool,
    missing_block: bool,
    empty_scripts: bool,
    page_size: u16,
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let filter_len = input.int_in_range(0..=512)?;
        let filter_bytes = input.bytes(filter_len)?.to_vec();
        Ok(Self {
            filter_bytes,
            use_well_formed_filter: input.arbitrary()?,
            missing_filter: input.arbitrary()?,
            invalid_header: input.arbitrary()?,
            missing_block: input.arbitrary()?,
            empty_scripts: input.arbitrary()?,
            page_size: input.arbitrary()?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct MockError;

impl fmt::Display for MockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("mock node failure")
    }
}

impl Error for MockError {}

#[derive(Clone)]
struct MockChain {
    block_hash: BlockHash,
}

impl FilterChain for MockChain {
    type Error = MockError;

    fn get_height(&self) -> Result<u32, Self::Error> {
        Ok(0)
    }

    fn get_block_hash(&self, height: u32) -> Result<BlockHash, Self::Error> {
        (height == 0).then_some(self.block_hash).ok_or(MockError)
    }
}

#[derive(Clone)]
struct MockNode {
    block: Option<Block>,
    filter: Option<BlockFilter>,
}

impl ChainMethods for MockNode {
    type Error = MockError;

    async fn get_block(&self, block_hash: BlockHash) -> Result<Option<Block>, Self::Error> {
        Ok(self
            .block
            .as_ref()
            .filter(|block| block.block_hash() == block_hash)
            .cloned())
    }

    async fn get_cfilters_headers(
        &self,
        _start_height: u32,
        _stop_hash: BlockHash,
    ) -> Result<CFHeaders, Self::Error> {
        Err(MockError)
    }

    async fn get_cfilter(
        &self,
        start_height: u32,
        block_hashes: Vec<BlockHash>,
    ) -> Result<Vec<BlockFilter>, Self::Error> {
        if start_height != 0 || block_hashes.len() != 1 {
            return Err(MockError);
        }
        self.filter
            .clone()
            .map(|filter| vec![filter])
            .ok_or(MockError)
    }

    async fn get_cfcheckpt(&self, _stop_hash: BlockHash) -> Result<CFCheckpt, Self::Error> {
        Err(MockError)
    }
}

struct MemoryStore {
    block_hash: BlockHash,
    filter_header: FilterHeader,
    filter: Option<BlockFilter>,
}

impl FilterHeadersStore for MemoryStore {
    fn put_filter_header(
        &mut self,
        block_hash: BlockHash,
        filter_header: FilterHeader,
    ) -> Result<(), FlatFilterStoreError> {
        self.block_hash = block_hash;
        self.filter_header = filter_header;
        Ok(())
    }

    fn get_filter_header(&mut self, height: u32) -> Result<FilterHeader, FlatFilterStoreError> {
        (height == 0)
            .then_some(self.filter_header)
            .ok_or(FlatFilterStoreError::NotFound)
    }

    fn get_block_hash(&mut self, height: u32) -> Result<BlockHash, FlatFilterStoreError> {
        (height == 0)
            .then_some(self.block_hash)
            .ok_or(FlatFilterStoreError::NotFound)
    }

    fn update_filter_header(
        &mut self,
        height: u32,
        block_hash: BlockHash,
        filter_header: FilterHeader,
    ) -> Result<FilterHeader, FlatFilterStoreError> {
        if height != 0 {
            return Err(FlatFilterStoreError::NotFound);
        }
        let previous = self.filter_header;
        self.block_hash = block_hash;
        self.filter_header = filter_header;
        Ok(previous)
    }

    fn get_height(&self) -> Result<Option<u32>, FlatFilterStoreError> {
        Ok(Some(0))
    }

    fn truncate(&mut self, height: Option<u32>) -> Result<(), FlatFilterStoreError> {
        if height == Some(0) {
            Ok(())
        } else {
            Err(FlatFilterStoreError::NotFound)
        }
    }

    fn put_filter(&mut self, height: u32, filter: BlockFilter) -> Result<(), FlatFilterStoreError> {
        if height != 0 {
            return Err(FlatFilterStoreError::NotFound);
        }
        self.filter = Some(filter);
        Ok(())
    }

    fn get_filter(&self, height: u32) -> Result<Option<BlockFilter>, FlatFilterStoreError> {
        if height != 0 {
            return Err(FlatFilterStoreError::NotFound);
        }
        Ok(self.filter.clone())
    }

    fn flush(&mut self) -> Result<(), FlatFilterStoreError> {
        Ok(())
    }
}

fuzz_target!(|input: FuzzInput| {
    let block = genesis_block(Network::Regtest);
    let block_hash = block.block_hash();
    let well_formed_filter = BlockFilter::new_script_filter(&block, |outpoint| {
        Err::<ScriptBuf, _>(bitcoin::bip158::Error::UtxoMissing(*outpoint))
    })
    .expect("the genesis coinbase has no spent outputs");
    let filter = if input.use_well_formed_filter {
        well_formed_filter
    } else {
        BlockFilter::new(&input.filter_bytes)
    };
    let mut header_bytes = filter
        .filter_header(&FilterHeader::all_zeros())
        .to_byte_array();
    if input.invalid_header {
        header_bytes[0] ^= 1;
    }
    let filter_header = FilterHeader::from_byte_array(header_bytes);
    let store = MemoryStore {
        block_hash,
        filter_header,
        filter: None,
    };
    let node = MockNode {
        block: (!input.missing_block).then_some(block.clone()),
        filter: (!input.missing_filter).then_some(filter.clone()),
    };
    let manager = FiltersMan::new(store, node, MockChain { block_hash });
    let handle = manager.get_handle();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a current-thread runtime can be created");

    runtime.block_on(async move {
        let manager_task = tokio::spawn(manager.main_loop());
        let filter_result = handle.get_filter(0).await;
        if input.missing_filter {
            assert!(matches!(filter_result, Err(FilterManError::Node(_))));
        } else if input.invalid_header {
            assert!(matches!(
                filter_result,
                Err(FilterManError::InvalidFilter(0))
            ));
        } else {
            assert_eq!(filter_result.expect("valid filter is accepted"), filter);
        }

        let scripts = if input.empty_scripts {
            Vec::new()
        } else {
            vec![block.txdata[0].output[0].script_pubkey.clone()]
        };
        let request =
            RescanRequest::new(scripts).with_max_blocks_per_page(usize::from(input.page_size));
        match handle.rescan(request).await {
            Err(FilterManError::EmptyRescan | FilterManError::InvalidPageSize(_)) => {}
            Err(_) => {}
            Ok(ticket) => {
                for _ in 0..16 {
                    match handle.get_info(ticket).await {
                        Ok(RescanStatus::Available) => {
                            let _ = handle.get_blocks(ticket).await;
                        }
                        Ok(RescanStatus::Finished) | Err(_) => break,
                        Ok(RescanStatus::Started | RescanStatus::Waiting) => {
                            tokio::task::yield_now().await;
                        }
                    }
                }
            }
        }
        manager_task.abort();
    });
});

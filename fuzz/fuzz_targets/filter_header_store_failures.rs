// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use std::io::Write;

use bitcoin::BlockHash;
use bitcoin::FilterHeader;
use bitcoin::bip158::BlockFilter;
use bitcoin::hashes::Hash;
use floresta_compact_filters::FilterHeaderEntry;
use floresta_compact_filters::FilterHeadersStore;
use floresta_compact_filters::FlatFilterStore;
use floresta_compact_filters::FlatFilterStoreError;
use libfuzzer_sys::arbitrary::Arbitrary;
use libfuzzer_sys::arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use tempfile::NamedTempFile;

#[derive(Debug)]
struct FuzzInput {
    file_bytes: Vec<u8>,
    height: u32,
    truncate_height: Option<u32>,
    cache_capacity: u8,
    filter_bytes: Vec<u8>,
    block_hash: [u8; 32],
    filter_header: [u8; 32],
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let file_len = input.int_in_range(0..=4_096)?;
        let file_bytes = input.bytes(file_len)?.to_vec();
        let filter_len = input.int_in_range(0..=1_024)?;
        let filter_bytes = input.bytes(filter_len)?.to_vec();
        Ok(Self {
            file_bytes,
            height: input.arbitrary()?,
            truncate_height: input.arbitrary()?,
            cache_capacity: input.arbitrary()?,
            filter_bytes,
            block_hash: input.arbitrary()?,
            filter_header: input.arbitrary()?,
        })
    }
}

fuzz_target!(|input: FuzzInput| {
    let mut file = match NamedTempFile::new() {
        Ok(file) => file,
        Err(_) => return,
    };
    if file.write_all(&input.file_bytes).is_err() {
        return;
    }

    let record_size = FilterHeaderEntry::SERIALIZED_SIZE as usize;
    let mut store = match FlatFilterStore::with_cache_capacity(
        file.path(),
        usize::from(input.cache_capacity),
    ) {
        Ok(store) => {
            assert_eq!(input.file_bytes.len() % record_size, 0);
            store
        }
        Err(FlatFilterStoreError::CorruptedFile) => {
            assert_ne!(input.file_bytes.len() % record_size, 0);
            return;
        }
        Err(_) => return,
    };

    let entry_count = input.file_bytes.len() / record_size;
    let expected_height = u32::try_from(entry_count)
        .ok()
        .and_then(|count| count.checked_sub(1));
    assert_eq!(store.get_height().ok().flatten(), expected_height);

    let block_hash = BlockHash::from_byte_array(input.block_hash);
    let filter_header = FilterHeader::from_byte_array(input.filter_header);
    let filter = BlockFilter::new(&input.filter_bytes);

    let _ = store.get_filter_header(input.height);
    let _ = store.get_block_hash(input.height);
    let _ = store.update_filter_header(input.height, block_hash, filter_header);
    let truncate_result = store.truncate(input.truncate_height);
    if input
        .truncate_height
        .is_some_and(|height| usize::try_from(height).unwrap_or(usize::MAX) >= entry_count)
    {
        assert!(matches!(
            truncate_result,
            Err(FlatFilterStoreError::NotFound)
        ));
    }

    let _ = store.put_filter(input.height, filter.clone());
    let _ = store.get_filter(input.height);
    let _ = store.flush();
    drop(store);

    if let Ok(mut reopened) = FlatFilterStore::with_cache_capacity(file.path(), 0) {
        let _ = reopened.get_height();
        let _ = reopened.get_filter_header(input.height);
        let _ = reopened.get_block_hash(input.height);
    }
});

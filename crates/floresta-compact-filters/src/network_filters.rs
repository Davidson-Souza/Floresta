use bitcoin::BlockHash;
use floresta_chain::pruned_utreexo::BlockchainInterface;

use crate::BlockFilterStore;

#[derive(Debug)]
pub struct NetworkFilters<Storage: BlockFilterStore + Send + Sync> {
    filters: Storage,
}

impl<Storage: BlockFilterStore + Send + Sync> NetworkFilters<Storage> {
    pub fn new(filters: Storage) -> Self {
        Self { filters }
    }

    pub fn get_filter(&self, height: u32) -> Option<crate::BlockFilter> {
        self.filters.get_filter(height)
    }

    pub fn match_any(
        &self,
        query: Vec<&[u8]>,
        start_height: u32,
        end_height: u32,
        chain: impl BlockchainInterface,
    ) -> Vec<BlockHash> {
        let mut blocks = Vec::new();
        for height in start_height..end_height {
            let Some(filter) = self.filters.get_filter(height) else {
                continue;
            };

            let mut query = query.clone().into_iter();
            let hash = chain.get_block_hash(height).unwrap();

            if filter.match_any(&hash, &mut query).unwrap() {
                let block_hash = chain.get_block_hash(height).unwrap();
                blocks.push(block_hash);
            }
        }

        blocks
    }

    pub fn push_filter(&self, height: u32, filter: crate::BlockFilter) {
        self.filters.put_filter(height, filter);
        self.filters.put_height(height);
    }

    pub fn get_height(&self) -> u32 {
        self.filters.get_height().unwrap_or(0)
    }

    pub fn save_height(&self, height: u32) {
        self.filters.put_height(height);
    }
}

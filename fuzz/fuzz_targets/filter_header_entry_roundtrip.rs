// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use std::io::Cursor;

use bitcoin::consensus::Decodable;
use bitcoin::consensus::Encodable;
use floresta_compact_filters::FilterHeaderEntry;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to decode from arbitrary bytes. Decoding must never panic.
    if let Ok(dec) = FilterHeaderEntry::consensus_decode(&mut Cursor::new(data)) {
        // Encode
        let mut buf = Vec::new();
        let written = dec.consensus_encode(&mut buf).expect("encode failed");
        assert_eq!(written, buf.len(), "encode returned wrong length");

        // Re-decode and compare
        let dec2 =
            FilterHeaderEntry::consensus_decode(&mut Cursor::new(&buf)).expect("decode failed");
        assert_eq!(dec2, dec, "encode/decode not stable");
    }
});

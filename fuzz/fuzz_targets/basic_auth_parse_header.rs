// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    #[cfg(feature = "json-rpc")]
    {
        let Ok(s) = core::str::from_utf8(data) else {
            return;
        };
        let _ = floresta_node::basic::Credentials::parse(s);
    }
    #[cfg(not(feature = "json-rpc"))]
    let _ = data;
});

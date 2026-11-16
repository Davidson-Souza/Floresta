// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core domain types for Floresta implementation.
//!
//! This crate provides the fundamental data structures and abstractions used across
//! [Floresta](https://github.com/getfloresta/floresta).

// cargo docs customization
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_logo_url = "https://avatars.githubusercontent.com/u/249173822")]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/getfloresta/floresta-media/master/logo_png/Icon-Green(main).png"
)]

#[cfg(feature = "node-interface")]
pub mod node_interface;

#[cfg(feature = "mempool")]
pub mod mempool;

#[cfg(feature = "wallet")]
pub mod wallet;

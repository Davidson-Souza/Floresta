// SPDX-License-Identifier: MIT OR Apache-2.0

//! JSON-RPC authentication primitives, mirroring Bitcoin Core.
//!
//! [`generate_cookie`] writes `__cookie__:<64-char-hex>` to the caller's path,
//! rotating the token each restart. Clients then send
//! `Authorization: Basic <base64(__cookie__:<token>)>`.

use core::error::Error;
use core::fmt;
use std::ffi::OsString;
use std::fs;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use axum::body::Body;
use axum::extract::Request;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header::AUTHORIZATION;
use axum::http::header::WWW_AUTHENTICATE;
use axum::middleware::Next;
use axum::response::Response;
use bitcoin::hashes::Hash;
use bitcoin::hashes::HashEngine;
use bitcoin::hashes::Hmac;
use bitcoin::hashes::HmacEngine;
use bitcoin::hashes::sha256;
use bitcoin::hex::DisplayHex;

/// Username literal used for cookie auth (Core convention).
pub(crate) const COOKIE_USER: &str = "__cookie__";

/// Token length in raw random bytes; hex-encoded to 64 ASCII chars.
const COOKIE_TOKEN_BYTES: usize = 32;

/// Write `<user>:<token_hex>` to `path` and return the pair, so the caller
/// feeds the auth vector exactly what landed in the file.
///
/// Goes via `<path>.tmp`, unlinked first then opened with `create_new` so mode
/// `0600` always applies and a planted symlink is refused, then renamed.
pub(crate) fn generate_cookie(path: &Path) -> io::Result<(String, String)> {
    let token: [u8; COOKIE_TOKEN_BYTES] = rand::random();
    let token_hex = token.to_lower_hex_string();
    let line = format!("{COOKIE_USER}:{token_hex}");

    let tmp = tmp_path(path);

    // Clears a stale tmp from a crashed run, and any symlink left in its place.
    match fs::remove_file(&tmp) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }

    let mut file = opts.open(&tmp)?;
    file.write_all(line.as_bytes())?;
    drop(file);

    fs::rename(&tmp, path)?;

    Ok((COOKIE_USER.to_string(), token_hex))
}

/// `Authorization: Basic` header parsing.
pub mod basic {
    use core::error::Error;
    use core::fmt;

    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64;

    /// Upper bound on the inbound header to cap base64 decode allocation.
    pub(super) const MAX_LEN: usize = 16 * 1024;

    #[derive(Debug, PartialEq, Eq)]
    /// Errors produced by [`Credentials::parse`].
    pub enum HeaderError {
        /// The header value did not start with the literal `"Basic "` prefix.
        MissingBasicPrefix,

        /// The base64 payload could not be decoded.
        InvalidBase64,

        /// The decoded payload contained non-ASCII bytes.
        NonAsciiPayload,

        /// The decoded payload contained no `:` separator.
        MissingColon,

        /// The header value exceeded the inbound length cap.
        PayloadTooLarge,
    }

    impl fmt::Display for HeaderError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::MissingBasicPrefix => {
                    write!(f, "Authorization header must start with 'Basic '")
                }
                Self::InvalidBase64 => write!(f, "Authorization payload is not valid base64"),
                Self::NonAsciiPayload => {
                    write!(f, "Authorization payload contains non-ASCII bytes")
                }
                Self::MissingColon => write!(f, "Authorization payload missing ':' separator"),
                Self::PayloadTooLarge => {
                    write!(f, "Authorization header exceeds {MAX_LEN}-byte cap")
                }
            }
        }
    }

    impl Error for HeaderError {}

    #[derive(Debug, PartialEq, Eq)]
    /// Credentials claimed by an inbound `Authorization: Basic` header.
    ///
    /// These are unverified; matching them against the configured RPC
    /// credentials is the caller's job.
    pub struct Credentials {
        /// The username, everything before the first `:`.
        pub user: String,

        /// The password, everything after the first `:`.
        pub pass: String,
    }

    impl Credentials {
        /// Parse an HTTP `Authorization: Basic <b64>` header value.
        ///
        /// Mirrors Bitcoin Core: the `"Basic "` prefix check is case-sensitive, the
        /// base64 payload is whitespace-trimmed before decoding, and the split
        /// between user and password is on the **first** `:`. Passwords may contain
        /// `:`; usernames may not.
        pub fn parse(value: &str) -> Result<Self, HeaderError> {
            if value.len() > MAX_LEN {
                return Err(HeaderError::PayloadTooLarge);
            }
            let payload = value
                .strip_prefix("Basic ")
                .ok_or(HeaderError::MissingBasicPrefix)?
                .trim();
            let decoded = BASE64
                .decode(payload)
                .map_err(|_| HeaderError::InvalidBase64)?;
            if !decoded.is_ascii() {
                return Err(HeaderError::NonAsciiPayload);
            }
            // Unreachable after the ASCII check above, but mapped rather than
            // unwrapped so the parser has no panic path at all.
            let decoded =
                core::str::from_utf8(&decoded).map_err(|_| HeaderError::NonAsciiPayload)?;
            let (user, pass) = decoded.split_once(':').ok_or(HeaderError::MissingColon)?;
            Ok(Self {
                user: user.to_string(),
                pass: pass.to_string(),
            })
        }
    }
}

/// Axum middleware gating each request on the configured [`Auth`]. Anything
/// but a match returns 401 with `WWW-Authenticate` per RFC 7235.
pub(crate) async fn auth_middleware(
    State(creds): State<std::sync::Arc<Auth>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(header) = req.headers().get(AUTHORIZATION) else {
        tracing::debug!("rpc auth header missing; rejecting");
        return unauthorized();
    };
    let value = match header.to_str() {
        Ok(s) => s,
        Err(_) => {
            tracing::debug!("rpc auth header is not valid ascii; rejecting");
            return unauthorized();
        }
    };
    let claimed = match basic::Credentials::parse(value) {
        Ok(creds) => creds,
        Err(e) => {
            tracing::debug!("rpc auth header parse failed: {e}; rejecting");
            return unauthorized();
        }
    };
    if !creds.matches(&claimed.user, &claimed.pass) {
        let user = &claimed.user;
        tracing::debug!("rpc auth credentials mismatched for user {user}; rejecting");
        return unauthorized();
    }
    tracing::debug!("rpc auth ok for user {}", claimed.user);
    next.run(req).await
}

fn unauthorized() -> Response {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header(WWW_AUTHENTICATE, r#"Basic realm="jsonrpc""#)
        .body(Body::empty())
        .expect("static 401 response is always well-formed")
}

/// Configured RPC credentials for this process: cookie, `-rpcuser`/`-rpcpassword`,
/// and any `-rpcauth` entries all live in a single vector. First match wins.
/// Mirrors Bitcoin Core's `g_rpcauth` (httprpc.cpp:36).
pub(crate) struct Auth {
    entries: Vec<RpcAuth>,
}

impl Auth {
    /// Build from a pre-collected vector of entries.
    pub(crate) fn new(entries: Vec<RpcAuth>) -> Self {
        Self { entries }
    }

    /// True if the basic-auth `user`/`pass` pair authenticates against any
    /// configured entry. Each entry's check is constant-time.
    pub(crate) fn matches(&self, user: &str, pass: &str) -> bool {
        self.entries.iter().any(|a| a.verify(user, pass))
    }
}

/// One salted-hash credential entry: `(user, salt_hex, hash_hex)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RpcAuth {
    pub user: String,
    pub salt_hex: String,
    pub hash_hex: String,
}

impl RpcAuth {
    /// Roll a fresh salt and HMAC the password.
    pub(crate) fn from_password(user: &str, pass: &str) -> Self {
        let salt_hex = generate_salt_hex();
        let hash_hex = hmac_password(&salt_hex, pass);
        Self {
            user: user.to_string(),
            salt_hex,
            hash_hex,
        }
    }

    /// Parse a `<user>:<salt_hex>$<hash_hex>` line, mirroring Bitcoin Core.
    pub(crate) fn parse(line: &str) -> Result<Self, RpcAuthParseError> {
        let colon_parts: Vec<&str> = line.split(':').collect();
        if colon_parts.len() != 2 {
            return Err(RpcAuthParseError::MalformedLine);
        }
        let user = colon_parts[0];
        let salt_hash: Vec<&str> = colon_parts[1].split('$').collect();
        if salt_hash.len() != 2 {
            return Err(RpcAuthParseError::MalformedLine);
        }
        let salt_hex = salt_hash[0];
        let hash_hex = salt_hash[1];
        if user.is_empty() || salt_hex.is_empty() || hash_hex.is_empty() {
            return Err(RpcAuthParseError::EmptyField);
        }
        // ASCII-only header parse; a non-ASCII user never matches.
        if !user.is_ascii() {
            return Err(RpcAuthParseError::NonAsciiUser);
        }
        // The hash is compared against lowercase hex we compute ourselves, so
        // an uppercase hash can never match. The salt is only ever used
        // verbatim as the HMAC key, so its case is irrelevant.
        let is_lowercase_hex = |s: &str| {
            s.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        };
        if !is_lowercase_hex(hash_hex) {
            return Err(RpcAuthParseError::NonLowercaseHash);
        }
        Ok(Self {
            user: user.to_string(),
            salt_hex: salt_hex.to_string(),
            hash_hex: hash_hex.to_string(),
        })
    }

    /// Constant-time check: user matches and HMAC(pass) equals the stored hash.
    pub(crate) fn verify(&self, user: &str, pass: &str) -> bool {
        if !constant_time_eq(user.as_bytes(), self.user.as_bytes()) {
            return false;
        }
        let computed = hmac_password(&self.salt_hex, pass);
        constant_time_eq(computed.as_bytes(), self.hash_hex.as_bytes())
    }
}

#[derive(Debug, PartialEq, Eq)]
/// Errors from [`RpcAuth::parse`].
pub(crate) enum RpcAuthParseError {
    /// Wrong number of `:` or `$` separators.
    MalformedLine,

    /// User, salt, or hash field is empty.
    EmptyField,

    /// Hash field contains non-`[0-9a-f]` characters, so it can never match.
    NonLowercaseHash,

    /// User field is not ASCII, so a `Basic` header carrying it never parses.
    NonAsciiUser,
}

impl fmt::Display for RpcAuthParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Match Bitcoin Core's wording (httprpc.cpp:308) for the operator log.
        write!(f, "Invalid -rpcauth argument.")
    }
}

impl Error for RpcAuthParseError {}

/// Constant-time byte slice comparison. Returns `false` immediately on length
/// mismatch (lengths of both comparands are public), then XORs every byte into
/// an accumulator before returning.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
}

/// Generate a fresh 16-byte salt, lowercase-hex-encoded (32 ASCII chars).
fn generate_salt_hex() -> String {
    let bytes: [u8; 16] = rand::random();
    bytes.to_lower_hex_string()
}

/// HMAC-SHA256 with `salt_hex.as_bytes()` as the key (NOT decoded bytes; this
/// is the interop hinge with Bitcoin Core's `rpcauth.py`) and `pass` as the
/// message; returns the 32-byte digest as 64 lowercase hex chars.
fn hmac_password(salt_hex: &str, pass: &str) -> String {
    let mut engine = HmacEngine::<sha256::Hash>::new(salt_hex.as_bytes());
    engine.input(pass.as_bytes());
    Hmac::<sha256::Hash>::from_engine(engine)
        .to_byte_array()
        .to_lower_hex_string()
}

/// Remove the cookie file at `path`. Treats `NotFound` as success so shutdown
/// is idempotent. Caller must only invoke this after a successful
/// [`generate_cookie`] in this process.
pub(crate) fn delete_cookie(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut buf = OsString::from(path);
    buf.push(".tmp");
    PathBuf::from(buf)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn tmp_cookie_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "floresta_cookie_test_{name}_{}",
            rand::random::<u32>()
        ));
        p
    }

    #[test]
    fn generate_cookie_writes_expected_format() {
        let path = tmp_cookie_path("format");
        let (user, token) = generate_cookie(&path).expect("cookie write should succeed in test");

        assert_eq!(user, "__cookie__");
        // Hex is ASCII, so `len()` in bytes equals the character count.
        assert_eq!(
            token.len(),
            2 * COOKIE_TOKEN_BYTES,
            "token should be {} hex chars, got {}",
            2 * COOKIE_TOKEN_BYTES,
            token.len()
        );
        assert!(
            token
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "token should be lowercase hex: {token}",
        );

        let written = fs::read_to_string(&path).expect("test wrote this cookie file above");
        assert_eq!(
            written,
            format!("{user}:{token}"),
            "file content should match returned (user, token) joined by ':'",
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn generate_cookie_writes_no_trailing_newline() {
        let path = tmp_cookie_path("newline");
        generate_cookie(&path).expect("cookie write should succeed in test");

        let bytes = fs::read(&path).expect("test wrote this cookie file above");
        assert!(!bytes.ends_with(b"\n"), "file should not end with newline");

        fs::remove_file(&path).ok();
    }

    #[test]
    fn generate_cookie_produces_distinct_tokens() {
        let path1 = tmp_cookie_path("distinct1");
        let path2 = tmp_cookie_path("distinct2");
        let (_, t1) = generate_cookie(&path1).expect("cookie write should succeed in test");
        let (_, t2) = generate_cookie(&path2).expect("cookie write should succeed in test");
        assert_ne!(t1, t2, "two consecutive calls produced identical tokens");

        fs::remove_file(&path1).ok();
        fs::remove_file(&path2).ok();
    }

    #[test]
    fn generate_cookie_overwrites_existing_file() {
        let path = tmp_cookie_path("overwrite");
        fs::write(&path, "stale-content").expect("test temp path should be writable");
        let (user, token) = generate_cookie(&path).expect("cookie write should succeed in test");
        let written = fs::read_to_string(&path).expect("test wrote this cookie file above");
        assert_eq!(
            written,
            format!("{user}:{token}"),
            "file content should match returned (user, token) joined by ':'",
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn generate_cookie_leaves_no_tmp_file() {
        let path = tmp_cookie_path("notmp");
        generate_cookie(&path).expect("cookie write should succeed in test");
        let tmp = tmp_path(&path);
        assert!(
            !tmp.exists(),
            "tmp file should be renamed away, got {tmp:?}"
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn generate_cookie_recovers_from_stale_tmp_file() {
        let path = tmp_cookie_path("stale_tmp");
        let tmp = tmp_path(&path);

        // Simulate a previous run that crashed between create and rename:
        // a stale <path>.tmp file lingers with arbitrary content, left
        // world-readable so we'd notice if its mode carried over.
        fs::write(&tmp, b"partial-from-crashed-run").expect("test temp path should be writable");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&tmp, fs::Permissions::from_mode(0o644))
                .expect("test wrote this tmp file above");
        }
        assert!(tmp.exists(), "precondition: stale tmp must exist");

        generate_cookie(&path).expect("cookie write should succeed in test");

        let written = fs::read_to_string(&path).expect("test wrote this cookie file above");
        assert!(
            written.starts_with("__cookie__:"),
            "cookie file should hold a fresh cookie, got {written}"
        );
        assert!(!tmp.exists(), "stale tmp file should be consumed by rename");

        // The stale tmp's looser mode must not survive into the cookie.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path)
                .expect("test wrote this cookie file above")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "cookie must stay owner-only, got {mode:o}");
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(unix)]
    /// A symlink at the tmp path must not be followed: doing so writes the
    /// token into the link's target and leaves the cookie path a symlink.
    fn generate_cookie_does_not_follow_a_symlinked_tmp_path() {
        use std::os::unix::fs::symlink;

        let path = tmp_cookie_path("symlink_tmp");
        let tmp = tmp_path(&path);
        let target = tmp_cookie_path("symlink_target");

        fs::write(&target, b"ORIGINAL").expect("test temp path should be writable");
        fs::remove_file(&tmp).ok();
        symlink(&target, &tmp).expect("test temp path should be writable");

        generate_cookie(&path).expect("cookie write should succeed in test");

        let target_now = fs::read_to_string(&target).expect("test wrote this file above");
        assert_eq!(
            target_now, "ORIGINAL",
            "the symlink target must not receive the token"
        );
        assert!(
            !fs::symlink_metadata(&path)
                .expect("test wrote this cookie file above")
                .file_type()
                .is_symlink(),
            "the cookie path must be a regular file, not a symlink"
        );

        fs::remove_file(&path).ok();
        fs::remove_file(&target).ok();
    }

    #[test]
    fn delete_cookie_removes_existing_file() {
        let path = tmp_cookie_path("delete_existing");
        generate_cookie(&path).expect("cookie write should succeed in test");
        assert!(path.exists());

        delete_cookie(&path).expect("delete should succeed for an existing file");
        assert!(!path.exists());
    }

    #[test]
    fn delete_cookie_is_idempotent_on_missing_file() {
        let path = tmp_cookie_path("delete_missing");
        assert!(!path.exists());

        delete_cookie(&path).expect("delete is idempotent on missing file");
        delete_cookie(&path).expect("delete is idempotent on missing file");
    }

    #[test]
    fn parse_basic_auth_header_accepts_valid_header() {
        // base64("alice:hunter2") = "YWxpY2U6aHVudGVyMg=="
        let creds = basic::Credentials::parse("Basic YWxpY2U6aHVudGVyMg==")
            .expect("valid Basic header should parse");
        assert_eq!(creds.user, "alice");
        assert_eq!(creds.pass, "hunter2");
    }

    #[test]
    fn parse_basic_auth_header_trims_whitespace_around_payload() {
        let creds = basic::Credentials::parse("Basic   YWxpY2U6aHVudGVyMg==  ")
            .expect("valid Basic header with whitespace should parse");
        assert_eq!(creds.user, "alice");
        assert_eq!(creds.pass, "hunter2");
    }

    #[test]
    fn parse_basic_auth_header_allows_colon_in_password() {
        // base64("alice:pass:with:colons") = "YWxpY2U6cGFzczp3aXRoOmNvbG9ucw=="
        let creds = basic::Credentials::parse("Basic YWxpY2U6cGFzczp3aXRoOmNvbG9ucw==")
            .expect("valid Basic header with colons in pass should parse");
        assert_eq!(creds.user, "alice");
        assert_eq!(creds.pass, "pass:with:colons");
    }

    #[test]
    fn parse_basic_auth_header_rejects_wrong_scheme() {
        // The "Basic " prefix is case-sensitive; reject "basic ", "Bearer ...", and empty.
        assert_eq!(
            basic::Credentials::parse("basic YWxpY2U6aHVudGVyMg=="),
            Err(basic::HeaderError::MissingBasicPrefix),
        );
        assert_eq!(
            basic::Credentials::parse("Bearer YWxpY2U6aHVudGVyMg=="),
            Err(basic::HeaderError::MissingBasicPrefix),
        );
        assert_eq!(
            basic::Credentials::parse(""),
            Err(basic::HeaderError::MissingBasicPrefix),
        );
    }

    #[test]
    fn parse_basic_auth_header_rejects_bad_base64() {
        assert_eq!(
            basic::Credentials::parse("Basic !!!not-base64!!!"),
            Err(basic::HeaderError::InvalidBase64),
        );
    }

    #[test]
    fn parse_basic_auth_header_rejects_missing_colon() {
        // base64("nocolon") = "bm9jb2xvbg=="
        assert_eq!(
            basic::Credentials::parse("Basic bm9jb2xvbg=="),
            Err(basic::HeaderError::MissingColon),
        );
    }

    #[test]
    fn parse_basic_auth_header_rejects_non_ascii_payload() {
        // base64("аlice:hunter2") with a Cyrillic 'а' (U+0430) as the first byte.
        assert_eq!(
            basic::Credentials::parse("Basic 0LBsaWNlOmh1bnRlcjI="),
            Err(basic::HeaderError::NonAsciiPayload),
        );
    }

    #[test]
    fn parse_basic_auth_header_rejects_non_utf8_payload() {
        // base64(0xff 0xfe 0xfd) = "//79". Not valid UTF-8; caught by the
        // ASCII gate before any string conversion happens.
        assert_eq!(
            basic::Credentials::parse("Basic //79"),
            Err(basic::HeaderError::NonAsciiPayload),
        );
    }

    #[test]
    fn parse_basic_auth_header_rejects_oversized_payload() {
        let oversized = format!("Basic {}", "A".repeat(basic::MAX_LEN));
        assert_eq!(
            basic::Credentials::parse(&oversized),
            Err(basic::HeaderError::PayloadTooLarge),
        );
    }

    #[test]
    fn constant_time_eq_returns_true_for_equal_bytes() {
        assert!(constant_time_eq(b"abcdef", b"abcdef"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn constant_time_eq_returns_false_for_different_bytes() {
        assert!(!constant_time_eq(b"abcdef", b"abcdeg"));
        assert!(!constant_time_eq(b"abcdef", b"xbcdef"));
    }

    #[test]
    fn constant_time_eq_returns_false_for_length_mismatch() {
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"abcd", b"abc"));
        assert!(!constant_time_eq(b"", b"a"));
    }

    #[test]
    fn cookie_entry_matches_its_own_user_and_token() {
        let path = tmp_cookie_path("creds_cookie");
        let (user, token) = generate_cookie(&path).expect("cookie write should succeed in test");
        let creds = Auth::new(vec![RpcAuth::from_password(&user, &token)]);

        assert!(creds.matches(&user, &token));
        assert!(!creds.matches(&user, "wrong"));
        assert!(!creds.matches("wronguser", &token));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn auth_matches_any_configured_entry() {
        let creds = Auth::new(vec![
            RpcAuth::from_password("alice", "hunter2"),
            RpcAuth::from_password("bob", "letmein"),
        ]);
        assert!(creds.matches("alice", "hunter2"));
        assert!(creds.matches("bob", "letmein"));
        assert!(!creds.matches("alice", "hunter3"));
        assert!(!creds.matches("bob", "hunter2"));
        assert!(!creds.matches("eve", "hunter2"));
    }

    #[test]
    fn rpcauth_from_password_round_trips_with_verify() {
        let auth = RpcAuth::from_password("alice", "hunter2");
        assert_eq!(auth.user, "alice");
        assert_eq!(
            auth.salt_hex.len(),
            32,
            "salt should be 16 bytes hex-encoded"
        );
        assert_eq!(
            auth.hash_hex.len(),
            64,
            "hash should be 32 bytes hex-encoded"
        );
        assert!(auth.verify("alice", "hunter2"));
        assert!(!auth.verify("alice", "wrong"));
        assert!(!auth.verify("bob", "hunter2"));
    }

    #[test]
    fn rpcauth_from_password_uses_fresh_salt_per_call() {
        let a = RpcAuth::from_password("alice", "hunter2");
        let b = RpcAuth::from_password("alice", "hunter2");
        assert_ne!(a.salt_hex, b.salt_hex);
        assert_ne!(a.hash_hex, b.hash_hex);
    }

    #[test]
    fn hmac_password_matches_rpcauth_py_vector() {
        // Generated with `python3 share/rpcauth/rpcauth.py alice` from
        // bitcoin/bitcoin master. Pins the salt-hex-as-ASCII-bytes interop hinge.
        let salt_hex = "cf2c6b493e4690de904306d1a82ef1cc";
        let pass = "Ys-WMXs6znL6q2BH9yjh77k9keytl4JpCshiapNCMyg";
        let expected = "9e0770d953ba59dcd5cf447615717764936d48b01d4428d251628f0156088901";
        assert_eq!(hmac_password(salt_hex, pass), expected);
    }

    #[test]
    fn rpcauth_parse_accepts_well_formed_line() {
        // Reuses the rpcauth.py vector pinned by hmac_password_matches_rpcauth_py_vector.
        let line = "alice:cf2c6b493e4690de904306d1a82ef1cc$9e0770d953ba59dcd5cf447615717764936d48b01d4428d251628f0156088901";
        let parsed = RpcAuth::parse(line).expect("rpcauth.py output should parse");
        assert_eq!(parsed.user, "alice");
        assert_eq!(parsed.salt_hex, "cf2c6b493e4690de904306d1a82ef1cc");
        assert_eq!(
            parsed.hash_hex,
            "9e0770d953ba59dcd5cf447615717764936d48b01d4428d251628f0156088901"
        );
    }

    #[test]
    fn rpcauth_parse_rejects_malformed_lines() {
        assert_eq!(
            RpcAuth::parse("nocolon$hash"),
            Err(RpcAuthParseError::MalformedLine)
        );
        assert_eq!(
            RpcAuth::parse("user:salt"),
            Err(RpcAuthParseError::MalformedLine)
        );
        assert_eq!(
            RpcAuth::parse("user:extra:s$h"),
            Err(RpcAuthParseError::MalformedLine)
        );
        assert_eq!(
            RpcAuth::parse("user:salt$h$extra"),
            Err(RpcAuthParseError::MalformedLine)
        );
    }

    #[test]
    fn rpcauth_parse_rejects_empty_field() {
        assert_eq!(
            RpcAuth::parse(":salt$abcdef"),
            Err(RpcAuthParseError::EmptyField)
        );
        assert_eq!(
            RpcAuth::parse("user:$abcdef"),
            Err(RpcAuthParseError::EmptyField)
        );
        assert_eq!(
            RpcAuth::parse("user:salt$"),
            Err(RpcAuthParseError::EmptyField)
        );
    }

    #[test]
    /// A non-ASCII user can never reach us through a `Basic` header, so the
    /// entry is rejected at parse time rather than silently never matching.
    fn rpcauth_parse_rejects_non_ascii_user() {
        assert_eq!(
            RpcAuth::parse("älice:cf2c6b$abcdef0123"),
            Err(RpcAuthParseError::NonAsciiUser)
        );
    }

    #[test]
    fn rpcauth_parse_is_case_sensitive_for_the_hash_only() {
        // An uppercase hash can never match the lowercase hex we compute.
        assert_eq!(
            RpcAuth::parse("user:cf2c6b$ABCDEF0123"),
            Err(RpcAuthParseError::NonLowercaseHash),
        );

        // An uppercase salt is fine: Core uses the salt string verbatim as the
        // HMAC key on both sides (httprpc.cpp:74), so rejecting it would
        // refuse a config bitcoind accepts. Verify a real login succeeds.
        let salt = "CF2C6B493E4690DE904306D1A82EF1CC";
        let pass = "Ys-WMXs6znL6q2BH9yjh77k9keytl4JpCshiapNCMyg";
        let entry = RpcAuth::parse(&format!("alice:{salt}${}", hmac_password(salt, pass)))
            .expect("uppercase salt is a valid entry");
        assert!(entry.verify("alice", pass));
    }

    #[cfg(unix)]
    #[test]
    fn generate_cookie_sets_owner_only_mode_on_unix() {
        use std::os::unix::fs::PermissionsExt;

        let path = tmp_cookie_path("perms");
        generate_cookie(&path).expect("cookie write should succeed in test");
        let mode = fs::metadata(&path)
            .expect("test wrote this cookie file above")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {mode:o}");

        fs::remove_file(&path).ok();
    }
}

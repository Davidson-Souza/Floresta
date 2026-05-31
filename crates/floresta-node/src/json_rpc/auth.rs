// SPDX-License-Identifier: MIT OR Apache-2.0

//! JSON-RPC authentication primitives, mirroring Bitcoin Core.
//!
//! [`generate_cookie`] writes `__cookie__:<64-char-hex>` to the caller's path,
//! rotating the token each restart. Clients then send
//! `Authorization: Basic <base64(__cookie__:<token>)>`.

use std::ffi::OsString;
use std::fs;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use bitcoin::hex::DisplayHex;

/// Username literal used for cookie auth (Core convention).
pub(crate) const COOKIE_USER: &str = "__cookie__";

/// Default cookie file name; placed under the net-specific datadir.
pub(crate) const COOKIE_FILE_NAME: &str = ".cookie";

/// Token length in raw random bytes; hex-encoded to 64 ASCII chars.
const COOKIE_TOKEN_BYTES: usize = 32;

/// Generate a fresh cookie and write it to `path`, overwriting any existing one.
///
/// Goes via `<path>.tmp`, unlinked first then opened with `create_new` so mode
/// `0600` always applies and a planted symlink is refused, then renamed.
pub(crate) fn generate_cookie(path: &Path) -> io::Result<()> {
    let token: [u8; COOKIE_TOKEN_BYTES] = rand::random();
    let auth = format!("{COOKIE_USER}:{}", token.to_lower_hex_string());

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
    file.write_all(auth.as_bytes())?;
    drop(file);

    fs::rename(&tmp, path)?;

    Ok(())
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
        generate_cookie(&path).expect("cookie write should succeed in test");

        let written = fs::read_to_string(&path).expect("test wrote this cookie file above");
        assert!(
            written.starts_with("__cookie__:"),
            "cookie file missing prefix: {written}"
        );
        let token = written
            .strip_prefix("__cookie__:")
            .expect("generated cookie always begins with __cookie__:");
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
        generate_cookie(&path1).expect("cookie write should succeed in test");
        generate_cookie(&path2).expect("cookie write should succeed in test");
        let auth1 = fs::read_to_string(&path1).expect("test wrote path1 above");
        let auth2 = fs::read_to_string(&path2).expect("test wrote path2 above");
        assert_ne!(
            auth1, auth2,
            "two consecutive calls produced identical tokens"
        );

        fs::remove_file(&path1).ok();
        fs::remove_file(&path2).ok();
    }

    #[test]
    fn generate_cookie_overwrites_existing_file() {
        let path = tmp_cookie_path("overwrite");
        fs::write(&path, "stale-content").expect("test temp path should be writable");
        generate_cookie(&path).expect("cookie write should succeed in test");
        let written = fs::read_to_string(&path).expect("test wrote this cookie file above");
        assert_ne!(written, "stale-content", "stale content was not replaced");
        assert!(
            written.starts_with("__cookie__:"),
            "replacement is not a cookie line: {written}"
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

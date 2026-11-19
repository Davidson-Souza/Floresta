// SPDX-License-Identifier: MIT OR Apache-2.0

//! This module has useful macros for usage on other crates.

#[macro_export]
/// Implements the `From` trait for converting a specific type into an enum variant or struct field.
///
/// This macro is useful when building error enums or wrapper types that need to support automatic
/// conversions from underlying error types using the `?` operator.
///
/// # Parameters
///
/// - `$thing`: The target type to implement `From` for (usually an enum or struct).
/// - `$from_thing`: The source type that should be converted into `$thing`.
/// - `$field`: The variant or field of `$thing` that wraps a value of type `$from_thing`.
///
/// # Example
///
/// ```rust
/// # use floresta_common::impl_error_from;
/// enum MyError {
///     Io(std::io::Error),
///     Parse(core::num::ParseIntError),
/// }
///
/// impl_error_from!(MyError, std::io::Error, Io);
/// impl_error_from!(MyError, core::num::ParseIntError, Parse);
///
/// fn parse_number(s: &str) -> Result<i32, MyError> {
///     let n: i32 = s.parse()?; // Automatically converts ParseIntError into MyError
///     Ok(n)
/// }
/// ```
macro_rules! impl_error_from {
    ($thing:ty, $from_thing:ty, $field:ident) => {
        impl From<$from_thing> for $thing {
            fn from(e: $from_thing) -> Self {
                <$thing>::$field(e)
            }
        }
    };
}

#[macro_export]
/// Panic if the expression is not `Ok(_)`.
///
/// # Examples
///
/// ```rust
/// # use floresta_common::assert_ok;
/// // Successful
/// assert_ok!(Ok::<u32, &str>(2025));
/// ```
///
/// ```rust,should_panic
/// # use floresta_common::assert_ok;
/// // Panics
/// assert_ok!(Err::<u32, &str>("failed"));
/// ```
///
/// ```rust,compile_fail
/// # use floresta_common::assert_ok;
/// // Compile error: `assert_ok!` requires a `Result` value
/// assert_ok!(Some(42));
/// ```
macro_rules! assert_ok {
    ($expr:expr $(,)?) => {
        if let Err(e) = $expr {
            panic!("assertion failed: expected `Ok(_)`, got `Err({:?})`", e);
        }
    };
}

#[macro_export]
/// Panic if the expression is not `Err(_)`.
///
/// # Examples
///
/// ```rust
/// # use floresta_common::assert_err;
/// // Successful
/// assert_err!(Err::<u32, &str>("failed"));
/// ```
///
/// ```rust,should_panic
/// # use floresta_common::assert_err;
/// // Panics
/// assert_err!(Ok::<u32, &str>(2025));
/// ```
///
/// ```rust,compile_fail
/// # use floresta_common::assert_err;
/// // Compile error: `assert_err!` requires a `Result` value
/// assert_err!(Some(42));
/// ```
macro_rules! assert_err {
    ($expr:expr $(,)?) => {
        if let Ok(v) = $expr {
            panic!("assertion failed: expected `Err(_)`, got `Ok({:?})`", v);
        }
    };
}

#[macro_export]
/// Validates a block hash at compile time. Requires `BlockHash` to be in scope.
macro_rules! bhash {
    ($s:literal) => {{
        // Catch invalid literals at compile time.
        const _: () = match $crate::macros::validate_hash_compile_time($s) {
            Ok(()) => (),
            Err(e) => panic!("{}", e),
        };
        $s.parse::<BlockHash>().expect("Literal should be valid")
    }};
}

#[macro_export]
/// Validates Utreexo node hashes at compile time. Requires `BitcoinNodeHash` to be in scope.
///
/// - Accepts one or more comma-separated hash literals.
/// - Allows an optional trailing comma.
macro_rules! acchashes {
    ( $( $s:literal ),+ $(,)? ) => {
        [ $( {
            // Catch invalid literals at compile time.
            const _: () = match $crate::macros::validate_hash_compile_time($s) {
                Ok(()) => (),
                Err(e) => panic!("{}", e),
            };
            $s.parse::<BitcoinNodeHash>().expect("Literal should be valid")
        } ),+ ]
    };
}

#[doc(hidden)]
// This const function is used to validate hash literals at compile time
pub const fn validate_hash_compile_time(s: &str) -> Result<(), &str> {
    let bytes = s.as_bytes();

    // Note: An ASCII character is 1 byte, so the expected byte count is 64
    if bytes.len() != 64 {
        return Err("Hash literal is not exactly 64 hex digits");
    }

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if !((b >= b'0' && b <= b'9') || (b >= b'a' && b <= b'f') || (b >= b'A' && b <= b'F')) {
            return Err("Hash literal contains an invalid ASCII hex digit");
        }
        i += 1;
    }

    Ok(())
}

#[macro_export]
/// Run a task and log any errors that might occur.
///
/// The error is logged with `tracing::error!`, which is resolved in the calling crate, so that
/// crate must depend on `tracing`.
macro_rules! try_and_log {
    ($what:expr) => {
        if let Err(error) = $what {
            tracing::error!("{}: {} - {:?}", line!(), file!(), error);
        }
    };
}

#[macro_export]
/// Run a task and warn any errors that might occur.
///
/// `try_and_log!` variant for tasks that can fail safely.
///
/// The error is logged with `tracing::warn!`, which is resolved in the calling crate, so that
/// crate must depend on `tracing`.
macro_rules! try_and_warn {
    ($what:expr) => {
        if let Err(warning) = $what {
            tracing::warn!("{}", warning);
        }
    };
}

#[macro_export]
/// Run a task and log any errors that might occur at debug level.
///
/// `try_and_log!` variant for failures that are expected and benign, with an
/// extra context message so the call site can say what was being attempted.
///
/// The error is logged with `tracing::debug!`, which is resolved in the calling crate, so that
/// crate must depend on `tracing`.
macro_rules! try_and_debug {
    ($what:expr, $ctx:expr) => {
        if let Err(error) = $what {
            tracing::debug!("{}: {} - {}: {:?}", line!(), file!(), $ctx, error);
        }
    };
}

#[cfg(test)]
mod test {
    use super::validate_hash_compile_time as validate_hash;

    // Submodule to test the behavior of the `assert_ok` and `assert_err` macros
    mod test_assert_ok_err {
        fn ok_fn() -> Result<u32, &'static str> {
            Ok(778)
        }

        fn err_fn() -> Result<u32, &'static str> {
            Err("failure")
        }

        #[test]
        fn test_assert_ok_err_pass() {
            assert_ok!(ok_fn());
            assert_err!(err_fn());
        }

        #[test]
        #[should_panic(expected = "assertion failed: expected `Ok(_)`, got `Err(\"failure\")`")]
        fn test_assert_ok_panics_on_err() {
            // Should panic with our message
            assert_ok!(err_fn());
        }

        #[test]
        #[should_panic(expected = "assertion failed: expected `Err(_)`, got `Ok(778)`")]
        fn test_assert_err_panics_on_ok() {
            // Should panic with our message
            assert_err!(ok_fn());
        }
    }

    #[test]
    fn test_validate_hash_compile_time() {
        // Valid: exactly 64 ASCII hex digits.
        let valid = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_ok!(validate_hash(valid));

        for len in 0..=128 {
            let test_str = "a".repeat(len);
            if len == 64 {
                assert_ok!(validate_hash(&test_str));
            } else {
                assert_err!(validate_hash(&test_str));
            }
        }

        // Invalid hex character at the end: 'g'
        let invalid = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg";
        assert_eq!(invalid.len(), 64);
        assert_err!(validate_hash(invalid));

        // Invalid ASCII character in the middle: 'é'
        let invalid_ascii = "0123456789abcdef0123456789abcdéf0123456789abcdef0123456789abcde";
        assert_eq!(invalid_ascii.len(), 64);
        assert_err!(validate_hash(invalid_ascii));
    }
}

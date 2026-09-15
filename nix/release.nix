# SPDX-License-Identifier: MIT OR Apache-2.0
{
  fenix,
  macos-sdk,
  nixpkgs,
  self,
  system,
}:

let
  pkgs = import nixpkgs {
    inherit system;
    config.allowUnfree = true;
  };
  inherit (pkgs) lib;
  manifest = builtins.fromTOML (builtins.readFile ../bin/florestad/Cargo.toml);
  version = manifest.package.version;
  sourceDateEpoch = toString (self.lastModified or 1);

  targets = {
    linux-x86_64 = {
      packageSet = pkgs.pkgsCross.musl64;
      rustTarget = "x86_64-unknown-linux-musl";
      archive = "tar.gz";
      signable = false;
    };
    linux-aarch64 = {
      packageSet = pkgs.pkgsCross.aarch64-multiplatform-musl;
      rustTarget = "aarch64-unknown-linux-musl";
      archive = "tar.gz";
      signable = false;
    };
    linux-riscv64 = {
      packageSet = pkgs.pkgsCross.riscv64-musl;
      rustTarget = "riscv64gc-unknown-linux-musl";
      archive = "tar.gz";
      signable = false;
    };
    windows-x86_64 = {
      packageSet = pkgs.pkgsCross.mingwW64;
      rustTarget = "x86_64-pc-windows-gnu";
      archive = "zip";
      signable = true;
    };
    windows-aarch64 = {
      packageSet = pkgs;
      rustTarget = "aarch64-pc-windows-gnullvm";
      archive = "zip";
      signable = true;
    };
    macos-x86_64 = {
      packageSet = pkgs;
      rustTarget = "x86_64-apple-darwin";
      archive = "tar.gz";
      signable = true;
    };
    macos-aarch64 = {
      packageSet = pkgs;
      rustTarget = "aarch64-apple-darwin";
      archive = "tar.gz";
      signable = true;
    };
  };

  rustSource = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../README.md
      ../LICENSE-APACHE
      ../LICENSE-MIT
      ../LICENSE.md
      ../bin
      ../crates
      ../doc/rpc
      ../fuzz
    ];
  };

  fenixPackages = fenix.packages.${system};
  fenixStableManifest = builtins.fromJSON (builtins.readFile "${fenix.outPath}/data/stable.json");
  fenixRustcUrl = fenixStableManifest.pkg.rustc.target.x86_64-unknown-linux-gnu.url;
  fenixRustcVersionMatch = builtins.match ".*/rustc-([0-9]+\\.[0-9]+\\.[0-9]+)-.*" fenixRustcUrl;
  fenixRustcVersion =
    if fenixRustcVersionMatch == null then
      throw "could not determine the pinned Fenix rustc version"
    else
      builtins.elemAt fenixRustcVersionMatch 0;
  darwinRustToolchain = fenixPackages.combine [
    fenixPackages.stable.cargo
    fenixPackages.stable.rustc
    fenixPackages.targets.aarch64-apple-darwin.stable.rust-std
    fenixPackages.targets.x86_64-apple-darwin.stable.rust-std
  ];
  windowsArmRustToolchain = fenixPackages.combine [
    fenixPackages.stable.cargo
    fenixPackages.stable.rustc
    fenixPackages.targets.aarch64-pc-windows-gnullvm.stable.rust-std
  ];
  mkBinaries =
    targetName: target:
    let
      baseCargoVendor = pkgs.rustPlatform.importCargoLock {
        lockFile = ../Cargo.lock;
      };
      isMusl = lib.hasSuffix "-linux-musl" target.rustTarget;
      isWindows = lib.hasInfix "-windows-" target.rustTarget;
      isDarwin = lib.hasSuffix "-apple-darwin" target.rustTarget;
      isWindowsArm = target.rustTarget == "aarch64-pc-windows-gnullvm";
      rustPlatform =
        if isDarwin || isWindowsArm then pkgs.rustPlatform else target.packageSet.rustPlatform;
      darwinEnvTarget = builtins.replaceStrings [ "-" ] [ "_" ] target.rustTarget;
      darwinCargoTarget = lib.toUpper darwinEnvTarget;
      darwinArch = if target.rustTarget == "aarch64-apple-darwin" then "arm64" else "x86_64";
      darwinZigArch = if target.rustTarget == "aarch64-apple-darwin" then "aarch64" else "x86_64";
      darwinZigTarget = "${darwinZigArch}-macos.11.0-none";
      darwinClangTarget = "${darwinArch}-apple-macos11";
      darwinCxxRuntime = pkgs.runCommand "floresta-${targetName}-libcxx" { } ''
        export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global"
        export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local"
        printf '%s\n' 'int main() { return 0; }' > probe.cpp
        ${pkgs.zig}/bin/zig c++ \
          -target ${darwinZigTarget} \
          -Wno-nullability-completeness \
          -o probe \
          probe.cpp
        mkdir -p "$out/lib"
        libcxx="$(find "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR" -type f -name 'libc++.a' -print -quit)"
        libcxxabi="$(find "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR" -type f -name 'libc++abi.a' -print -quit)"
        compiler_rt="$(find "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR" -type f -name 'libcompiler_rt.a' -print -quit)"
        ubsan_rt="$(find "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR" -type f -name 'libubsan_rt.a' -print -quit)"
        test -n "$libcxx"
        test -n "$libcxxabi"
        test -n "$compiler_rt"
        test -n "$ubsan_rt"
        install -m 0644 "$libcxx" "$out/lib/libc++.a"
        install -m 0644 "$libcxxabi" "$out/lib/libc++abi.a"
        install -m 0644 "$compiler_rt" "$out/lib/libcompiler_rt.a"
        install -m 0644 "$ubsan_rt" "$out/lib/libubsan_rt.a"
      '';
      windowsArmCxxRuntime = pkgs.runCommand "floresta-${targetName}-libcxx" { } ''
        export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global"
        export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local"
        printf '%s\n' 'int main() { return 0; }' > probe.cpp
        ${pkgs.zig}/bin/zig c++ \
          -target aarch64-windows-gnu \
          -O2 \
          -g0 \
          -Wno-nullability-completeness \
          -o probe.exe \
          probe.cpp
        mkdir -p "$out/lib"
        libcxx="$(find "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR" -type f -name 'c++.lib' -print -quit)"
        libcxxabi="$(find "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR" -type f -name 'c++abi.lib' -print -quit)"
        test -n "$libcxx"
        test -n "$libcxxabi"
        install -m 0644 "$libcxx" "$out/lib/c++.lib"
        install -m 0644 "$libcxxabi" "$out/lib/c++abi.lib"
      '';
      darwinCc = pkgs.writeShellScript "floresta-${targetName}-cc" ''
        args=()
        skip_next=false
        for arg in "$@"; do
          if "$skip_next"; then
            skip_next=false
            continue
          fi
          case "$arg" in
            -arch | -target | --target) skip_next=true ;;
            -target=* | --target=* | -mmacosx-version-min=*) ;;
            *) args+=("$arg") ;;
          esac
        done
        exec ${pkgs.zig}/bin/zig cc \
          -target ${darwinZigTarget} \
          -isysroot ${macos-sdk} \
          -isystem ${macos-sdk}/usr/include \
          -iframework ${macos-sdk}/System/Library/Frameworks \
          -Wno-unknown-warning-option \
          "''${args[@]}"
      '';
      darwinCxx = pkgs.writeShellScript "floresta-${targetName}-cxx" ''
        args=()
        skip_next=false
        for arg in "$@"; do
          if "$skip_next"; then
            skip_next=false
            continue
          fi
          case "$arg" in
            -arch | -target | --target) skip_next=true ;;
            -target=* | --target=* | -mmacosx-version-min=*) ;;
            *) args+=("$arg") ;;
          esac
        done
        exec ${pkgs.zig}/bin/zig c++ \
          -target ${darwinZigTarget} \
          -isysroot ${macos-sdk} \
          -isystem ${macos-sdk}/usr/include \
          -iframework ${macos-sdk}/System/Library/Frameworks \
          -Wno-unknown-warning-option \
          "''${args[@]}"
      '';
      darwinLinker = pkgs.writeShellScript "floresta-${targetName}-linker" ''
        exec ${pkgs.llvmPackages.clang-unwrapped}/bin/clang \
          --target=${darwinClangTarget} \
          -isysroot ${macos-sdk} \
          --ld-path=${pkgs.llvmPackages.lld}/bin/ld64.lld \
          -Wl,-platform_version,macos,11.0,11.3 \
          "$@"
      '';
      darwinCmakeToolchain = pkgs.writeText "floresta-${targetName}-toolchain.cmake" ''
        set(CMAKE_SYSTEM_NAME Darwin)
        set(CMAKE_SYSTEM_PROCESSOR ${darwinArch})
        set(CMAKE_C_COMPILER "${darwinCc}")
        set(CMAKE_CXX_COMPILER "${darwinCxx}")
        set(CMAKE_AR "${pkgs.llvmPackages.llvm}/bin/llvm-ar")
        set(CMAKE_RANLIB "${pkgs.llvmPackages.llvm}/bin/llvm-ranlib")
        set(CMAKE_OSX_ARCHITECTURES "${darwinArch}")
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
        set(CMAKE_OSX_SYSROOT "${macos-sdk}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
      '';
      mingwThreads =
        if target.rustTarget == "x86_64-pc-windows-gnu" then
          target.packageSet.windows.mcfgthreads
        else
          null;
      explicitCxxLibraries = if target.rustTarget == "x86_64-pc-windows-gnu" then [ "stdc++" ] else [ ];
      cxxSearchLibrary =
        if isMusl then
          "stdc++"
        else if explicitCxxLibraries != [ ] then
          builtins.head explicitCxxLibraries
        else
          null;
      # Keep vendored source immutable except for cross-compilation corrections
      # that libbitcoinkernel-sys 0.4.0 does not currently expose as options.
      cargoVendor =
        if isMusl || isWindows || isDarwin then
          pkgs.runCommand "cargo-vendor-dir" { } ''
            cp --recursive --dereference ${baseCargoVendor} "$out"
            chmod --recursive u+w "$out"
            ${lib.optionalString isMusl ''
              substituteInPlace "$out/libbitcoinkernel-sys-0.4.0/build.rs" \
                --replace-fail 'rustc-link-lib=dylib={lib}' 'rustc-link-lib=static={lib}'
            ''}
            ${lib.optionalString isWindows ''
              substituteInPlace "$out/libbitcoinkernel-sys-0.4.0/build.rs" \
                --replace-fail '"windows" => TargetConfig {' \
                '"windows" => TargetConfig { cmake_args: vec!["-DCMAKE_SYSTEM_NAME=Windows".into(), "-DCMAKE_C_FLAGS=-UNDEBUG".into(), "-DCMAKE_CXX_FLAGS=-UNDEBUG".into(), "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -g".into(), "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g".into()],'
            ''}
            ${lib.optionalString isWindowsArm ''
              substituteInPlace "$out/libbitcoinkernel-sys-0.4.0/build.rs" \
                --replace-fail 'let target = target_config();' \
                'let mut target = target_config(); let cmake_target = env::var("TARGET").unwrap().replace("-", "_"); if let Ok(toolchain) = env::var(format!("CMAKE_TOOLCHAIN_FILE_{cmake_target}")) { target.cmake_args.push(format!("-DCMAKE_TOOLCHAIN_FILE={toolchain}")); }'
            ''}
            ${lib.optionalString isDarwin ''
              substituteInPlace "$out/libbitcoinkernel-sys-0.4.0/build.rs" \
                --replace-fail '"macos" | "ios" | "tvos" | "watchos" | "visionos" => TargetConfig {' \
                '"macos" | "ios" | "tvos" | "watchos" | "visionos" => TargetConfig { cmake_args: vec!["-DCMAKE_SYSTEM_NAME=Darwin".into(), "-DCMAKE_C_FLAGS=-UNDEBUG".into(), "-DCMAKE_CXX_FLAGS=-UNDEBUG".into(), "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -g".into(), "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g".into()],'
              substituteInPlace "$out/libbitcoinkernel-sys-0.4.0/build.rs" \
                --replace-fail 'let target = target_config();' \
                'let mut target = target_config(); let cmake_target = env::var("TARGET").unwrap().replace("-", "_"); if let Ok(toolchain) = env::var(format!("CMAKE_TOOLCHAIN_FILE_{cmake_target}")) { target.cmake_args.push(format!("-DCMAKE_TOOLCHAIN_FILE={toolchain}")); }'
              substituteInPlace "$out/libbitcoinkernel-sys-0.4.0/build.rs" \
                --replace-fail 'link_directives: cxx_runtime("c++"),' \
                'link_directives: vec!["rustc-link-lib=static=c++".into(), "rustc-link-lib=static=c++abi".into()],'
            ''}
          ''
        else
          baseCargoVendor;
    in
    rustPlatform.buildRustPackage (
      {
        pname = "floresta-binaries-${targetName}";
        inherit version;
        src = rustSource;

        cargoDeps = cargoVendor;
        cargoBuildFlags = [
          "--package=florestad"
          "--package=floresta-cli"
        ];
        doCheck = false;
        strictDeps = true;

        nativeBuildInputs =
          (with pkgs; [
            boost
            cmake
            perl
            pkg-config
          ])
          ++ lib.optionals (target.rustTarget == "x86_64-pc-windows-gnu") [ pkgs.nasm ]
          ++ lib.optionals isWindowsArm [
            pkgs.cargo-zigbuild
            pkgs.zig
          ];
        buildInputs = lib.optional (mingwThreads != null) mingwThreads;

        SOURCE_DATE_EPOCH = sourceDateEpoch;
        CMAKE_PREFIX_PATH = "${pkgs.boost.dev}";
        CARGO_INCREMENTAL = "0";
        preBuild = ''
          export RUSTFLAGS="''${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$NIX_BUILD_TOP=.${lib.optionalString isMusl " -C target-feature=+crt-static"}${
            lib.concatMapStrings (library: " -l static=${library}") explicitCxxLibraries
          }${
            lib.optionalString (
              mingwThreads != null
            ) " -L native=${mingwThreads}/lib -l static=mcfgthread -l dylib=ntdll"
          }${lib.optionalString isWindowsArm " -C link-arg=${windowsArmCxxRuntime}/lib/c++abi.lib -C link-arg=${windowsArmCxxRuntime}/lib/c++.lib"}${lib.optionalString isDarwin " -L native=${darwinCxxRuntime}/lib -C link-arg=-Wl,-no_fixup_chains -C link-arg=${darwinCxxRuntime}/lib/libubsan_rt.a -C link-arg=${darwinCxxRuntime}/lib/libcompiler_rt.a -C link-arg=${darwinCxxRuntime}/lib/libc++.a -C link-arg=${darwinCxxRuntime}/lib/libc++abi.a"}"
          ${lib.optionalString (cxxSearchLibrary != null) ''
            cxx_runtime="$("$CXX" -print-file-name=lib${cxxSearchLibrary}.a)"
            test -f "$cxx_runtime"
            export RUSTFLAGS="$RUSTFLAGS -L native=$(dirname "$cxx_runtime")"
          ''}
        '';
        postInstall = ''
          test -x "$out/bin/florestad${lib.optionalString isWindows ".exe"}"
          test -x "$out/bin/floresta-cli${lib.optionalString isWindows ".exe"}"
        '';

        meta = {
          description = "Deterministically cross-built Floresta binaries for ${target.rustTarget}";
        };
      }
      // lib.optionalAttrs isWindowsArm {
        cargo = windowsArmRustToolchain;
        rustc = windowsArmRustToolchain;
        dontStrip = true;

        buildPhase = ''
          runHook preBuild
          export PATH="${windowsArmRustToolchain}/bin:$PATH"
          export RUSTC="${windowsArmRustToolchain}/bin/rustc"
          export CARGO_ZIGBUILD_CACHE_DIR="$TMPDIR/cargo-zigbuild"
          export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global"
          export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local"
          mkdir -p "$CARGO_ZIGBUILD_CACHE_DIR" "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR"
          "${windowsArmRustToolchain}/bin/cargo" zigbuild \
            --jobs "$NIX_BUILD_CORES" \
            --target "${target.rustTarget}" \
            --offline \
            --release \
            --package florestad \
            --package floresta-cli
          runHook postBuild
        '';

        installPhase = ''
          runHook preInstall
          mkdir -p "$out/bin"
          install -m 0755 "target/${target.rustTarget}/release/florestad.exe" "$out/bin/"
          install -m 0755 "target/${target.rustTarget}/release/floresta-cli.exe" "$out/bin/"
          runHook postInstall
        '';
      }
      // lib.optionalAttrs isDarwin {
        cargo = darwinRustToolchain;
        rustc = darwinRustToolchain;
        SDKROOT = "${macos-sdk}";
        MACOSX_DEPLOYMENT_TARGET = "11.0";
        dontStrip = true;

        buildPhase = ''
          runHook preBuild
          export PATH="${darwinRustToolchain}/bin:$PATH"
          export RUSTC="${darwinRustToolchain}/bin/rustc"
          export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global"
          export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zig-local"
          mkdir -p "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR"
          export CARGO_TARGET_${darwinCargoTarget}_LINKER="${darwinLinker}"
          export CC_${darwinEnvTarget}="${darwinCc}"
          export CXX_${darwinEnvTarget}="${darwinCxx}"
          export AR_${darwinEnvTarget}="${pkgs.llvmPackages.llvm}/bin/llvm-ar"
          export RANLIB_${darwinEnvTarget}="${pkgs.llvmPackages.llvm}/bin/llvm-ranlib"
          export CMAKE_TOOLCHAIN_FILE_${darwinEnvTarget}="${darwinCmakeToolchain}"
          "${darwinRustToolchain}/bin/cargo" build \
            --jobs "$NIX_BUILD_CORES" \
            --target "${target.rustTarget}" \
            --offline \
            --release \
            --package florestad \
            --package floresta-cli
          runHook postBuild
        '';

        installPhase = ''
          runHook preInstall
          mkdir -p "$out/bin"
          install -m 0755 "target/${target.rustTarget}/release/florestad" "$out/bin/"
          install -m 0755 "target/${target.rustTarget}/release/floresta-cli" "$out/bin/"
          runHook postInstall
        '';
      }
    );

  mkRelease =
    targetName: target:
    let
      binaries = mkBinaries targetName target;
      isWindows = lib.hasPrefix "windows-" targetName;
      executableSuffix = lib.optionalString isWindows ".exe";
      archiveBase = "floresta-${version}-${targetName}";
      archiveName = "${archiveBase}${lib.optionalString target.signable "-unsigned"}.${target.archive}";
      buildInfoName = "${archiveBase}-buildinfo.json";
      buildInfo = builtins.toJSON {
        inherit sourceDateEpoch targetName version;
        nixpkgsRevision = nixpkgs.rev or "unknown";
        inherit (target) rustTarget;
        rustcVersion =
          if
            lib.hasSuffix "-apple-darwin" target.rustTarget || target.rustTarget == "aarch64-pc-windows-gnullvm"
          then
            fenixRustcVersion
          else
            target.packageSet.rustc.version;
        sourceRevision = self.rev or self.dirtyRev or "dirty";
      };
    in
    pkgs.runCommand "${archiveBase}-release"
      {
        nativeBuildInputs = with pkgs; [
          binutils
          coreutils
          findutils
          gnugrep
          gnutar
          gzip
          zip
        ];
        SOURCE_DATE_EPOCH = sourceDateEpoch;
      }
      ''
        set -o errexit -o nounset -o pipefail
        export LC_ALL=C
        export TZ=UTC
        umask 0022

        stage="$TMPDIR/stage"
        release_root="$stage/${archiveBase}"
        mkdir -p "$release_root/bin" "$out"
        install -m 0755 "${binaries}/bin/florestad${executableSuffix}" "$release_root/bin/"
        install -m 0755 "${binaries}/bin/floresta-cli${executableSuffix}" "$release_root/bin/"
        for executable in "$release_root/bin/"*; do
          if grep --text --fixed-strings --quiet '/nix/store/' "$executable"; then
            echo "Nix store reference found in $executable" >&2
            exit 1
          fi
        done
        ${lib.optionalString (lib.hasPrefix "linux-" targetName) ''
          for executable in "$release_root/bin/"*; do
            if readelf --program-headers --wide "$executable" | grep --quiet ' INTERP '; then
              echo "Dynamic interpreter found in $executable" >&2
              exit 1
            fi
          done
        ''}
        find "$release_root" -print0 | xargs -0r touch --no-dereference --date="@$SOURCE_DATE_EPOCH"

        ${
          if target.archive == "zip" then
            ''
              (
                cd "$stage"
                find "${archiveBase}" -print | sort | zip -X -9 "$out/${archiveName}" -@
              )
            ''
          else
            ''
              tar --create \
                  --sort=name \
                  --format=ustar \
                  --owner=0 \
                  --group=0 \
                  --numeric-owner \
                  --mtime="@$SOURCE_DATE_EPOCH" \
                  --mode='u+rw,go+r-w,a+X' \
                  --directory="$stage" \
                  "${archiveBase}" \
                | gzip -9n > "$out/${archiveName}"
            ''
        }

        ${lib.optionalString target.signable ''
          signing_stage="$TMPDIR/signing"
          signing_root="$signing_stage/${archiveBase}-codesigning"
          mkdir -p "$signing_root/unsigned"
          cp -R "$release_root" "$signing_root/unsigned/"
          printf '%s\n' '${targetName}' > "$signing_root/target"
          install -m 0755 ${../contrib/reproducible/detached-sig-create} "$signing_root/detached-sig-create"
          find "$signing_root" -print0 | xargs -0r touch --no-dereference --date="@$SOURCE_DATE_EPOCH"
          tar --create \
              --sort=name \
              --format=ustar \
              --owner=0 \
              --group=0 \
              --numeric-owner \
              --mtime="@$SOURCE_DATE_EPOCH" \
              --mode='u+rw,go+r-w,a+X' \
              --directory="$signing_stage" \
              "${archiveBase}-codesigning" \
            | gzip -9n > "$out/${archiveBase}-codesigning.tar.gz"
        ''}
        printf '%s\n' ${lib.escapeShellArg buildInfo} > "$out/${buildInfoName}"

        (
          cd "$out"
          sha256sum floresta-* | sort -k2 > SHA256SUMS.part
        )
      '';
in
{
  inherit targets version sourceDateEpoch;
  packages = lib.mapAttrs' (
    targetName: target: lib.nameValuePair "release-${targetName}" (mkRelease targetName target)
  ) targets;
}

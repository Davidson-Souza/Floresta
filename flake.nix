{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    macos-sdk = {
      url = "tarball+https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX11.3.sdk.tar.xz";
      flake = false;
    };
    signapple-src = {
      url = "github:achow101/signapple/3fab3bb57f227f0dd31007b417683035f5204838";
      flake = false;
    };
    certvalidator-src = {
      url = "github:achow101/certvalidator/a145bf25eb75a9f014b3e7678826132efbba6213";
      flake = false;
    };
    elfesteem-src = {
      url = "github:LRGH/elfesteem/2eb1e5384ff7a220fd1afacd4a0170acff54fe56";
      flake = false;
    };
  };

  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      treefmt-nix,
      fenix,
      macos-sdk,
      signapple-src,
      certvalidator-src,
      elfesteem-src,
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ treefmt-nix.flakeModule ];

      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        {
          system,
          pkgs,
          ...
        }:
        let
          release =
            if system == "x86_64-linux" then
              import ./nix/release.nix {
                inherit
                  fenix
                  macos-sdk
                  nixpkgs
                  self
                  system
                  ;
              }
            else
              null;
          signapple = import ./nix/signapple.nix {
            inherit
              certvalidator-src
              elfesteem-src
              pkgs
              signapple-src
              ;
          };
          codesignPackages = with pkgs; [
            bashInteractive
            cacert
            coreutils
            findutils
            gnupg
            gnutar
            gzip
            openssl
            osslsigncode
            signapple
            zip
          ];
        in
        {
          packages = pkgs.lib.optionalAttrs (system == "x86_64-linux") (
            release.packages
            // {
              inherit signapple;
              codesign-tools = pkgs.symlinkJoin {
                name = "floresta-codesign-tools";
                paths = codesignPackages;
              };
            }
          );

          treefmt = {
            projectRootFile = "flake.nix";
            programs.nixfmt.enable = true;
            programs.statix.enable = true;
          };

          devShells = {
            default =
              let
                packages = with pkgs; [
                  just
                  rustup
                  git
                  boost
                  cmake
                  typos
                  python312
                  uv
                  go
                  cargo-hack
                  pkg-config
                  openssl
                  llvmPackages.clang
                ];
              in
              pkgs.mkShell {
                inherit packages;
                LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
                CMAKE_PREFIX_PATH = "${pkgs.boost.dev}";
              };
          }
          // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
            codesign = pkgs.mkShell {
              packages = codesignPackages;
            };
          };
        };
    };
}

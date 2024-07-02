{ lib, rustPlatform, rust, buildInputs, nativeBuildInputs, ... }:

let
  pname = "florestad";
  version = "0.5.1";

  buildRustPackage = rustPlatform.buildRustPackage.override {
    rustc = rust;
    cargo = rust;
  };
in

buildRustPackage {
  inherit pname version;

  doCheck = false;

  src = ./.;

  cargoLock = {
    lockFile = ./Cargo.lock;
    outputHashes = {
      "bitcoin-0.31.0" = "sha256-2v87cb+Wd4OhgDG45Za3bxumaY7QKdw7nKuqKrmtHMs=";
    };
  };

  inherit buildInputs nativeBuildInputs;

  # Override the Rust compiler used
  rustc = "${rust}/bin/rustc";
  cargo = "${rust}/bin/cargo";

  meta = with lib; {
    description = "A full bitcoin node with Utreexo";
    homepage = "https://github.com/Davidson-Souza/Floresta";
    license = licenses.mit;
    maintainers = [ maintainers.Davidson maintainers.afm maintainers.jaoleal ];
    platforms = platforms.all;
  };
}
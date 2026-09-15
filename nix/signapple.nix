# SPDX-License-Identifier: MIT OR Apache-2.0
{
  certvalidator-src,
  elfesteem-src,
  pkgs,
  signapple-src,
}:

let
  python = pkgs.python3Packages;

  elfesteem = python.buildPythonPackage {
    pname = "elfesteem";
    version = "0.1-2eb1e53";
    src = elfesteem-src;
    format = "setuptools";
    # LLVM-produced bind streams exercise a signed LEB128 encoder bug in the
    # Bitcoin Core-pinned Elfesteem revision.
    patches = [ ./elfesteem-sleb128.patch ];
    doCheck = false;
  };

  certvalidator = python.buildPythonPackage {
    pname = "certvalidator";
    version = "0.1-a145bf2";
    src = certvalidator-src;
    format = "setuptools";
    dependencies = with python; [
      asn1crypto
      oscrypto
    ];
    doCheck = false;
  };
in
python.buildPythonApplication {
  pname = "signapple";
  version = "0.2.0-3fab3bb";
  src = signapple-src;
  pyproject = true;

  build-system = [ python.poetry-core ];
  dependencies = with python; [
    asn1crypto
    oscrypto
    certvalidator
    elfesteem
  ];
  doCheck = false;
  pythonImportsCheck = [ "signapple" ];

  makeWrapperArgs = [
    "--set"
    "SIGNAPPLE_OSCRYPTO_SSL_PATHS"
    "${pkgs.openssl.out}/lib/libcrypto.so,${pkgs.openssl.out}/lib/libssl.so"
  ];

  meta = {
    description = "Bitcoin Core's tool for detached macOS code signatures";
    homepage = "https://github.com/achow101/signapple";
    license = pkgs.lib.licenses.mit;
    mainProgram = "signapple";
  };
}

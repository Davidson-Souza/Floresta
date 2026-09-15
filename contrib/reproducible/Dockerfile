# syntax=docker/dockerfile:1
FROM nixos/nix:2.31.2@sha256:29fc5fe207f159ceb0143c25c19c774062fee02ce5eda118f3067547b3054894

ENV NIX_CONFIG="experimental-features = nix-command flakes"
WORKDIR /src
COPY . /src

ENTRYPOINT ["/src/contrib/reproducible/nix-build"]

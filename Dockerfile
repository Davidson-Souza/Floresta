FROM rust:1.74.0@sha256:fd45a543ed41160eae2ce9e749e5b3c972625b0778104e8962e9bfb113535301 as builder

WORKDIR /opt/app

COPY Cargo.* ./
COPY florestad/ florestad/
COPY crates/ crates/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
  cargo build --release

FROM debian:11.6-slim@sha256:171530d298096f0697da36b3324182e872db77c66452b85783ea893680cc1b62

COPY --from=builder /opt/app/target/release/florestad /usr/local/bin/florestad
RUN chmod +x /usr/local/bin/florestad

EXPOSE 50001

CMD [ "florestad", "run"]

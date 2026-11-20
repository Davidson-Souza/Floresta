FROM debian:13.2-slim@sha256:18764e98673c3baf1a6f8d960b5b5a1ec69092049522abac4e24a7726425b016 AS builder

ARG BUILD_FEATURES=""

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    clang \
    libclang-dev \
    git \
    libssl-dev \
    pkg-config \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default 1.85.0

WORKDIR /opt/app

COPY README.md .
COPY Cargo.* ./
COPY bin/ bin/
COPY crates/ crates/
COPY fuzz/ fuzz/
COPY doc/ doc/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    echo "Building florestad..." && \
    if [ -n "$BUILD_FEATURES" ]; then \
        cargo build --release -p florestad --features "$BUILD_FEATURES"; \
    else \
        cargo build --release -p florestad; \
    fi && \
    echo "Building floresta-cli..." && \
    cargo build --release -p floresta-cli

FROM debian:13.2-slim@sha256:18764e98673c3baf1a6f8d960b5b5a1ec69092049522abac4e24a7726425b016

COPY --from=builder /opt/app/target/release/florestad /usr/local/bin/florestad
COPY --from=builder /opt/app/target/release/floresta-cli /usr/local/bin/floresta-cli
RUN chmod +x /usr/local/bin/florestad

EXPOSE 50001
EXPOSE 8332
EXPOSE 3333

LABEL org.opencontainers.image.description="A lightweight and embeddable Bitcoin client, built for sovereignty!"
LABEL org.opencontainers.image.documentation="https://raw.githubusercontent.com/getfloresta/Floresta/refs/heads/master/README.md"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/getfloresta/Floresta"

CMD [ "florestad" ]

FROM rust:1.75-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


COPY Cargo.toml Cargo.lock ./


RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src


COPY src ./src


RUN cargo build --release


EXPOSE 8080


CMD ["./target/release/rust-backend"]


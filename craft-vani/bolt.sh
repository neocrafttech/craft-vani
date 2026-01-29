#!/usr/bin/env bash
set -e

RUST_VERSION="1.93.0"
PROTOBUF_VERSION="21.12"
format() {
    cargo +nightly fmt;
}

setup_protobuf() {
    echo "[INFO] Checking Protobuf installation..."

    OS="$(uname -s)"
    ARCH="$(uname -m)"

    if [ "$OS" = "Linux" ]; then
        sudo apt-get update -y
        sudo apt-get install -y protobuf-compiler unzip
        PROTO_OS="linux"
    elif [ "$OS" = "Darwin" ]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "[WARN] Homebrew not found. Manual install will proceed but system dependencies might be missing."
        fi
        PROTO_OS="osx"
    else
        echo "[ERROR] Unsupported OS: $OS"
        exit 1
    fi

    if [ "$ARCH" = "x86_64" ]; then
        PROTO_ARCH="x86_64"
    elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        if [ "$PROTO_OS" = "osx" ]; then
            PROTO_ARCH="aarch_64"
        else
            PROTO_ARCH="aarch_64"
        fi
    else
        echo "[ERROR] Unsupported architecture: $ARCH"
        exit 1
    fi

    ZIP_FILE="protoc-$PROTOBUF_VERSION-$PROTO_OS-$PROTO_ARCH.zip"

    if command -v protoc >/dev/null 2>&1; then
        CURRENT_VERSION=$(protoc --version | awk '{print $2}')
        echo "[INFO] Found Protobuf version $CURRENT_VERSION"
        if [ "$CURRENT_VERSION" != "$PROTOBUF_VERSION" ]; then
            echo "[INFO] Updating Protobuf to $PROTOBUF_VERSION..."
            curl -L "https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF_VERSION/$ZIP_FILE" -o protoc.zip
            unzip -o protoc.zip

            SUDO=""
            if [ ! -w "/usr/local/bin" ] || [ ! -w "/usr/local/include" ]; then
                SUDO="sudo"
            fi

            $SUDO mv bin/protoc /usr/local/bin/
            # Copy contents to avoid nesting and ensure we don't delete existing include dir if it has other things
            $SUDO mkdir -p /usr/local/include/google
            $SUDO cp -r include/google/* /usr/local/include/google/

            rm -rf protoc.zip bin include readme.txt
        else
            echo "[OK] Protobuf is already $PROTOBUF_VERSION"
        fi
    else
        echo "[INFO] Protobuf not found. Installing Protobuf $PROTOBUF_VERSION..."
        curl -L "https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF_VERSION/$ZIP_FILE" -o protoc.zip
        unzip -o protoc.zip

        SUDO=""
        if [ ! -w "/usr/local/bin" ] || [ ! -w "/usr/local/include" ]; then
            SUDO="sudo"
        fi

        $SUDO mv bin/protoc /usr/local/bin/
        $SUDO mkdir -p /usr/local/include/google
        $SUDO cp -r include/google/* /usr/local/include/google/

        rm -rf protoc.zip bin include readme.txt
    fi
}

setup_rust(){
    echo "[INFO] Checking Rust installation..."

    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        sudo apt-get update -y
        sudo apt-get install -y build-essential curl
    fi

    if command -v rustc >/dev/null 2>&1; then
        CURRENT_VERSION=$(rustc --version | awk '{print $2}')
        echo "[INFO] Found Rust version $CURRENT_VERSION"
        if [ "$CURRENT_VERSION" != "$RUST_VERSION" ]; then
            echo "[INFO] Updating Rust to $RUST_VERSION..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
        else
            echo "[OK] Rust is already $RUST_VERSION"
        fi
    else
        echo "[INFO] Rust not found. Installing Rust $RUST_VERSION..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
    fi

    export PATH="$HOME/.cargo/bin:$PATH"
    source "$HOME/.cargo/env" 2>/dev/null || true

    rustc --version
    cargo --version

    echo "[INFO] Installing cargo-nextest if missing..."
    if ! cargo nextest --version >/dev/null 2>&1; then
        cargo install cargo-nextest
    fi
    cargo nextest --version
    rustup target add wasm32-unknown-unknown
}

setup() {
    setup_rust
    setup_protobuf
}

clean() {
    echo "[INFO] Cleaning workspace..."
    cargo clean
    echo "[OK] Workspace cleaned!"
}

check() {
    echo "[INFO] Running cargo check..."
    cargo check
    echo "[OK] Cargo check passed!"

    echo "[INFO] Checking code formatting..."
    cargo fmt -- --check
    echo "[OK] Code is properly formatted!"

    echo "[INFO] Running clippy lints..."
    cargo clippy -- -D warnings
    echo "[OK] Clippy checks passed!"
}

build() {
    echo "[INFO] Building with native CPU optimizations..."
    RUSTFLAGS="-C target-cpu=native" cargo build --release
    echo "[OK] Build completed!"
}

test() {
    echo "[INFO] Testing workspace..."
    cargo nextest run
    echo "[OK] Testing completed!"
}

bench() {
    echo "[INFO] Running benchmarks..."
    cargo bench
    echo "[OK] Benchmarks completed!"
}

launch() {
    echo "[INFO] Launching application..."
    trunk serve --release --public-url / --port 8080
    echo "[OK] Application launched!"
}

help() {
    echo "Usage: $0 [setup|check|format|clean|build|test|bench|all|help]"
    echo
    echo "Commands:"
    echo "  setup   - Install Rust and cargo-nextest"
    echo "  format  - Format the code"
    echo "  check   - Run cargo check, fmt, and clippy"
    echo "  clean   - Clean the workspace"
    echo "  build   - Only build the workspace (runs check first)"
    echo "  test    - Only run tests"
    echo "  bench   - Only run benchmarks"
    echo "  all     - Run check, build, and test"
    echo "  help    - Show this help message"
}

main() {
    cmd="$1"
    case "$cmd" in
        setup)
            setup
            ;;
        format)
            format
            ;;
        check)
            check
            ;;
        clean)
            clean
            ;;
        build)
            build
            ;;
        test)
            test
            ;;
        bench)
            bench
            ;;
        launch)
            launch
            ;;
        all)
            setup
            check
            build
            ;;
        help|""|*)
            help
            ;;
    esac
}

main "$@"

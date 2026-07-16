#!/usr/bin/env bash
set -euo pipefail

# ─── Logger Helpers ───────────────────────────────────────────────────────────
info()    { echo -e "\e[34m[INFO]\e[0m $*"; }
warn()    { echo -e "\e[33m[WARN]\e[0m $*"; }
success() { echo -e "\e[32m[SUCCESS]\e[0m $*"; }
error()   { echo -e "\e[31m[ERROR]\e[0m $*" >&2; }

# ─── Defaults ─────────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-llama.cpp/build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local/bin}"
CUDA_ARCH="${CUDA_ARCH:-}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PARALLEL_JOBS="${PARALLEL_JOBS:-}"
TARGETS="${TARGETS:-llama-cli llama-mtmd-cli llama-server llama-gguf-split}"
SKIP_INSTALL=false
SKIP_CLONE=false
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ─── Help ─────────────────────────────────────────────────────────────────────
usage() {
    cat << EOF
Install llama.cpp from source with CUDA support.

USAGE:
    $0 [options]

OPTIONS:
    -h, --help              Show this help message
    --repo URL              llama.cpp repository URL
    --build-dir DIR         Build directory (default: llama.cpp/build)
    --prefix DIR            Install prefix (default: /usr/local/bin)
    --cuda-arch ARCH        CUDA architecture (e.g. "89"; auto-detected if empty)
    --build-type TYPE       CMake build type (default: Release)
    -j N                    Parallel build jobs (default: nproc)
    --targets LIST          Build targets (default: llama-cli llama-mtmd-cli llama-server llama-gguf-split)
    --skip-install          Build only, do not install
    --skip-clone            Skip git clone (use existing llama.cpp directory)
    --gpus LIST             CUDA_VISIBLE_DEVICES (default: 0,1)

ENVIRONMENT:
    REPO_URL, BUILD_DIR, INSTALL_PREFIX, CUDA_ARCH, BUILD_TYPE,
    PARALLEL_JOBS, TARGETS, CUDA_VISIBLE_DEVICES

EXAMPLES:
    $0                                          # Default install
    $0 --cuda-arch 89 -j 32                     # Build for H100
    $0 --skip-install --build-type Debug        # Debug build only
    $0 --targets "llama-server" --prefix ~/bin  # Single target, user install
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --prefix) INSTALL_PREFIX="$2"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --build-type) BUILD_TYPE="$2"; shift 2 ;;
        -j) PARALLEL_JOBS="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        --skip-clone) SKIP_CLONE=true; shift ;;
        --gpus) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        *) error "Unknown option: $1"; usage ;;
    esac
done

export CUDA_VISIBLE_DEVICES

# ─── Pre-checks ───────────────────────────────────────────────────────────────
if ! command -v cmake &>/dev/null; then
    error "cmake is not installed. Install build-essential and cmake first."
    exit 1
fi

if ! command -v nvcc &>/dev/null && ! command -v nvidia-smi &>/dev/null; then
    warn "NVIDIA CUDA toolkit or driver not detected. GGML_CUDA=ON may fail."
fi

if [[ -z "$PARALLEL_JOBS" ]]; then
    PARALLEL_JOBS=$(nproc 2>/dev/null || echo 8)
fi

info "Installation configuration:"
info "  Repository:    $REPO_URL"
info "  Build dir:     $BUILD_DIR"
info "  Install dir:   $INSTALL_PREFIX"
info "  Build type:    $BUILD_TYPE"
info "  Parallel jobs: $PARALLEL_JOBS"
info "  Targets:       $TARGETS"
info "  CUDA devices:  $CUDA_VISIBLE_DEVICES"
echo ""

# ─── Install system deps ──────────────────────────────────────────────────────
if command -v apt-get &>/dev/null; then
    info "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y pciutils build-essential cmake curl libcurl4-openssl-dev
    # Install NVIDIA compute libraries if driver version is 535
    if nvidia-smi 2>/dev/null | grep -q "535"; then
        sudo apt-get install -y libnvidia-compute-535 libnvidia-compute-535-server 2>/dev/null || true
    fi
elif command -v yum &>/dev/null; then
    info "Installing system dependencies (yum)..."
    sudo yum install -y cmake make gcc gcc-c++ curl libcurl-devel
elif command -v apk &>/dev/null; then
    info "Installing system dependencies (apk)..."
    apk add build-base cmake curl-dev
fi

# ─── Clone / Update ────────────────────────────────────────────────────────────
LLAMA_DIR="$(dirname "$BUILD_DIR")"
if [[ "$SKIP_CLONE" == false ]]; then
    if [[ -d "$LLAMA_DIR" ]]; then
        info "Updating existing llama.cpp repository..."
        git -C "$LLAMA_DIR" pull --rebase
    else
        info "Cloning llama.cpp from $REPO_URL..."
        git clone "$REPO_URL"
    fi
else
    if [[ ! -d "$LLAMA_DIR" ]]; then
        error "Directory $LLAMA_DIR does not exist (--skip-clone specified)."
        exit 1
    fi
    info "Skipping clone, using existing $LLAMA_DIR"
fi

# ─── Detect CUDA architecture ─────────────────────────────────────────────────
if [[ -z "$CUDA_ARCH" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        # Map GPU compute capability to CUDA arch
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        case "$gpu_name" in
            *H100*|*H200*) CUDA_ARCH="90" ;;
            *A100*|*A30*)  CUDA_ARCH="80" ;;
            *A40*|*A6000*) CUDA_ARCH="86" ;;
            *V100*)        CUDA_ARCH="70" ;;
            *T4*|*RTX2080*) CUDA_ARCH="75" ;;
            *RTX3090*|*RTX4090*) CUDA_ARCH="86" ;;
            *RTX4070*|*RTX4080*|*RTX4060*) CUDA_ARCH="89" ;;
            *L40S*|*L40*)  CUDA_ARCH="89" ;;
            *) CUDA_ARCH="" ;;  # Let CMake auto-detect
        esac
        info "Detected GPU: $gpu_name -> CUDA arch: ${CUDA_ARCH:-auto}"
    fi
fi

CMAKE_GPU_FLAGS=()
if [[ -n "$CUDA_ARCH" ]]; then
    CMAKE_GPU_FLAGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
fi

# ─── CMake Configure ──────────────────────────────────────────────────────────
info "Configuring with CMake..."
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    "${CMAKE_GPU_FLAGS[@]}" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"

# ─── Build ────────────────────────────────────────────────────────────────────
info "Building targets: $TARGETS..."
cmake --build "$BUILD_DIR" \
    --config "$BUILD_TYPE" \
    -j "$PARALLEL_JOBS" \
    --clean-first \
    --target $TARGETS

success "Build completed successfully."

# ─── Install ───────────────────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == false ]]; then
    info "Installing binaries to $INSTALL_PREFIX..."
    mkdir -p "$INSTALL_PREFIX"
    for target in $TARGETS; do
        bin_path=$(find "$BUILD_DIR/bin" "$BUILD_DIR" -name "$target" -type f 2>/dev/null | head -1)
        if [[ -f "$bin_path" ]]; then
            install -m 755 "$bin_path" "$INSTALL_PREFIX/"
            info "  Installed: $INSTALL_PREFIX/$target"
        else
            warn "  Binary not found for target: $target"
        fi
    done
    success "llama.cpp installed to $INSTALL_PREFIX"
else
    info "Skipping installation (--skip-install). Binaries are in $BUILD_DIR/bin/"
fi

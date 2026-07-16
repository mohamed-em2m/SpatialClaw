#!/usr/bin/env bash
# ============================================================================
# SpatialAgent — Environment Setup Script
#
# Creates three environments + optional llama.cpp build:
#   1. spatialagent       (conda)  — agent, managers, GPU server
#   2. .venv              (uv)     — vLLM inference server
#   3. spatialclaw-cuda   (conda)  — CUDA shared libraries borrowed by vLLM at runtime
#   4. llama.cpp          (binary) — GGUF inference server built with CUDA support
#
# Usage:
#   bash setup.sh              # full install (all three envs + llama.cpp)
#   bash setup.sh --agent      # agent conda env only
#   bash setup.sh --vllm       # vLLM .venv only
#   bash setup.sh --cuda       # spatialclaw-cuda CUDA env only
#   bash setup.sh --llama      # llama.cpp build only
#   bash setup.sh --no-llama   # all three envs, skip llama.cpp
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV_NAME="spatialagent"
CUDA_ENV_NAME="spatialclaw-cuda"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.8"
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Llama.cpp config
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_DIR="$PROJECT_ROOT/tools/third_party/llama.cpp"
LLAMA_CPP_BUILD_JOBS="${LLAMA_CPP_BUILD_JOBS:-$(nproc)}"
LLAMA_INSTALL_BINS=true   # set to false to skip /usr/local/bin install

# scripts/ -> spatial_agent/ -> SpatialAgent/
REQUIREMENTS_AGENT="$PROJECT_ROOT/spatial_agent/requirements/requirements-agent.txt"
REQUIREMENTS_VLLM="$PROJECT_ROOT/spatial_agent/requirements/requirements-vllm.txt"
THIRD_PARTY="$PROJECT_ROOT/tools/third_party"

# Third-party model code (SAM3, Pi3, Depth-Anything-3,
# map-anything) is tracked as git submodules — see .gitmodules at the
# project root. Pinned commits live there.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*"; exit 1; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
check_conda() {
    if ! command -v conda &>/dev/null; then
        fail "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    fi
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    # Make sure conda shell functions (activate/deactivate) are available.
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    ok "conda found at $CONDA_BASE"
}

check_requirements() {
    [ -f "$REQUIREMENTS_AGENT" ] || fail "Missing $REQUIREMENTS_AGENT"
    [ -f "$REQUIREMENTS_VLLM" ]  || fail "Missing $REQUIREMENTS_VLLM"
    [ -d "$PROJECT_ROOT/spatial_agent" ] || fail "Missing spatial_agent/ package directory"
    ok "Project files present"
}

# ---------------------------------------------------------------------------
# 1. Agent conda environment
# ---------------------------------------------------------------------------
setup_agent() {
    info "=== Setting up agent conda environment ($CONDA_ENV_NAME) ==="

    if conda env list 2>/dev/null | grep -qw "$CONDA_ENV_NAME"; then
        info "Conda env '$CONDA_ENV_NAME' already exists — installing into it"
    else
        info "Creating conda env '$CONDA_ENV_NAME' (Python $PYTHON_VERSION)..."
        conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y -q
    fi

    conda activate "$CONDA_ENV_NAME"

    info "Installing agent dependencies..."
    pip install --no-user -q -r "$REQUIREMENTS_AGENT"

    # ffmpeg binary (needed for video frame extraction in evals)
    if ! command -v ffmpeg &>/dev/null; then
        info "Installing ffmpeg via conda..."
        conda install -y -q ffmpeg
    fi

    # Install sam3 as editable with all deps (pycocotools, timm, iopath, etc.)
    if [ -d "$THIRD_PARTY/sam3" ]; then
        info "Installing sam3 (editable, with dev extras)..."
        pip install --no-user -q -e "$THIRD_PARTY/sam3[dev]"
    fi

    # uv is needed to create the vLLM venv later
    if ! command -v uv &>/dev/null; then
        info "Installing uv..."
        pip install --no-user -q uv
    fi

    # Verify critical imports
    python -c "
import langgraph, openai, jupyter_client, fastapi, pynvml, numpy, cloudpickle, rich
print('All agent imports OK')
" 2>/dev/null || fail "Agent import verification failed"

    conda deactivate
    ok "Agent environment ready"
}

# ---------------------------------------------------------------------------
# 2. vLLM virtual environment (.venv via uv)
# ---------------------------------------------------------------------------
setup_vllm() {
    info "=== Setting up vLLM virtual environment (.venv) ==="

    # Ensure uv is available (may be in the agent env or on PATH)
    if ! command -v uv &>/dev/null; then
        if conda env list 2>/dev/null | grep -qw "$CONDA_ENV_NAME"; then
            conda activate "$CONDA_ENV_NAME"
        else
            fail "uv not found and agent env not set up. Run 'bash setup.sh --agent' first."
        fi
    fi

    if [ -d "$PROJECT_ROOT/.venv" ]; then
        info ".venv already exists — reinstalling packages"
    else
        info "Creating .venv (Python $PYTHON_VERSION)..."
        uv venv "$PROJECT_ROOT/.venv" --python "$PYTHON_VERSION"
    fi

    info "Installing nightly vLLM with CUDA 12.9 (required for Gemma4)..."
    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" \
        -U vllm --pre \
        --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --index-strategy unsafe-best-match

    info "Pinning transformers==5.5.0..."
    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" transformers==5.5.0

    info "Installing additional dependencies (pynvml, pandas)..."
    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" pynvml pandas

    # deep_gemm: required for FP8 models (e.g. Gemma-4-31B-it-FP8).
    # Must be built from source with git submodules (CUTLASS).
    if "$PROJECT_ROOT/.venv/bin/python" -c "import deep_gemm" 2>/dev/null; then
        ok "deep_gemm already installed"
    else
        info "Building deep_gemm from source (FP8 kernel support)..."
        local DG_DIR="${TMPDIR:-/tmp}/DeepGEMM"
        rm -rf "$DG_DIR"
        if git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "$DG_DIR" 2>/dev/null; then
            local CUDA_HOME_BK="$CUDA_HOME"
            export CUDA_HOME="$CONDA_BASE/envs/$CUDA_ENV_NAME"
            export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
            (cd "$DG_DIR" && "$PROJECT_ROOT/.venv/bin/python" setup.py bdist_wheel 2>/dev/null) && {
                # Rename wheel tag for compatibility (linux_x86_64 -> manylinux)
                local WHL
                WHL=$(ls "$DG_DIR"/dist/*.whl 2>/dev/null | head -1)
                if [ -n "$WHL" ]; then
                    local FIXED="${WHL/linux_x86_64/manylinux_2_35_x86_64}"
                    [ "$WHL" != "$FIXED" ] && cp "$WHL" "$FIXED" && WHL="$FIXED"
                    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" "$WHL"
                    ok "deep_gemm installed"
                fi
            } || warn "deep_gemm build failed — FP8 models will not work"
            export CUDA_HOME="${CUDA_HOME_BK:-}"
            rm -rf "$DG_DIR"
        else
            warn "Failed to clone DeepGEMM — FP8 models will not work"
        fi
    fi

    # Verify
    "$PROJECT_ROOT/.venv/bin/python" -c "
import vllm, pynvml, transformers, pandas
print(f'vLLM {vllm.__version__}, transformers {transformers.__version__} OK')
" 2>/dev/null || fail "vLLM import verification failed"

    # Deactivate if we activated the agent env above
    conda deactivate 2>/dev/null || true
    ok "vLLM environment ready"
}

# ---------------------------------------------------------------------------
# 3. CUDA conda environment (spatialclaw-cuda) — shared libraries for vLLM
# ---------------------------------------------------------------------------
setup_cuda() {
    info "=== Setting up CUDA conda environment ($CUDA_ENV_NAME) ==="

    if conda env list 2>/dev/null | grep -qw "$CUDA_ENV_NAME"; then
        info "Conda env '$CUDA_ENV_NAME' already exists — checking CUDA toolkit"
    else
        info "Creating conda env '$CUDA_ENV_NAME'..."
        conda create -n "$CUDA_ENV_NAME" -y -q
    fi

    # Check if cuda-toolkit is already installed
    CUDA_LIB="$CONDA_BASE/envs/$CUDA_ENV_NAME/lib/libcudart.so"
    if [ -f "$CUDA_LIB" ]; then
        ok "CUDA toolkit already installed"
    else
        info "Installing CUDA toolkit $CUDA_VERSION (this may take several minutes)..."
        conda install -n "$CUDA_ENV_NAME" -c nvidia "cuda-toolkit=$CUDA_VERSION" -y -q
    fi

    # Verify
    if ls "$CONDA_BASE/envs/$CUDA_ENV_NAME/lib/libcudart"* &>/dev/null; then
        ok "CUDA environment ready"
    else
        fail "CUDA libs not found in $CUDA_ENV_NAME"
    fi
}

# ---------------------------------------------------------------------------
# 4. llama.cpp — build from source with CUDA support
#
# Clones (or updates) the repo into tools/third_party/llama.cpp, builds with
# GGML_CUDA=ON using the system CUDA toolkit, and optionally installs all
# resulting binaries to /usr/local/bin so they are on PATH everywhere.
#
# Key binaries produced:
#   llama-cli        — interactive / one-shot GGUF inference
#   llama-server     — OpenAI-compatible HTTP server (use instead of vLLM for GGUF models)
#   llama-mtmd-cli   — multimodal (vision) inference
#   llama-gguf-split — shard / merge GGUF files
#
# Environment variables you can override before calling this script:
#   LLAMA_CPP_DIR        — where to clone/build (default: tools/third_party/llama.cpp)
#   LLAMA_CPP_BUILD_JOBS — parallel make jobs (default: nproc)
#   LLAMA_INSTALL_BINS   — set to "false" to skip /usr/local/bin install
#   CUDA_VISIBLE_DEVICES — restrict which GPUs are used (e.g. "0,1")
# ---------------------------------------------------------------------------
setup_llama() {
    info "=== Setting up llama.cpp (CUDA build) ==="

    # ---- System dependencies -----------------------------------------------
    info "Installing system build dependencies (cmake, curl, pciutils)..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            pciutils build-essential cmake curl libcurl4-openssl-dev
    else
        warn "apt-get not available — ensure cmake, curl, and libcurl4-openssl-dev are installed"
    fi

    # ---- NVIDIA compute library (driver-agnostic stubs for building) --------
    # Only install if libcudart is not already findable on the system.
    if ! ldconfig -p 2>/dev/null | grep -q libcuda; then
        info "Installing libnvidia-compute stubs (needed to link CUDA at build time)..."
        # Try versioned packages first; fall back to the unversioned meta-package.
        sudo apt-get install -y -qq libnvidia-compute-535-server 2>/dev/null \
            || sudo apt-get install -y -qq libnvidia-compute-535 2>/dev/null \
            || warn "Could not install libnvidia-compute — CUDA link may fail"
    else
        ok "CUDA compute library already present on system"
    fi

    # ---- Clone or update repo -----------------------------------------------
    if [ -d "$LLAMA_CPP_DIR/.git" ]; then
        info "llama.cpp repo found at $LLAMA_CPP_DIR — pulling latest..."
        git -C "$LLAMA_CPP_DIR" pull --ff-only 2>/dev/null \
            || warn "git pull failed — continuing with existing checkout"
    else
        info "Cloning llama.cpp into $LLAMA_CPP_DIR..."
        mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
        git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
    fi

    # ---- Configure ----------------------------------------------------------
    info "Configuring CMake (GGML_CUDA=ON, static build)..."

    # Use the CUDA toolkit from the spatialclaw-cuda conda env if available,
    # otherwise fall back to system CUDA_HOME / nvcc on PATH.
    local CMAKE_CUDA_ARGS=()
    local CONDA_CUDA_HOME="$CONDA_BASE/envs/$CUDA_ENV_NAME"
    if [ -f "$CONDA_CUDA_HOME/bin/nvcc" ]; then
        info "Using CUDA toolkit from conda env: $CONDA_CUDA_HOME"
        CMAKE_CUDA_ARGS+=(-DCMAKE_CUDA_COMPILER="$CONDA_CUDA_HOME/bin/nvcc")
        export CUDA_HOME="$CONDA_CUDA_HOME"
        export PATH="$CONDA_CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CONDA_CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    elif command -v nvcc &>/dev/null; then
        ok "Using system nvcc: $(nvcc --version 2>&1 | grep release | head -1)"
    else
        warn "nvcc not found in conda env or on PATH — CUDA build may fail"
    fi

    cmake "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        "${CMAKE_CUDA_ARGS[@]}"

    # ---- Build --------------------------------------------------------------
    info "Building llama.cpp targets (jobs: $LLAMA_CPP_BUILD_JOBS)..."
    cmake --build "$LLAMA_CPP_DIR/build" \
        --config Release \
        -j "$LLAMA_CPP_BUILD_JOBS" \
        --clean-first \
        --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

    # ---- Verify key binaries ------------------------------------------------
    local REQUIRED_BINS=(llama-cli llama-server)
    for bin in "${REQUIRED_BINS[@]}"; do
        if [ ! -f "$LLAMA_CPP_DIR/build/bin/$bin" ]; then
            fail "Expected binary not found after build: $LLAMA_CPP_DIR/build/bin/$bin"
        fi
    done

    # ---- Install to /usr/local/bin ------------------------------------------
    if [ "$LLAMA_INSTALL_BINS" = "true" ]; then
        info "Installing llama.cpp binaries to /usr/local/bin..."
        local installed=0
        for bin in "$LLAMA_CPP_DIR/build/bin/"*; do
            if [ -f "$bin" ]; then
                sudo install -m 755 "$bin" /usr/local/bin/
                installed=$((installed + 1))
            fi
        done
        ok "Installed $installed binaries to /usr/local/bin"
    else
        info "Skipping /usr/local/bin install (LLAMA_INSTALL_BINS=false)"
        info "Binaries available at: $LLAMA_CPP_DIR/build/bin/"
    fi

    # ---- Smoke test ---------------------------------------------------------
    if command -v llama-cli &>/dev/null; then
        llama-cli --version 2>/dev/null && ok "llama-cli smoke test passed" \
            || warn "llama-cli --version returned non-zero (may be normal for some versions)"
    else
        warn "llama-cli not on PATH — add $LLAMA_CPP_DIR/build/bin to your PATH or re-run with LLAMA_INSTALL_BINS=true"
    fi

    ok "llama.cpp build complete"
}

# ---------------------------------------------------------------------------
# 5. Initialize third-party submodules + download model weights
#
# All third-party model code (SAM3, Pi3, Depth-Anything-3,
# map-anything) lives under tools/third_party/ as git submodules pinned to
# the commits below. A normal `git clone --recursive ...` of this repo
# already populates them; this function only fills in submodules that were
# missed (e.g. someone cloned without --recursive).
# ---------------------------------------------------------------------------
setup_third_party() {
    info "=== Initializing third-party submodules and weights ==="

    if [ -f "$PROJECT_ROOT/.gitmodules" ]; then
        info "Running git submodule update --init --recursive..."
        git -C "$PROJECT_ROOT" submodule update --init --recursive
        ok "Submodules initialized"
    else
        warn "No .gitmodules at $PROJECT_ROOT — skipping submodule init"
    fi

    mkdir -p "$THIRD_PARTY"

    # --- SAM3.1 weights (gated HF repo — requires login + license) ---
    SAM3_WEIGHTS_DIR="$THIRD_PARTY/sam3/weights"
    SAM3_CHECKPOINT="$SAM3_WEIGHTS_DIR/sam3.1_multiplex.pt"
    SAM3_BPE="$SAM3_WEIGHTS_DIR/bpe_simple_vocab_16e6.txt.gz"
    SAM3_BPE_SRC="$THIRD_PARTY/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    if [ -f "$SAM3_CHECKPOINT" ]; then
        ok "SAM3.1 checkpoint already present"
    else
        mkdir -p "$SAM3_WEIGHTS_DIR"
        info "Downloading SAM3.1 weights from huggingface.co/facebook/sam3.1..."
        info "(Requires HuggingFace login and license acceptance)"
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download facebook/sam3.1 \
                --local-dir "$SAM3_WEIGHTS_DIR" \
                --include "*.pt" "*.txt.gz" || {
                warn "SAM3.1 download failed. You may need to:"
                warn "  1. Accept the license at https://huggingface.co/facebook/sam3.1"
                warn "  2. Run: huggingface-cli login"
            }
        else
            warn "huggingface-cli not found — install with: pip install huggingface-hub"
        fi
    fi

    # Symlink BPE vocab into weights/ if not already there
    if [ -f "$SAM3_CHECKPOINT" ] && [ ! -e "$SAM3_BPE" ] && [ -f "$SAM3_BPE_SRC" ]; then
        ln -sf ../sam3/assets/bpe_simple_vocab_16e6.txt.gz "$SAM3_BPE"
        ok "BPE vocab symlinked"
    fi

    # Pi3 weights auto-download from HuggingFace on first GPU server launch.
    # DA3 and MapAnything default to local cache use in their GPU wrappers.
    info "Pi3 weights will auto-download from HF (yyfz233/Pi3X) on first use"
    info "DA3 weights should be pre-cached from HF (depth-anything/DA3NESTED-GIANT-LARGE-1.1)"
    info "MapAnything weights should be pre-cached from HF (facebook/map-anything)"
    info "MapAnything also requires DINOv2 giant torch-hub cache under tools/third_party/torch_hub"
}

# ---------------------------------------------------------------------------
# 6. Create log directories
# ---------------------------------------------------------------------------
setup_dirs() {
    info "Creating log directories..."
    mkdir -p "$PROJECT_ROOT/spatial_agent/logs"/{slurm_vllm,slurm_agent,slurm_cot,slurm_gpu_server}
    ok "Directories ready"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "============================================="
    echo "  SpatialAgent Environment Setup"
    echo "============================================="
    echo "  Project root: $PROJECT_ROOT"
    echo "============================================="
    echo ""

    check_conda
    check_requirements

    # Parse arguments
    local do_agent=false do_vllm=false do_cuda=false do_llama=false
    local explicit=false

    if [ $# -eq 0 ]; then
        do_agent=true; do_vllm=true; do_cuda=true; do_llama=true
    else
        for arg in "$@"; do
            case "$arg" in
                --agent)     do_agent=true;  explicit=true ;;
                --vllm)      do_vllm=true;   explicit=true ;;
                --cuda)      do_cuda=true;   explicit=true ;;
                --llama)     do_llama=true;  explicit=true ;;
                --no-llama)
                    # Shorthand: all three envs, no llama.cpp
                    do_agent=true; do_vllm=true; do_cuda=true
                    explicit=true
                    ;;
                --help|-h)
                    echo "Usage: bash setup.sh [--agent] [--vllm] [--cuda] [--llama] [--no-llama]"
                    echo ""
                    echo "  --agent     Set up the spatialagent conda env (agent + managers + GPU server)"
                    echo "  --vllm      Set up the .venv for vLLM inference server"
                    echo "  --cuda      Set up the spatialclaw-cuda conda env with CUDA toolkit"
                    echo "  --llama     Build llama.cpp with CUDA support (GGUF inference)"
                    echo "  --no-llama  Set up all three conda/venv envs, skip llama.cpp build"
                    echo ""
                    echo "  No flags = install all three environments + build llama.cpp."
                    echo ""
                    echo "Environment variables:"
                    echo "  LLAMA_CPP_DIR        Override llama.cpp clone path (default: tools/third_party/llama.cpp)"
                    echo "  LLAMA_CPP_BUILD_JOBS Parallel build jobs (default: nproc = $(nproc))"
                    echo "  LLAMA_INSTALL_BINS   Set to 'false' to skip /usr/local/bin install (default: true)"
                    echo "  CUDA_VISIBLE_DEVICES Restrict GPUs for build/test (e.g. '0,1')"
                    exit 0
                    ;;
                *) fail "Unknown argument: $arg (try --help)" ;;
            esac
        done
    fi

    $do_agent && setup_agent
    $do_cuda  && setup_cuda
    $do_vllm  && setup_vllm
    $do_llama && setup_llama
    setup_third_party
    setup_dirs

    echo ""
    echo "============================================="
    echo "  Setup complete!"
    echo "============================================="
    echo ""
    echo "  Quick start:"
    echo "    conda activate spatialagent"
    echo "    cd $PROJECT_ROOT"
    echo ""
    echo "    # Start a vLLM server (PyTorch models)"
    echo "    python -m spatial_agent.launch_managers.vllm_manager"
    echo ""
    echo "    # Start a llama.cpp server (GGUF models)"
    if [ "$do_llama" = true ]; then
    echo "    llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080 -ngl 99"
    echo ""
    echo "    # One-shot GGUF inference"
    echo "    llama-cli -m /path/to/model.gguf -p 'Your prompt here' -ngl 99"
    fi
    echo ""
    echo "    # Start an agent experiment"
    echo "    python -m spatial_agent.launch_managers.agent_manager"
    echo ""
    echo "  See spatial_agent/README.md for full documentation."
    echo "============================================="
}

main "$@"
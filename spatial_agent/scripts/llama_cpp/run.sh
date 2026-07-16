#!/bin/bash
# llama.cpp SLURM Execution Script (runs INSIDE each SLURM job)
# Sets up environment and launches llama-server for the requested model.
# Configuration is passed as arguments from the manager.

set -e

# Parse arguments
MODEL="$1"
SERVED_NAME="$2"
PORT="${3:-8081}"
CTX_SIZE="${4:-204800}"
GPU_LAYERS="${5:--1}"
PARALLEL_SLOTS="${6:-1}"
MTP="${7:-true}"
HOST="${8:-0.0.0.0}"

if [ -z "$MODEL" ] || [ -z "$SERVED_NAME" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 <model> <served_name> [port] [ctx_size] [gpu_layers] [parallel_slots] [mtp] [host]"
    exit 1
fi

echo "========================================"
echo "llama.cpp Server Startup"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-N/A}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 1: Borrow CUDA shared libraries from spatialclaw-cuda conda environment.
# ---------------------------------------------------------------------------
CUDA_CONDA_ENV="${CUDA_CONDA_ENV:-spatialclaw-cuda}"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -n "$CONDA_BASE" ]; then
    CUDA_PREFIX="$CONDA_BASE/envs/$CUDA_CONDA_ENV"
    if [ -d "$CUDA_PREFIX" ]; then
        export CUDA_HOME="$CUDA_PREFIX"
        export LD_LIBRARY_PATH="$CUDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
        echo "CUDA borrowed from: $CUDA_PREFIX"
    fi
fi

echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Step 2: Environment flags
# ---------------------------------------------------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# HF_TOKEN etc. from .env.local
if [ -f .env.local ]; then
    set -a
    . ./.env.local
    set +a
fi

echo "Model: $MODEL"
echo "Served name: $SERVED_NAME"
echo "Port: $PORT"
echo "Context size: $CTX_SIZE"
echo "GPU layers: $GPU_LAYERS"
echo "Parallel slots: $PARALLEL_SLOTS"
echo "MTP: $MTP"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 3: Determine model flag (-hf for HuggingFace, -m for local file)
# ---------------------------------------------------------------------------
model_flag="-hf"
if [ -f "$MODEL" ]; then
    model_flag="-m"
fi

# ---------------------------------------------------------------------------
# Step 4: Build and launch llama-server
# ---------------------------------------------------------------------------
echo "Starting llama-server..."

llama_args=(
    "$model_flag" "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --alias "$SERVED_NAME"
    -c "$CTX_SIZE"
    -np "$PARALLEL_SLOTS"
    --threads "${THREADS:-16}"
    --threads-batch "${THREADS:-16}"
    -ngl "$GPU_LAYERS"
    --chat-template-kwargs '{"enable_thinking": true}'
    --reasoning on
    --mlock
    --mmap
    --cache-prompt
    --cache-type-k "${KV_CACHE_TYPE:-q4_0}"
    --cache-type-v "${KV_CACHE_TYPE:-q4_0}"
    --cache-reuse 0.7
    --flash-attn on
    --no-warmup
    --temp "${TEMP:-1.0}"
    --top-p "${TOP_P:-0.95}"
    --top-k "${TOP_K:-64}"
    -cb
)

if [ "${MTP,,}" == "true" ]; then
    llama_args+=( --spec-type draft-mtp --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-4}" )
fi

mkdir -p logs
log_file="logs/llama-server-${PORT}.log"

echo "Command: llama-server ${llama_args[*]}"
stdbuf -oL -eL llama-server "${llama_args[@]}" >> "$log_file" 2>&1 &
server_pid=$!
echo "llama-server started (PID: $server_pid). Log: $log_file"

# ---------------------------------------------------------------------------
# Step 5: Health check
# ---------------------------------------------------------------------------
echo "Waiting for server health check..."
waited=0
while ! curl -s "http://${HOST}:${PORT}/health" &>/dev/null; do
    sleep 1
    (( waited++ )) || true
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: llama-server process exited unexpectedly."
        exit 1
    fi
    if (( waited >= 180 )); then
        echo "ERROR: Server did not respond to /health within 180 seconds."
        exit 1
    fi
done
echo "llama-server is running and healthy on port $PORT."

# ---------------------------------------------------------------------------
# Step 5.5: Register in serve.json so LLMClient can discover this server
# ---------------------------------------------------------------------------
echo "Registering server in serve.json..."


export SERVED_NAME PORT CTX_SIZE PARALLEL_SLOTS
python -u << 'PYEOF' 2>&1 || echo "WARNING: Failed to register in serve.json (non-fatal)"
import json, os, socket, sys
from pathlib import Path


class _FileLock:
    def __init__(self, lock_path):
        self.lock_path = lock_path
        self._f = None

    def __enter__(self):
        self._f = open(self.lock_path, 'a+')
        self._f.seek(0)
        try:
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
        except ImportError:
            import msvcrt
            msvcrt.locking(self._f.fileno(), msvcrt.LK_LOCK, 1)
        return self._f

    def __exit__(self, *exc):
        if self._f:
            try:
                import fcntl
                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            self._f.close()
            self._f = None


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


project_root = Path('.').absolute()
log_dir = project_root / 'spatial_agent' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
serve_file = log_dir / 'serve.json'
lock_file = str(serve_file) + '.lock'

served_name = os.environ['SERVED_NAME']
port = os.environ['PORT']
ctx_size = os.environ['CTX_SIZE']
parallel_slots = os.environ['PARALLEL_SLOTS']
uid = f'llama-{os.getpid()}'

with _FileLock(lock_file):
    if serve_file.exists():
        try:
            with open(serve_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
    else:
        data = {}

    if served_name not in data:
        data[served_name] = {}

    data[served_name][uid] = {
        'pid': str(os.getpid()),
        'ip': get_local_ip(),
        'port': str(port),
        'tp': '1',
        'gpus': [0],
        'max_model_len': str(ctx_size),
        'max_num_seqs': str(parallel_slots),
        'create_time': __import__('datetime').datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', None),
    }

    with open(serve_file, 'w') as f:
        json.dump(data, f, indent=2)

print(f'Registered in serve.json as {served_name}/{uid}')
PYEOF

# ---------------------------------------------------------------------------
# Step 6: Wait for server process (block so SLURM keeps job alive)
# ---------------------------------------------------------------------------
wait "$server_pid"

# ---------------------------------------------------------------------------
# Step 7: Deregister from serve.json
# ---------------------------------------------------------------------------
echo "Deregistering server from serve.json..."

python -u << 'PYEOF' 2>&1 || echo "WARNING: Failed to deregister from serve.json (non-fatal)"
import json, os, sys
from pathlib import Path


class _FileLock:
    def __init__(self, lock_path):
        self.lock_path = lock_path
        self._f = None

    def __enter__(self):
        self._f = open(self.lock_path, 'a+')
        self._f.seek(0)
        try:
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
        except ImportError:
            import msvcrt
            msvcrt.locking(self._f.fileno(), msvcrt.LK_LOCK, 1)
        return self._f

    def __exit__(self, *exc):
        if self._f:
            try:
                import fcntl
                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            self._f.close()
            self._f = None


project_root = Path('.').absolute()
serve_file = project_root / 'spatial_agent' / 'logs' / 'serve.json'
lock_file = str(serve_file) + '.lock'

served_name = os.environ.get('SERVED_NAME', '')
uid = f'llama-{os.getpid()}'

with _FileLock(lock_file):
    if not serve_file.exists():
        sys.exit(0)
    try:
        with open(serve_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        sys.exit(0)

    if served_name in data and uid in data[served_name]:
        del data[served_name][uid]
        if not data[served_name]:
            del data[served_name]

        with open(serve_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Deregistered {served_name}/{uid} from serve.json')
PYEOF

echo "========================================"
echo "llama-server stopped"
echo "End time: $(date)"
echo "========================================"

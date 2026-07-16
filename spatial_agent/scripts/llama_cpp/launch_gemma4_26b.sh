#!/bin/bash
# User-facing launch script for llama.cpp server
# This script contains all configurable parameters

#=============================================================================
# Model Configuration
#=============================================================================
MODEL="unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ2_M"
MODEL_NAME="Gemma-4-26B-A4B-IT-GGUF"
SERVED_NAME="gemma-4-26b-a4b"
PORT=8081
IS_MTP=true

#=============================================================================
# Help message
#=============================================================================
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << EOF
llama.cpp Server Launcher

This script launches a llama.cpp server instance with the configured model.

USAGE:
    bash spatial_agent/scripts/llama_cpp/launch_gemma4_26b.sh

CONFIGURATION:
    Edit this file to modify:
    - Model settings (MODEL, MODEL_NAME, SERVED_NAME)
    - Port setting (PORT)
    - MTP drafting (IS_MTP)

STOP:
    Press Ctrl+C in this terminal (will stop and clean up the server)
EOF
    exit 0
fi

#=============================================================================
# Launch the server
#=============================================================================
echo "========================================"
echo "Launching llama.cpp Server"
echo "========================================"
echo "Model Path:     $MODEL"
echo "Model Name:     $MODEL_NAME"
echo "Served name:    $SERVED_NAME"
echo "Port:           $PORT"
echo "MTP Enabled:    $IS_MTP"
echo "========================================"
echo ""

# Get the project root directory
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Call start_llama_server from python and block to keep alive/cleanup
python -c "
import sys, time
from pathlib import Path

# Add project root to path
project_root = Path(r'$SCRIPTS_DIR')
sys.path.insert(0, str(project_root))

from spatial_agent.launch_managers.llama_cpp.server_chain import start_llama_server, stop_llama_server

inst = start_llama_server(
    project_root=project_root,
    model_name='$MODEL_NAME',
    model_path='$MODEL',
    served_name='$SERVED_NAME',
    port=$PORT,
    is_mtp=$IS_MTP
)

if inst:
    try:
        print('Server running. Press Ctrl+C to stop.')
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nStopping server...')
    finally:
        stop_llama_server(project_root, inst)
else:
    print('Failed to start llama-server')
"

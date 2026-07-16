import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from spatial_agent.launch_managers.llama_cpp.state import FileLock, LlamaState, LlamaStateManager


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def register_serve(project_root: Path, served_name: str, port: int, pid: int) -> str:
    """Register the server endpoint in serve.json so LLMClient can discover it."""
    log_dir = project_root / "spatial_agent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    serve_file = log_dir / "serve.json"
    lock_file = str(serve_file) + ".lock"

    uid = f"llama-{port}"

    with FileLock(lock_file):
        if serve_file.exists():
            try:
                with open(serve_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = {}
        else:
            data = {}

        if served_name not in data:
            data[served_name] = {}

        data[served_name][uid] = {
            "pid": str(pid),
            "ip": get_local_ip(),
            "port": str(port),
            "tp": "1",
            "gpus": [0],
            "max_model_len": "204800",
            "max_num_seqs": "1",
            "create_time": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "slurm_job_id": None,
        }

        with open(serve_file, "w") as f:
            json.dump(data, f, indent=2)

    return uid


def deregister_serve(project_root: Path, served_name: str, port: int) -> None:
    """Remove the server from serve.json."""
    serve_file = project_root / "spatial_agent" / "logs" / "serve.json"
    lock_file = str(serve_file) + ".lock"

    uid = f"llama-{port}"

    with FileLock(lock_file):
        if not serve_file.exists():
            return
        try:
            with open(serve_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return

        if served_name in data and uid in data[served_name]:
            del data[served_name][uid]
            if not data[served_name]:
                del data[served_name]

            with open(serve_file, "w") as f:
                json.dump(data, f, indent=2)


def start_llama_server(
    project_root: Path,
    model_name: str,
    model_path: str,
    served_name: str,
    port: int,
    is_mtp: bool,
) -> Optional[LlamaState]:
    """Start llama-server in the background directly using python subprocess."""
    # Ensure logs folder exists
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "api-server.log"

    model_flag = "-m" if os.path.exists(model_path) else "-hf"

    cmd = [
        "llama-server",
        model_flag, model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--alias", model_name,
        "-c", "204800",
        "-np", "1",
        "--threads", "16",
        "--threads-batch", "16",
        "-ngl", "-1",
        "--chat-template-kwargs", '{"enable_thinking": true}',
        "--reasoning", "on",
        "--mlock",
        "--mmap",
        "--cache-prompt",
        "--slot-save-path", str(project_root / "cache_dir"),
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--cache-reuse", "0.7",
        "--flash-attn", "on",
        "--no-warmup",
        "--temp", "1.0",
        "--top-p", "0.95",
        "--top-k", "64",
        "-cb"
    ]

    if is_mtp:
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"])

    creationflags = 0
    if sys.platform == "win32":
        creationflags = 0x00000008  # DETACHED_PROCESS

    print(f"[LlamaManager] Spawning command: {' '.join(cmd)}")
    log_f = open(log_file, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=log_f,
        cwd=str(project_root),
        start_new_session=True,
        creationflags=creationflags,
    )

    pid = proc.pid

    server_url = f"http://127.0.0.1:{port}"
    print(f"[LlamaManager] Waiting for server health check at {server_url}...")
    waited = 0
    healthy = False
    while waited < 180:
        if proc.poll() is not None:
            print("[LlamaManager] llama-server process exited unexpectedly.")
            break
        try:
            r = requests.get(f"{server_url}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                healthy = True
                break
        except Exception:
            pass
        time.sleep(1)
        waited += 1

    if not healthy:
        print("[LlamaManager] Server did not become healthy in time. Terminating...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return None

    print(f"[LlamaManager] llama-server is running and healthy on port {port}.")

    # Warmup model
    try:
        print("[LlamaManager] Warming up model...")
        requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model_path,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "stream": False,
            },
            timeout=60,
        )
        print("[LlamaManager] Warmup done — TTFT will be faster for real requests")
    except Exception as e:
        print(f"[LlamaManager] Warmup failed (non-fatal): {e}")

    chain_id = f"llama-chain-{port}"
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Register in state manager and serve.json
    state_manager = LlamaStateManager(project_root)
    instance = LlamaState(
        chain_id=chain_id,
        served_name=served_name,
        model_name=model_name,
        model_path=model_path,
        pid=pid,
        port=port,
        started_at=started_at,
    )
    state_manager.add_instance(instance)
    register_serve(project_root, served_name, port, pid)

    return instance


def stop_llama_server(project_root: Path, instance: LlamaState) -> None:
    """Stop the running llama-server process."""
    print(f"[LlamaManager] Stopping server at PID {instance.pid} (port {instance.port})...")

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(instance.pid)], capture_output=True)
        else:
            os.kill(instance.pid, signal.SIGTERM)
            waited = 0
            while waited < 5:
                try:
                    os.kill(instance.pid, 0)
                except OSError:
                    break
                time.sleep(1)
                waited += 1
            else:
                os.kill(instance.pid, signal.SIGKILL)
    except Exception as e:
        print(f"[LlamaManager] Error while stopping process: {e}")

    # Clean up state and serve.json
    state_manager = LlamaStateManager(project_root)
    state_manager.remove_instance(instance.chain_id)
    deregister_serve(project_root, instance.served_name, instance.port)

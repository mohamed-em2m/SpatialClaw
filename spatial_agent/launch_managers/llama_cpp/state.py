"""Persistent state for active llama.cpp server processes."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LlamaState:
    chain_id: str
    served_name: str
    model_name: str
    model_path: str
    pid: int
    port: int
    started_at: str = ""


class FileLock:
    """Cross-platform file lock supporting Windows and Unix."""

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._f = None

    def __enter__(self):
        self._f = open(self.lock_path, "a+")
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
                import msvcrt
                try:
                    self._f.seek(0)
                    msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            self._f.close()
            self._f = None


class LlamaStateManager:
    """Read/write llama_cpp_manager_state.json with file locking."""

    def __init__(self, project_root: Path):
        self.state_dir = project_root / "spatial_agent" / "logs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "llama_cpp_manager_state.json"
        self.lock_file = str(self.state_file) + ".lock"

    def _read(self) -> List[dict]:
        if not self.state_file.exists():
            return []
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, instances: List[dict]) -> None:
        with open(self.state_file, "w") as f:
            json.dump(instances, f, indent=2)

    def add_instance(self, instance: LlamaState) -> None:
        with FileLock(self.lock_file):
            instances = self._read()
            instances.append(asdict(instance))
            self._write(instances)

    def remove_instance(self, chain_id: str) -> None:
        with FileLock(self.lock_file):
            instances = self._read()
            instances = [c for c in instances if c.get("chain_id") != chain_id]
            self._write(instances)

    def list_instances(self) -> List[LlamaState]:
        with FileLock(self.lock_file):
            raw = self._read()
        result = []
        for c in raw:
            try:
                result.append(LlamaState(**c))
            except TypeError:
                continue
        return result

    def is_instance_alive(self, instance: LlamaState) -> bool:
        """Check if the llama-server process PID is still running."""
        try:
            os.kill(instance.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def cleanup_dead_instances(self) -> List[LlamaState]:
        """Remove instances whose PID is dead. Returns removed instances."""
        with FileLock(self.lock_file):
            instances = self._read()
            alive = []
            dead = []
            for c in instances:
                pid = c.get("pid", 0)
                try:
                    os.kill(pid, 0)
                    alive.append(c)
                except (ProcessLookupError, PermissionError, OSError):
                    dead.append(c)
            self._write(alive)
        return [LlamaState(**c) for c in dead if _is_valid_instance(c)]


def _is_valid_instance(c: dict) -> bool:
    try:
        LlamaState(**c)
        return True
    except TypeError:
        return False

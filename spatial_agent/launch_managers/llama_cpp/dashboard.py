"""Dashboard: render status of all llama.cpp servers."""

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.llama_cpp.state import LlamaState, LlamaStateManager


class Dashboard:

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.serve_file = project_root / "spatial_agent" / "logs" / "serve.json"
        self.state_manager = LlamaStateManager(project_root)
        self.console = Console()

    def render(self) -> None:
        """Render the full dashboard."""
        instances = self.state_manager.list_instances()
        registry = self._load_serve_registry()

        # Clean up dead instances silently
        dead = self.state_manager.cleanup_dead_instances()
        if dead:
            instances = self.state_manager.list_instances()

        # === Managed Servers Table ===
        if instances:
            table = Table(
                title="Managed llama.cpp Servers",
                title_style="bold cyan",
                border_style="cyan",
                show_lines=True,
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Model", style="bold white", min_width=18)
            table.add_column("Served Name", style="white", min_width=16)
            table.add_column("Status", min_width=10, justify="center")
            table.add_column("PID", min_width=8)
            table.add_column("Endpoint", min_width=22)
            table.add_column("Started At", min_width=18)

            for idx, inst in enumerate(instances, 1):
                alive = self.state_manager.is_instance_alive(inst)

                if not alive:
                    status = Text("DEAD", style="bold red")
                    table.add_row(
                        str(idx), inst.model_name, inst.served_name,
                        status, str(inst.pid), "-", inst.started_at,
                    )
                else:
                    status = Text("RUNNING", style="bold green")
                    endpoint = f"http://127.0.0.1:{inst.port}"
                    table.add_row(
                        str(idx), inst.model_name, inst.served_name,
                        status, str(inst.pid), endpoint, inst.started_at,
                    )

            self.console.print(table)
            self.console.print()
        else:
            self.console.print(
                Panel(
                    "[dim]No llama.cpp servers currently running[/dim]",
                    title="llama.cpp Dashboard",
                    border_style="dim",
                ),
            )

    def _load_serve_registry(self) -> Dict:
        if not self.serve_file.exists():
            return {}
        try:
            with open(self.serve_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

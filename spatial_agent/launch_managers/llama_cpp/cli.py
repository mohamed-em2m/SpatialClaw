"""Interactive CLI for the llama.cpp Server Manager."""

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.llama_cpp.config import ModelConfig, load_config
from spatial_agent.launch_managers.llama_cpp.dashboard import Dashboard
from spatial_agent.launch_managers.llama_cpp.server_chain import start_llama_server, stop_llama_server
from spatial_agent.launch_managers.llama_cpp.state import LlamaState, LlamaStateManager


class _Abort(Exception):
    """Raised when user types 'q' to cancel during a flow."""


class LlamaCPPManagerCLI:

    def __init__(self):
        # Walk up llama_cpp/ → launch_managers/ → spatial_agent/ → project root.
        self.project_root = Path(__file__).parent.parent.parent.parent.absolute()
        self.config = load_config()
        self.state_manager = LlamaStateManager(self.project_root)
        self.dashboard = Dashboard(self.project_root)
        self.console = Console()

    def run(self) -> None:
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]llama.cpp Server Manager[/bold cyan]\n"
                "[dim]Manage local llama.cpp servers[/dim]",
                border_style="cyan",
                padding=(1, 2),
            ),
        )

        while True:
            self.console.print()
            self._show_quick_status()
            self.console.print()

            menu = Table(show_header=False, box=None, padding=(0, 2))
            menu.add_column(style="bold cyan", width=5)
            menu.add_column()
            menu.add_row("[1]", "Dashboard — view all servers")
            menu.add_row("[2]", "Start Server")
            menu.add_row("[3]", "Stop Server")
            menu.add_row("[q]", "Quit")
            self.console.print(menu)
            self.console.print()

            choice = Prompt.ask(
                "[bold]Select",
                choices=["1", "2", "3", "q"],
                default="1",
            )

            if choice == "1":
                self._show_dashboard()
            elif choice == "2":
                self._start_server()
            elif choice == "3":
                self._stop_server()
            elif choice == "q":
                self.console.print("[dim]Goodbye.[/dim]")
                break

    def _show_quick_status(self) -> None:
        """One-line status summary."""
        instances = self.state_manager.list_instances()
        alive = [c for c in instances if self.state_manager.is_instance_alive(c)]
        if alive:
            names = ", ".join(c.served_name for c in alive)
            self.console.print(
                f"[green]Active llama.cpp servers ({len(alive)}):[/green] {names}",
            )
        else:
            self.console.print("[dim]No llama.cpp servers currently managed.[/dim]")

    def _show_dashboard(self) -> None:
        self.console.print()
        self.dashboard.render()

    def _ask(self, prompt: str, default: str = "") -> str:
        """Prompt that accepts 'q' to abort. Returns value or raises _Abort."""
        val = Prompt.ask(f"{prompt} [dim](q to cancel)[/dim]", default=default)
        if val.strip().lower() == "q":
            raise _Abort()
        return val

    def _ask_int(self, prompt: str, default: int) -> int:
        """Int prompt that accepts 'q' to abort."""
        val = self._ask(prompt, default=str(default))
        try:
            return int(val)
        except ValueError:
            self.console.print("[red]Invalid number.[/red]")
            raise _Abort()

    def _start_server(self) -> None:
        self.console.print()

        try:
            self._start_server_inner()
        except _Abort:
            self.console.print("[dim]Cancelled.[/dim]")

    def _start_server_inner(self) -> None:
        # Step 1: Select model
        models = self.config.models
        if not models:
            self.console.print("[red]No models configured in models.json.[/red]")
            return

        table = Table(
            title="Available GGUF Models",
            title_style="bold cyan",
            border_style="cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Name", style="white", min_width=20)
        table.add_column("Served Name", style="cyan", min_width=18)
        table.add_column("Default Port", justify="center", width=12)
        table.add_column("MTP Enabled", justify="center", width=12)

        for i, m in enumerate(models, 1):
            table.add_row(
                str(i), m.name, m.served_name,
                str(m.port), str(m.is_mtp),
            )

        self.console.print(table)
        self.console.print()

        model_idx = self._ask_int("[bold]Select model", default=1)
        if model_idx < 1 or model_idx > len(models):
            self.console.print("[red]Invalid selection.[/red]")
            return

        model = models[model_idx - 1]

        # Step 2: Port and MTP
        self.console.print()
        port = self._ask_int("  Port", default=model.port)
        is_mtp_str = self._ask("  Enable MTP (true/false)", default=str(model.is_mtp).lower())
        is_mtp = is_mtp_str.strip().lower() == "true"

        # Step 3: Confirm
        self.console.print()
        summary = (
            f"[bold]{model.name}[/bold] ({model.served_name})\n"
            f"  Model Source: {model.model}\n"
            f"  Port:         {port}\n"
            f"  MTP Enabled:  {is_mtp}"
        )
        self.console.print(Panel(summary, title="Confirm Launch", border_style="green"))

        confirm = Prompt.ask("[bold]Launch?", choices=["y", "n"], default="y")
        if confirm != "y":
            self.console.print("[dim]Cancelled.[/dim]")
            return

        # Step 4: Start
        self.console.print()
        self.console.print("[yellow]Starting llama-server...[/yellow]")

        inst = start_llama_server(
            project_root=self.project_root,
            model_name=model.name,
            model_path=model.model,
            served_name=model.served_name,
            port=port,
            is_mtp=is_mtp,
        )

        if inst:
            self.console.print(
                f"[bold green]llama-server started successfully![/bold green]\n"
                f"  PID:      {inst.pid}\n"
                f"  Endpoint: http://127.0.0.1:{inst.port}\n"
                f"  Logs:     logs/api-server.log\n"
                f"[dim]Endpoint registered in serve.json for auto-discovery.[/dim]"
            )
        else:
            self.console.print("[bold red]Failed to start llama-server. Check logs.[/bold red]")

    def _stop_server(self) -> None:
        self.console.print()

        # Clean up dead instances first
        dead = self.state_manager.cleanup_dead_instances()
        if dead:
            for d in dead:
                self.console.print(f"[dim]Cleaned up dead server: {d.served_name} (PID {d.pid})[/dim]")

        instances = self.state_manager.list_instances()
        alive_instances = [c for c in instances if self.state_manager.is_instance_alive(c)]

        if not alive_instances:
            self.console.print("[dim]No managed servers to stop.[/dim]")
            return

        table = Table(
            title="Running llama.cpp Servers",
            title_style="bold red",
            border_style="red",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Model", style="white", min_width=20)
        table.add_column("Served Name", style="cyan", min_width=18)
        table.add_column("PID", width=8)
        table.add_column("Endpoint", min_width=20)

        for display_idx, inst in enumerate(alive_instances, 1):
            endpoint = f"http://127.0.0.1:{inst.port}"
            table.add_row(
                str(display_idx), inst.model_name, inst.served_name,
                str(inst.pid), endpoint,
            )

        self.console.print(table)
        self.console.print()

        try:
            choice = self._ask_int("[bold]Select server to stop", default=1)
            if choice < 1 or choice > len(alive_instances):
                self.console.print("[red]Invalid selection.[/red]")
                return

            inst_to_stop = alive_instances[choice - 1]
            self.console.print(f"[yellow]Stopping server {inst_to_stop.served_name} (PID {inst_to_stop.pid})...[/yellow]")
            stop_llama_server(self.project_root, inst_to_stop)
            self.console.print("[bold green]Stopped successfully.[/bold green]")
        except _Abort:
            self.console.print("[dim]Cancelled.[/dim]")

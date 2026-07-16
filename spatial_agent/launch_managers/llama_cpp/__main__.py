"""Entry point: python -m spatial_agent.launch_managers.llama_cpp"""

from spatial_agent.launch_managers.llama_cpp.cli import LlamaCPPManagerCLI


def main():
    cli = LlamaCPPManagerCLI()
    cli.run()


if __name__ == "__main__":
    main()

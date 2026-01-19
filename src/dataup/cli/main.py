"""DataUp CLI entry point."""

from __future__ import annotations

import os

import click
from rich.table import Table

from dataup.cli.config import CONFIG_FILE, get_config, save_config
from dataup.cli.evaluation import eval_alias, evaluation
from dataup.cli.utils import console


@click.group()
@click.version_option(package_name="dataup")
def cli():
    """DataUp CLI for model evaluation and more."""
    pass


# Register evaluation commands
cli.add_command(evaluation)
cli.add_command(eval_alias)


@cli.command()
def configure():
    """Configure DataUp CLI credentials.

    This command prompts for your credentials and saves them to ~/.dataup/config.
    Environment variables (DATAUP_API_KEY, CVAT_API_TOKEN, CVAT_BASE_URL) take
    precedence over the config file.

    Example:

        dataup configure
    """
    config = get_config()

    if not config.has_section("default"):
        config.add_section("default")

    # Get current values for defaults
    current_dataup_key = config.get("default", "dataup_api_key", fallback="")
    current_cvat_token = config.get("default", "cvat_api_token", fallback="")
    current_cvat_url = config.get("default", "cvat_base_url", fallback="https://app.cvat.ai")

    console.print("[bold]DataUp CLI Configuration[/bold]")
    console.print(f"Config file: {CONFIG_FILE}\n")

    # Prompt for DataUp API Key
    if current_dataup_key:
        if len(current_dataup_key) > 12:
            masked = current_dataup_key[:8] + "..." + current_dataup_key[-4:]
        else:
            masked = "****"
        prompt = f"DataUp API Key [{masked}]"
    else:
        prompt = "DataUp API Key"

    dataup_key = click.prompt(prompt, default="", show_default=False).strip()
    if dataup_key:
        config.set("default", "dataup_api_key", dataup_key)
    elif current_dataup_key:
        # Keep existing value if user just pressed enter
        pass

    # Prompt for CVAT API Token
    if current_cvat_token:
        if len(current_cvat_token) > 12:
            masked = current_cvat_token[:8] + "..." + current_cvat_token[-4:]
        else:
            masked = "****"
        prompt = f"CVAT API Token [{masked}]"
    else:
        prompt = "CVAT API Token"

    cvat_token = click.prompt(prompt, default="", show_default=False).strip()
    if cvat_token:
        config.set("default", "cvat_api_token", cvat_token)
    elif current_cvat_token:
        # Keep existing value if user just pressed enter
        pass

    # Prompt for CVAT Base URL
    cvat_url = click.prompt("CVAT Base URL", default=current_cvat_url).strip()
    config.set("default", "cvat_base_url", cvat_url)

    # Save config
    save_config(config)

    console.print(f"\n[green]Configuration saved to {CONFIG_FILE}[/green]")


@cli.command("show-config")
def show_config():
    """Show current configuration.

    Displays the current configuration from both the config file and
    environment variables, indicating which source is active.

    Example:

        dataup show-config
    """
    config = get_config()

    table = Table(title="DataUp Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    def mask_secret(value: str) -> str:
        if len(value) > 12:
            return value[:8] + "..." + value[-4:]
        return "****"

    # DataUp API Key
    env_key = os.environ.get("DATAUP_API_KEY")
    config_key = None
    if config.has_section("default"):
        config_key = config.get("default", "dataup_api_key", fallback=None)
    if env_key:
        table.add_row("DataUp API Key", mask_secret(env_key), "env: DATAUP_API_KEY")
    elif config_key:
        table.add_row("DataUp API Key", mask_secret(config_key), f"file: {CONFIG_FILE}")
    else:
        table.add_row("DataUp API Key", "[red]Not set[/red]", "-")

    # CVAT API Token
    env_token = os.environ.get("CVAT_API_TOKEN")
    config_token = None
    if config.has_section("default"):
        config_token = config.get("default", "cvat_api_token", fallback=None)
    if env_token:
        table.add_row("CVAT API Token", mask_secret(env_token), "env: CVAT_API_TOKEN")
    elif config_token:
        table.add_row("CVAT API Token", mask_secret(config_token), f"file: {CONFIG_FILE}")
    else:
        table.add_row("CVAT API Token", "[red]Not set[/red]", "-")

    # CVAT Base URL
    env_url = os.environ.get("CVAT_BASE_URL")
    config_url = None
    if config.has_section("default"):
        config_url = config.get("default", "cvat_base_url", fallback=None)
    if env_url:
        table.add_row("CVAT Base URL", env_url, "env: CVAT_BASE_URL")
    elif config_url:
        table.add_row("CVAT Base URL", config_url, f"file: {CONFIG_FILE}")
    else:
        table.add_row("CVAT Base URL", "https://app.cvat.ai", "default")

    console.print(table)


if __name__ == "__main__":
    cli()

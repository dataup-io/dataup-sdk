"""Configuration management for DataUp CLI."""

from __future__ import annotations

import configparser
import os
from pathlib import Path

# Config file location
CONFIG_DIR = Path.home() / ".dataup"
CONFIG_FILE = CONFIG_DIR / "config"


def get_config() -> configparser.ConfigParser:
    """Load configuration from file."""
    config = configparser.ConfigParser()
    if CONFIG_FILE.exists():
        config.read(CONFIG_FILE)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        config.write(f)
    # Set restrictive permissions (owner read/write only)
    CONFIG_FILE.chmod(0o600)


def get_setting(key: str, env_var: str, section: str = "default") -> str | None:
    """Get a setting from environment variable or config file.

    Environment variables take precedence over config file.
    """
    # Check environment variable first
    value = os.environ.get(env_var)
    if value:
        return value

    # Fall back to config file
    config = get_config()
    if config.has_option(section, key):
        return config.get(section, key)

    return None

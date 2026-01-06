import os
import pytest
import importlib
from src import config  # Import the module so we can reload it


def test_default_config_loading():
    """Test that Config has sensible defaults if no .env is present."""
    # Reload to clear any previous monkeypatching
    importlib.reload(config)

    assert config.Config.ROBOT_IP is not None
    assert isinstance(config.Config.ROBOT_PORT, int)
    # Check strict default values (optional, but good)
    assert config.Config.ROBOT_PORT == 5000


def test_env_variable_override(monkeypatch):
    """Test that .env variables actually override the defaults."""

    # 1. Set the "Fake" Environment
    monkeypatch.setenv("ROBOT_IP", "10.0.0.5")
    monkeypatch.setenv("ROBOT_PORT", "9090")

    # 2. RELOAD the module.
    # Without this, Config keeps the old values it loaded at startup.
    importlib.reload(config)

    # 3. Test the Class Attributes (The real test)
    assert config.Config.ROBOT_IP == "10.0.0.5"
    assert config.Config.ROBOT_PORT == 9090
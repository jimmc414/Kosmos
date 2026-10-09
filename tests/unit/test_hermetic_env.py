"""The unit and integration suites run hermetic (tests/conftest.py, VIAB#P3-3).

A failing assertion on a config value once printed a real key from .env into test output.
These tests pin the guard: no credential-like variable reaches a test except the
placeholders the conftest sets, .env is never read, and writable paths stay out of the repo.
"""
import os
from pathlib import Path

import dotenv

import kosmos.config

REPO_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
PLACEHOLDERS = {"DEEPSEEK_API_KEY": "test-placeholder-not-a-real-key"}


def _is_credential(name):
    return any(marker in name.upper() for marker in CREDENTIAL_MARKERS)


def test_no_credential_variable_except_placeholders(hermetic_env_at_import):
    """At conftest import (before fixtures set test values) only placeholders remain."""
    leaked = sorted(
        name for name, value in hermetic_env_at_import.items()
        if _is_credential(name) and value != PLACEHOLDERS.get(name)
    )
    assert leaked == []


def test_no_real_dotenv_credential_reaches_the_tests():
    """No credential value from the repo's .env is in the environment (names only on failure)."""
    real = dotenv.main.dotenv_values(REPO_ROOT / ".env") if (REPO_ROOT / ".env").exists() else {}
    secrets = {name: value for name, value in real.items()
               if value and len(value) >= 8 and (_is_credential(name) or name.upper().endswith("_URL"))}
    present = sorted(name for name, value in secrets.items() if value in os.environ.values())
    assert present == []


def test_settings_classes_do_not_read_dotenv():
    for settings_class in (kosmos.config.KosmosConfig, kosmos.config.LiteLLMConfig):
        assert settings_class.model_config.get("env_file") is None
    assert dotenv.load_dotenv() is False


def test_writable_paths_are_outside_the_repo():
    database_url = os.environ["DATABASE_URL"]
    assert database_url.startswith("sqlite:///")
    paths = [
        Path(database_url[len("sqlite:///"):]),
        Path(os.environ["AUDIT_LOG_PATH"]),
        Path(os.environ["INCIDENT_LOG_PATH"]),
        Path(os.environ["KOSMOS_LITERATURE_CACHE_DIR"]),
    ]
    for path in paths:
        assert REPO_ROOT not in path.resolve().parents, path


def test_default_config_uses_the_test_provider():
    config = kosmos.config.KosmosConfig()
    assert config.llm_provider == "litellm"
    assert config.get_active_model() == "deepseek/deepseek-chat"

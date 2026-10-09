"""Tests for scripts/check_env.py (VIAB#P3-2): rows reported, exit status, no secret values."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_env.py"


@pytest.fixture
def check_env():
    spec = importlib.util.spec_from_file_location("check_env", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_db_url_password_is_masked(check_env):
    masked = check_env.mask_db_url("postgresql://kosmos:s3cret-pw@postgres:5432/kosmos")
    assert "s3cret-pw" not in masked
    assert masked == "postgresql://kosmos:***@postgres:5432/kosmos"


def test_unreachable_daemon_fails_and_reports_every_row(check_env, monkeypatch, capsys):
    fake_docker = MagicMock()
    fake_docker.from_env.return_value.ping.side_effect = ConnectionError("no daemon")
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    monkeypatch.setenv("DATABASE_URL", "postgresql://kosmos:s3cret-pw@postgres:5432/kosmos")
    monkeypatch.setenv("KOSMOS_SANDBOX_IMAGE", "kosmos-sandbox:pinned-test")

    rc = check_env.main()
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAIL  Docker daemon" in out
    assert "daemon unreachable (ConnectionError)" in out
    assert "Sandbox image kosmos-sandbox:pinned-test" in out
    for row in ("LLM provider", "LLM model", "litellm", "sentence_transformers (optional)"):
        assert row in out
    assert "postgresql://kosmos:***@postgres:5432/kosmos" in out
    assert "s3cret-pw" not in out
    assert "failed: Docker daemon, Sandbox image kosmos-sandbox:pinned-test" in out


def test_config_error_prints_only_its_first_line(check_env, monkeypatch, capsys):
    import kosmos.config

    def broken_config():
        raise ValueError("1 validation error for KosmosConfig\n  input_value='sk-secret-value'")

    monkeypatch.setattr(kosmos.config, "KosmosConfig", broken_config)

    rc = check_env.main()
    out = capsys.readouterr().out

    assert rc == 1
    assert "1 validation error for KosmosConfig" in out
    assert "sk-secret-value" not in out

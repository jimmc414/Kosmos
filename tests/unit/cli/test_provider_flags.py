"""
Tests for `kosmos run --provider/--model` (A-2 in evaluation/VIABILITY_PROGRESS.md).

The run command is mounted on a minimal Typer app so the main callback's
database and logging setup do not run; the director and LLM client are mocked.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from kosmos.cli.commands.run import run_research
from kosmos.config import get_config, reset_config
from kosmos.core.providers.selection import (
    ProviderSelectionError,
    apply_provider_selection,
    resolve_model,
)

app = typer.Typer()


@app.callback()
def _root():
    """Test root."""


app.command("run")(run_research)

RESULTS = {"id": "r", "question": "Q", "domain": None, "state": "converged",
           "current_iteration": 1, "max_iterations": 1, "convergence_reason": None,
           "hypotheses": [], "experiments": [], "metrics": {}}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KOSMOS_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "litellm")
    monkeypatch.setenv("LITELLM_MODEL", "deepseek/deepseek-chat")
    reset_config()
    yield monkeypatch
    reset_config()


@pytest.fixture
def mocks():
    # Assert on print_error arguments, not CliRunner output: other CLI tests leave the
    # shared Rich console writing to a stream CliRunner does not capture
    with patch("kosmos.agents.research_director.ResearchDirectorAgent") as director, \
         patch("kosmos.cli.commands.run.run_with_progress_async", new=AsyncMock(return_value=RESULTS)), \
         patch("kosmos.agents.registry.get_registry", return_value=MagicMock()), \
         patch("kosmos.cli.commands.run.print_error") as print_error, \
         patch("kosmos.core.llm.get_client") as get_client:
        yield {"director": director, "get_client": get_client, "print_error": print_error}


def _errors(mocks):
    return " ".join(str(c.args[0]) for c in mocks["print_error"].call_args_list)


def _run(*args):
    return CliRunner().invoke(app, ["run", "Q", "--max-iterations", "1", *args])


def test_deepseek_flag(env, mocks):
    result = _run("--provider", "deepseek")
    assert result.exit_code == 0, result.output
    config = get_config()
    assert config.llm_provider == "litellm"
    assert config.litellm.model == "deepseek/deepseek-chat"
    mocks["get_client"].assert_called_with(reset=True)
    mocks["print_error"].assert_not_called()


def test_claude_code_with_alias(env, mocks):
    result = _run("--provider", "claude-code", "--model", "sonnet")
    assert result.exit_code == 0, result.output
    config = get_config()
    assert config.llm_provider == "claude_code"
    assert config.claude_code.model == "claude-sonnet-5-5"


def test_claude_model_on_deepseek_is_rejected(env, mocks):
    result = _run("--provider", "deepseek", "--model", "opus")
    assert result.exit_code == 1
    assert "--provider claude-code" in _errors(mocks)
    mocks["director"].assert_not_called()


def test_anthropic_without_key_is_rejected(env, mocks):
    result = _run("--provider", "anthropic")
    assert result.exit_code == 1
    assert "KOSMOS_ANTHROPIC_API_KEY" in _errors(mocks)
    assert "--provider claude-code" in _errors(mocks)


def test_no_flags_keep_env_defaults(env, mocks):
    before = (get_config().llm_provider, get_config().get_active_model())
    result = _run()
    assert result.exit_code == 0, result.output
    assert (get_config().llm_provider, get_config().get_active_model()) == before
    mocks["get_client"].assert_not_called()


def test_anthropic_api_key_never_written(env, mocks):
    env.setenv("KOSMOS_ANTHROPIC_API_KEY", "sk-ant-api03-x")
    for args in (["--provider", "anthropic", "--model", "opus"], ["--provider", "claude-code"], ["--provider", "deepseek"]):
        _run(*args)
        assert "ANTHROPIC_API_KEY" not in os.environ


def test_unknown_provider_choice_fails(env, mocks):
    result = _run("--provider", "gemini")
    assert result.exit_code != 0


def test_resolve_model_aliases():
    assert resolve_model("Opus") == "claude-opus-5-5"
    assert resolve_model("deepseek-reasoner") == "deepseek/deepseek-reasoner"
    assert resolve_model("claude-haiku-4-5") == "claude-haiku-4-5"
    assert resolve_model(None) is None


def test_deepseek_ignores_other_litellm_base(env):
    env.setenv("LITELLM_API_BASE", "http://localhost:11434")
    env.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    reset_config()
    config = get_config()
    apply_provider_selection(config, "deepseek", None)
    assert config.litellm.api_base is None
    assert config.litellm.api_key == "sk-deepseek"


def test_model_only_applies_to_env_provider(env):
    config = get_config()
    assert apply_provider_selection(config, None, "deepseek-reasoner") == ("litellm", "deepseek/deepseek-reasoner")
    with pytest.raises(ProviderSelectionError):
        apply_provider_selection(config, None, "fable")

"""
Shared fixtures for the safety tests.

SafetyGuardrails writes its emergency-stop flag file (.kosmos_emergency_stop) and,
by default, its incident log (safety_incidents.jsonl) relative to the working
directory. Every test here runs in its own temporary directory so neither file
lands in the repository or leaks a stop into the next test. The director
fixtures come from tests/unit/agents/conftest.py.
"""

import pytest

from tests.unit.agents.conftest import db_director, in_memory_db  # noqa: F401 (fixtures)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

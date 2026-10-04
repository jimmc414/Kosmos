"""pytest plugin for the ladder: keep the test suite off the owner's paths.

tests/conftest.py calls load_dotenv(.env, override=True) at import time, which runs
BEFORE pytest_configure, so DATABASE_URL=sqlite:///./kosmos.db reaches every test and the
director tests that use the configured database write ResearchSession rows into the
owner's kosmos.db. This hook runs after that import and wins: every writable path points
into VERIFY_RUN_DIR. Proven in Session 0 (2026-10-04): the owner's kosmos.db md5 was
unchanged across the full unit + integration suite while the scratch database received 92
ResearchSession rows and was migrated to head.

Loaded by scripts/verify.sh as `PYTHONPATH=scripts python -m pytest ... -p verify_isolate`.
Refuses to run without VERIFY_RUN_DIR, so it can never silently fall back to kosmos.db.
"""
import os

import pytest


def pytest_configure(config):
    run_dir = os.environ.get("VERIFY_RUN_DIR")
    if not run_dir:
        raise pytest.UsageError(
            "verify_isolate: VERIFY_RUN_DIR is not set; refusing to run against the owner's paths"
        )
    os.makedirs(run_dir, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{run_dir}/ladder.db"
    os.environ["KOSMOS_ARTIFACTS_DIR"] = f"{run_dir}/artifacts"
    os.environ["CHROMA_PERSIST_DIRECTORY"] = f"{run_dir}/chroma"
    os.environ["LOG_FILE_PATH"] = f"{run_dir}/kosmos.log"
    print(f"\n[verify_isolate] DATABASE_URL={os.environ['DATABASE_URL']}")

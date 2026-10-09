"""Running migrations inside the app must not reconfigure its logging (VIAB#P3-3).

alembic/env.py used to call logging.config.fileConfig(alembic.ini), which disables
every logger that already exists and resets the root logger, so the first `kosmos run`
on a fresh database silently muted kosmos logging (the cause of TEST#order-dependent-caplog).
"""
import logging

from kosmos.utils.setup import run_database_migrations


def test_migrations_leave_existing_loggers_and_root_level_alone(tmp_path):
    probe = logging.getLogger("kosmos.test_migration_logging_probe")
    root = logging.getLogger()
    root_level, root_handlers = root.level, list(root.handlers)

    ok, error = run_database_migrations(f"sqlite:///{tmp_path / 'fresh.db'}")

    assert ok, error
    assert probe.disabled is False
    assert root.level == root_level
    assert root.handlers == root_handlers

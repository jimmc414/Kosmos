"""
Setup utilities for first-time initialization.

Handles automatic .env creation and database migrations for pip-installed users.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_env_file() -> bool:
    """
    Ensure .env file exists by copying from .env.example if missing.

    Returns:
        bool: True if .env was created, False if it already existed

    Raises:
        FileNotFoundError: If .env.example is not found
    """
    # Get project root (parent of kosmos package)
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    # Check if .env already exists
    if env_file.exists():
        logger.debug(".env file already exists")
        return False

    # Check if .env.example exists
    if not env_example.exists():
        logger.warning(
            ".env.example not found. This may occur in pip-installed packages. "
            "You'll need to manually create a .env file or set environment variables."
        )
        return False

    # Copy .env.example to .env
    try:
        shutil.copy2(env_example, env_file)
        logger.info(f"Created .env file from .env.example at {env_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create .env file: {e}")
        raise


def run_database_migrations(database_url: str) -> tuple[bool, Optional[str]]:
    """
    Run Alembic database migrations to ensure schema is up to date.

    Args:
        database_url: SQLAlchemy database URL

    Returns:
        tuple: (success: bool, error_message: Optional[str])
    """
    try:
        from alembic import command
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine, text

        # Get alembic.ini location
        project_root = Path(__file__).parent.parent.parent
        alembic_ini = project_root / "alembic.ini"

        if not alembic_ini.exists():
            logger.warning("alembic.ini not found. Skipping migrations.")
            return False, "alembic.ini not found in project root"

        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        # Keep the application's logging: alembic/env.py skips fileConfig when this is False
        alembic_cfg.attributes["configure_logger"] = False

        # Check current migration version
        engine = create_engine(database_url)

        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current_rev = context.get_current_revision()

        # Get latest migration version
        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()

        # Check if migrations needed
        if current_rev == head_rev:
            logger.debug(f"Database already at latest migration: {head_rev}")
            return True, None

        # Run migrations
        logger.info(f"Running database migrations (current: {current_rev}, target: {head_rev})")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")

        return True, None

    except ImportError as e:
        error_msg = f"Alembic not available: {e}"
        logger.warning(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Migration failed: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def validate_database_schema(database_url: str) -> dict:
    """
    Validate database schema completeness.

    Args:
        database_url: SQLAlchemy database URL

    Returns:
        dict: Validation results with keys:
            - complete: bool
            - missing_tables: list[str]
            - missing_indexes: list[str]
            - current_revision: str
            - errors: list[str]
    """
    from sqlalchemy import create_engine, inspect, text
    from alembic.runtime.migration import MigrationContext

    results = {
        "complete": True,
        "missing_tables": [],
        "missing_indexes": [],
        "current_revision": None,
        "head_revision": None,
        "errors": []
    }

    try:
        engine = create_engine(database_url)
        inspector = inspect(engine)

        # Expected tables (from migrations)
        expected_tables = {
            "hypotheses",
            "experiments",
            "results",
            "papers",
            "agents",
            "research_sessions",
            "execution_profiles",       # From migration dc24ead48293 (profiling tables)
            "profiling_bottlenecks",   # From migration dc24ead48293 (profiling tables)
            "alembic_version"
        }

        # Check existing tables
        existing_tables = set(inspector.get_table_names())
        missing_tables = expected_tables - existing_tables

        if missing_tables:
            results["complete"] = False
            results["missing_tables"] = sorted(missing_tables)

        # Expected indexes (from migration 3) - uses idx_ prefix
        expected_indexes = {
            "hypotheses": ["idx_hypotheses_domain_status"],
            "experiments": ["idx_experiments_created_at", "idx_experiments_domain_status"],
            "results": ["idx_results_experiment_id"],
            "papers": ["idx_papers_domain_relevance"]
        }

        # Check indexes for each table
        for table_name, index_names in expected_indexes.items():
            if table_name in existing_tables:
                existing_indexes = {idx["name"] for idx in inspector.get_indexes(table_name)}
                missing = set(index_names) - existing_indexes
                if missing:
                    results["complete"] = False
                    results["missing_indexes"].extend([f"{table_name}.{idx}" for idx in missing])

        # Check migration version
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current_rev = context.get_current_revision()
            results["current_revision"] = current_rev

        # Get head revision
        try:
            from alembic.config import Config
            from alembic.script import ScriptDirectory
            project_root = Path(__file__).parent.parent.parent
            alembic_ini = project_root / "alembic.ini"

            if alembic_ini.exists():
                alembic_cfg = Config(str(alembic_ini))
                script = ScriptDirectory.from_config(alembic_cfg)
                head_rev = script.get_current_head()
                results["head_revision"] = head_rev

                if current_rev != head_rev:
                    results["complete"] = False
                    results["errors"].append(
                        f"Database migration outdated (current: {current_rev}, latest: {head_rev})"
                    )
        except Exception as e:
            results["errors"].append(f"Could not check migration version: {e}")

    except Exception as e:
        results["complete"] = False
        results["errors"].append(f"Schema validation failed: {e}")
        logger.error(f"Schema validation error: {e}", exc_info=True)

    return results


def first_time_setup(database_url: str) -> dict:
    """
    Perform first-time setup for pip-installed users.

    Includes:
    1. Creating .env file from .env.example
    2. Running database migrations
    3. Validating schema completeness

    Args:
        database_url: SQLAlchemy database URL

    Returns:
        dict: Setup results with keys:
            - env_created: bool
            - migrations_run: bool
            - schema_valid: bool
            - errors: list[str]
    """
    results = {
        "env_created": False,
        "migrations_run": False,
        "schema_valid": False,
        "errors": []
    }

    # Step 1: Ensure .env file exists
    try:
        results["env_created"] = ensure_env_file()
    except Exception as e:
        results["errors"].append(f"Failed to create .env: {e}")

    # Step 2: Run migrations
    success, error = run_database_migrations(database_url)
    results["migrations_run"] = success
    if error:
        results["errors"].append(error)

    # Step 3: Validate schema
    validation = validate_database_schema(database_url)
    results["schema_valid"] = validation["complete"]
    if not validation["complete"]:
        if validation["missing_tables"]:
            results["errors"].append(f"Missing tables: {', '.join(validation['missing_tables'])}")
        if validation["missing_indexes"]:
            results["errors"].append(f"Missing indexes: {', '.join(validation['missing_indexes'])}")
        results["errors"].extend(validation["errors"])

    return results

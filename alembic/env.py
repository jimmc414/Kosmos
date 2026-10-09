"""
Alembic environment configuration.

Connects to Kosmos database and manages migrations.
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import sys
from pathlib import Path

# Add kosmos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Kosmos models
from kosmos.db.models import Base
from kosmos.config import get_config

# Alembic Config object
config = context.config

# Interpret the config file for Python logging, only when alembic owns the process
# (the `alembic` CLI). kosmos.utils.setup runs migrations inside `kosmos run` and sets
# configure_logger=False: fileConfig would otherwise disable every existing kosmos logger.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Set target metadata for autogenerate
target_metadata = Base.metadata

def get_url():
    """Get database URL from Kosmos config."""
    try:
        kosmos_config = get_config()
        return kosmos_config.database.url
    except Exception:
        # Fallback to alembic.ini if config fails
        return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

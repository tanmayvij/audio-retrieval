"""Alembic migration environment for the application database."""

from logging.config import fileConfig

from alembic import context

from app.services.postgres import connect, database_url


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The schema is maintained by explicit Alembic migration operations.
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations without a live database connection."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the configured database."""
    with connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

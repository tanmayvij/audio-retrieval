"""PostgreSQL connection helpers."""

import psycopg
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

from app import config


def database_url() -> str:
    """Return a SQLAlchemy URL configured to use psycopg (v3)."""
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL must be configured before connecting to PostgreSQL.")

    if config.DATABASE_URL.startswith("postgresql://"):
        return config.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    if config.DATABASE_URL.startswith("postgres://"):
        return config.DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
    return config.DATABASE_URL


def connect() -> Connection:
    """Open and return a SQLAlchemy connection backed by psycopg."""
    engine = create_engine(database_url(), module=psycopg)
    return engine.connect()

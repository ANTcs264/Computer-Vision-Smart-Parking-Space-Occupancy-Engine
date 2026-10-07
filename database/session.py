"""
Database Session & Connection Configuration Module
Manages SQLAlchemy 2.0 Engine, Connection Pooling, and Session Factories
with automatic PostgreSQL connection and resilient SQLite local fallback.
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("database")

# Read database URL from .env, default to PostgreSQL local container
RAW_DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/smart_parking_db")

# Convert asyncpg URL to standard psycopg2 if needed for synchronous engine
if RAW_DB_URL.startswith("postgresql+asyncpg://"):
    SYNC_DB_URL = RAW_DB_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
else:
    SYNC_DB_URL = RAW_DB_URL

FALLBACK_SQLITE_URL = f"sqlite:///{PROJECT_ROOT}/smart_parking.db"


def get_engine():
    """
    Attempts to connect to PostgreSQL. If unavailable, falls back to local SQLite.
    Ensures developer is never blocked by an inactive PostgreSQL service.
    """
    try:
        if "postgresql" in SYNC_DB_URL:
            # Test PostgreSQL connectivity with a short timeout
            engine = create_engine(
                SYNC_DB_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 3}
            )
            with engine.connect() as conn:
                logger.info(f"Connected to PostgreSQL database: {SYNC_DB_URL.split('@')[-1]}")
                return engine
    except Exception as e:
        logger.warning(
            f"PostgreSQL connection failed ({e}). Falling back to local SQLite database: {FALLBACK_SQLITE_URL}"
        )

    # SQLite local engine
    sqlite_engine = create_engine(
        FALLBACK_SQLITE_URL,
        connect_args={"check_same_thread": False}
    )
    logger.info(f"Using SQLite database engine at: {FALLBACK_SQLITE_URL}")
    return sqlite_engine


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI & Service dependency for database session context management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

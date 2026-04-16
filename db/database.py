"""
db/database.py
Database engine, session, and schema-extraction helpers.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

LEGACY_DATABASE_URL = "sqlite:///./data/shop_accounts.db"


def _default_data_root() -> Path:
    """Prefer a stable local app-data folder on Windows over OneDrive-synced paths."""
    local_app_data = os.getenv("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "ShopOS"
    return Path("./data")


def _resolve_database_url() -> str:
    configured = os.getenv("DATABASE_URL", LEGACY_DATABASE_URL)
    if configured != LEGACY_DATABASE_URL:
        return configured

    db_path = (_default_data_root() / "shop_accounts.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _sqlite_file_path(database_url: str) -> Path | None:
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return Path(database_url[len(prefix):])
    return None


DATABASE_URL = _resolve_database_url()

# SQLite needs check_same_thread=False for multi-threaded Streamlit
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Yield a DB session (use in with-blocks)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    sqlite_path = _sqlite_file_path(DATABASE_URL)
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    from db.models import Base
    Base.metadata.create_all(bind=engine)


def get_table_schema() -> str:
    """
    Returns a text description of all tables + columns.
    Fed to the LLM as schema context for Text-to-SQL generation.
    """
    parts = []
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).fetchall()
        for (tbl,) in tables:
            if tbl.startswith("_"):
                continue
            cols = conn.execute(text(f"PRAGMA table_info({tbl})")).fetchall()
            col_str = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            parts.append(f"Table {tbl}: ({col_str})")
    return "\n".join(parts)

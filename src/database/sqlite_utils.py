import os
from pathlib import Path
from sqlalchemy import create_engine, text

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "cornerlab.db"


def get_database_url() -> str:
    return f"sqlite:///{DB_PATH}"


def init_database() -> None:
    os.makedirs(DB_PATH.parent, exist_ok=True)
    engine = create_engine(get_database_url())
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS project_status (key TEXT PRIMARY KEY, value TEXT NOT NULL)"))
        connection.execute(text("INSERT OR IGNORE INTO project_status (key, value) VALUES ('status', 'Sprint 1 scaffold ready')"))


def get_project_status() -> str:
    engine = create_engine(get_database_url())
    with engine.connect() as connection:
        result = connection.execute(text("SELECT value FROM project_status WHERE key = 'status'"))
        row = result.fetchone()
        return row[0] if row else "Unavailable"

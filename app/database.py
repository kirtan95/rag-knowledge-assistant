"""SQLAlchemy engine / session wiring (SQLAlchemy 2.0 style)."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session():
    """FastAPI dependency: yields a DB session, closes it afterwards."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

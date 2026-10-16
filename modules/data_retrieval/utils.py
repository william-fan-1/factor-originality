"""Utility functions for data retrieval."""
import wrds

_db: wrds.Connection | None = None


def connection() -> wrds.Connection:
    global _db
    if _db is None:
        _db = wrds.Connection()
    return _db


def reset_connection() -> None:
    """Close and discard the cached connection so the next call reconnects."""
    global _db
    if _db is not None:
        try:
            _db.close()
        except Exception:
            pass
    _db = None

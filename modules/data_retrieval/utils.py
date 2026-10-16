"""Utility functions for data retrieval."""
import wrds

_db: wrds.Connection | None = None


def connection() -> wrds.Connection:
    global _db
    if _db is None:
        _db = wrds.Connection()
    return _db
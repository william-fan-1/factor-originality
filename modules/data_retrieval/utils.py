"""Utility functions for data retrieval."""
import os

import wrds
from dotenv import load_dotenv

_db: wrds.Connection | None = None


def connection() -> wrds.Connection:
    """Return the shared WRDS connection, opening it on first use.

    Set ``WRDS_USERNAME`` in ``.env`` or the environment so the saved
    ``.pgpass`` entry is found; otherwise WRDS prompts with the OS username
    and asks for credentials again.
    """
    global _db
    if _db is None:
        load_dotenv()
        _db = wrds.Connection(wrds_username=os.getenv('WRDS_USERNAME'))
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

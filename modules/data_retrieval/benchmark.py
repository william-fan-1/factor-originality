"""Retrieve benchmark market returns and risk-free rates from CRSP via WRDS."""

from __future__ import annotations

from datetime import date

import pandas as pd

from modules.data_retrieval.utils import connection


def retrieve_benchmark(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> pd.DataFrame:
    """Retrieve monthly market and risk-free returns from CRSP and Fama-French.

    ``market_return`` is CRSP's value-weighted market return including
    dividends (``vwretd``). ``risk_free_rate`` is the Fama-French one-month
    T-bill return (``rf``) for the same month.

    Args:
        start_date (str | date | pd.Timestamp): Inclusive start date.
        end_date (str | date | pd.Timestamp): Inclusive end date.

    Returns:
        pd.DataFrame: Monthly observations dated on each month's last trading
            day, with date, market_return, and risk_free_rate as decimals.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError('start_date must be on or before end_date')

    # msi is dated on the last trading day of the month, so it is matched to
    # the Fama-French table on year and month rather than on the exact date.
    return connection().raw_sql(
        """
        SELECT i.date,
               i.vwretd AS market_return,
               f.rf     AS risk_free_rate
        FROM crsp.msi AS i
        LEFT JOIN ff.factors_monthly AS f
          ON date_trunc('month', f.date) = date_trunc('month', i.date)
        WHERE i.date BETWEEN %(start)s AND %(end)s
        ORDER BY i.date
        """,
        params={'start': start.date(), 'end': end.date()},
        date_cols=['date'],
    )


__all__ = ['retrieve_benchmark']

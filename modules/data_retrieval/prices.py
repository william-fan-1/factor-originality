"""Retrieve price-related data from CRSP via WRDS."""

from datetime import date

import pandas as pd
from modules.data_retrieval.utils import connection


def retrieve_prices(
    tickers: list[str],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    lookback_months: int = 13,
) -> pd.DataFrame:
    """Retrieve daily prices, volume, and shares outstanding from CRSP.

    Tickers are mapped to PERMNOs through ``crsp.dsenames`` using the name
    validity window, so a reused ticker only matches the security that held
    it on each date.

    Args:
        tickers (list[str]): CRSP ticker symbols to query.
        start_date (str | date | pd.Timestamp): Inclusive analysis start date.
        end_date (str | date | pd.Timestamp): Exclusive analysis end date.
        lookback_months (int): Calendar months of price history to retrieve before
            ``start_date``; 13 supports characteristics requiring 12-month history.

    Returns:
        pd.DataFrame: Daily observations with ticker, permno, date, close,
            adjusted_close (split-adjusted), volume, ret, and shares_outstanding.
    """
    if lookback_months < 0:
        raise ValueError('lookback_months must be non-negative')
    analysis_start = pd.Timestamp(start_date).normalize()
    retrieval_end = pd.Timestamp(end_date).normalize()
    if analysis_start >= retrieval_end:
        raise ValueError('start_date must be before end_date')
    retrieval_start = analysis_start - pd.DateOffset(months=lookback_months)

    prices = _connection().raw_sql(
        """
        SELECT n.ticker, d.permno, d.date,
               ABS(d.prc)            AS close,
               ABS(d.prc) / d.cfacpr AS adjusted_close,
               d.vol                 AS volume,
               d.ret,
               d.shrout * 1000       AS shares_outstanding
        FROM crsp.dsf AS d
        JOIN crsp.dsenames AS n
          ON d.permno = n.permno
         AND d.date BETWEEN n.namedt AND n.nameendt
        WHERE n.ticker IN %(tickers)s
          AND d.date >= %(start)s
          AND d.date <  %(end)s
        ORDER BY n.ticker, d.date
        """,
        params={
            'tickers': tuple(tickers),
            'start': retrieval_start.date(),
            'end': retrieval_end.date(),
        },
        date_cols=['date'],
    )
    return prices

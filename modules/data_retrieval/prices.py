"""Retrieve price-related data from CRSP via WRDS."""

from datetime import date

import pandas as pd
from modules.data_retrieval.utils import connection


def retrieve_prices(
    permnos: list[int],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    lookback_months: int = 13,
) -> pd.DataFrame:
    """Retrieve daily prices, volume, and shares outstanding from CRSP.

    Args:
        permnos (list[int]): CRSP PERMNOs to query.
        start_date (str | date | pd.Timestamp): Inclusive analysis start date.
        end_date (str | date | pd.Timestamp): Exclusive analysis end date.
        lookback_months (int): Calendar months of price history to retrieve before
            ``start_date``; 13 supports characteristics requiring 12-month history.

    Returns:
        pd.DataFrame: Daily observations with permno, date, close, adjusted_close
            (split-adjusted), volume, ret, and shares_outstanding.
    """
    if lookback_months < 0:
        raise ValueError('lookback_months must be non-negative')
    analysis_start = pd.Timestamp(start_date).normalize()
    retrieval_end = pd.Timestamp(end_date).normalize()
    if analysis_start >= retrieval_end:
        raise ValueError('start_date must be before end_date')
    retrieval_start = analysis_start - pd.DateOffset(months=lookback_months)

    prices = connection().raw_sql(
        """
        SELECT d.permno, d.date,
               ABS(d.prc)            AS close,
               ABS(d.prc) / d.cfacpr AS adjusted_close,
               d.vol                 AS volume,
               d.ret,
               d.shrout * 1000       AS shares_outstanding
        FROM crsp.dsf AS d
        WHERE d.permno IN %(permnos)s
          AND d.date >= %(start)s
          AND d.date <  %(end)s
        ORDER BY d.permno, d.date
        """,
        params={
            'permnos': tuple(int(permno) for permno in permnos),
            'start': retrieval_start.date(),
            'end': retrieval_end.date(),
        },
        date_cols=['date'],
    )
    prices['permno'] = prices['permno'].astype('int64')
    return prices

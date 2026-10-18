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

    ``adjusted_close`` is a total-return index: the split-adjusted price on a
    security's first date compounded by CRSP's daily holding-period return
    ``ret`` (which includes dividends). Price-based factors and returns computed
    from it therefore match the dividend-adjusted convention of yfinance's
    ``Adj Close``. ``split_adjusted_close`` keeps the price-only series.

    Args:
        permnos (list[int]): CRSP PERMNOs to query.
        start_date (str | date | pd.Timestamp): Inclusive analysis start date.
        end_date (str | date | pd.Timestamp): Exclusive analysis end date.
        lookback_months (int): Calendar months of price history to retrieve before
            ``start_date``; 13 supports characteristics requiring 12-month history.

    Returns:
        pd.DataFrame: Daily observations with permno, date, close,
            adjusted_close (total-return index), split_adjusted_close, volume,
            ret, and shares_outstanding.
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
               ABS(d.prc) / d.cfacpr AS split_adjusted_close,
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
    numeric = ['close', 'split_adjusted_close', 'volume', 'ret', 'shares_outstanding']
    prices[numeric] = prices[numeric].astype('float64')
    return prices.assign(adjusted_close=_total_return_index(prices))[
        ['permno', 'date', 'close', 'adjusted_close', 'split_adjusted_close',
         'volume', 'ret', 'shares_outstanding']
    ]


def _total_return_index(prices: pd.DataFrame) -> pd.Series:
    """Compound ``ret`` per PERMNO from the first split-adjusted price.

    A missing ``ret`` (a non-trading gap) contributes no change; CRSP's next
    reported return already spans the gap. The first observation has no prior
    price, so its return is ignored and the index starts at that price.
    """
    ordered = prices.sort_values(['permno', 'date'])
    first_row = ordered['permno'].ne(ordered['permno'].shift())
    growth = (
        (1 + ordered['ret'].fillna(0.0).where(~first_row, 0.0))
        .groupby(ordered['permno'])
        .cumprod()
    )
    base = ordered['split_adjusted_close'].groupby(ordered['permno']).transform('first')
    return (base * growth).reindex(prices.index)

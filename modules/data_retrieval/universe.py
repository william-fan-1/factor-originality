"""Retrieve S&P 500 membership from CRSP via WRDS."""

from __future__ import annotations

from datetime import date

import pandas as pd

from modules.data_retrieval.utils import connection


def retrieve_universe(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    min_member_days: int = 5,
) -> pd.DataFrame:
    """Return every security in the S&P 500 at any point in a date range.

    Membership comes from ``crsp.msp500list`` and is keyed by PERMNO, so it is
    unaffected by ticker changes, share classes, or reused tickers. Each
    security is labelled with the ticker it held at the end of its last
    membership spell inside the range. If two securities would share a label,
    the one that left the index last keeps it and the others get a
    ``_{permno}`` suffix so cache filenames stay unique.

    Spin-offs are often added to the index for a day or two to preserve index
    weights; securities whose membership spans fewer than ``min_member_days``
    are dropped unless membership runs to the end of the range.

    Args:
        start_date (str | date | pd.Timestamp): Inclusive range start.
        end_date (str | date | pd.Timestamp): Inclusive range end.
        min_member_days (int): Minimum calendar days between first and last
            membership date.

    Returns:
        pd.DataFrame: One row per PERMNO with ``permno, ticker, name`` and the
            first ``start`` and last ``end`` of membership clipped to the range.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError('start_date must be on or before end_date')

    spells = connection().raw_sql(
        """
        SELECT l.permno, l.start AS member_start, l.ending AS member_end,
               n.ticker, n.comnam AS name
        FROM crsp.msp500list AS l
        JOIN crsp.dsenames AS n
          ON n.permno = l.permno
         AND n.namedt   <= LEAST(l.ending, %(end)s::date)
         AND n.nameendt >= LEAST(l.ending, %(end)s::date)
        WHERE l.ending >= %(start)s::date
          AND l.start  <= %(end)s::date
        """,
        params={'start': start.date(), 'end': end.date()},
        date_cols=['member_start', 'member_end'],
    )
    if spells.empty:
        return pd.DataFrame(columns=['permno', 'ticker', 'name', 'start', 'end'])

    spells['permno'] = spells['permno'].astype('int64')
    spells['member_start'] = spells['member_start'].clip(lower=start)
    spells['member_end'] = spells['member_end'].clip(upper=end)
    spells = spells.sort_values(['permno', 'member_end'])

    last_spell = spells.groupby('permno').tail(1).set_index('permno')[['ticker', 'name']]
    universe = (
        spells.groupby('permno')
        .agg(start=('member_start', 'min'), end=('member_end', 'max'))
        .join(last_spell)
        .reset_index()
    )

    lasting = (universe['end'] - universe['start']).dt.days.ge(min_member_days)
    universe = universe.loc[lasting | universe['end'].ge(end)].copy()

    ticker = universe['ticker'].fillna('').astype(str).str.strip()
    universe['ticker'] = ticker.where(ticker.ne(''), 'P' + universe['permno'].astype(str))
    later_losers = universe.groupby('ticker')['end'].rank(method='first', ascending=False).gt(1)
    universe.loc[later_losers, 'ticker'] = (
        universe.loc[later_losers, 'ticker'] + '_' + universe.loc[later_losers, 'permno'].astype(str)
    )
    return universe[['permno', 'ticker', 'name', 'start', 'end']].sort_values('ticker').reset_index(drop=True)

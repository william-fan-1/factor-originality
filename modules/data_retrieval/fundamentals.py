"""Retrieve quarterly fundamental data from Compustat via WRDS."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from modules.data_retrieval.utils import connection

# Used only when Compustat has no report date (rdq). Set well past the 10-Q/10-K
# filing deadlines so an imputed date is never earlier than the real one.
_FALLBACK_REPORT_LAG_DAYS = 90
_LOG_PATH = Path(__file__).resolve().parents[2] / 'data' / 'logs' / 'imputed_filing_dates.csv'

# Compustat quarterly item -> canonical metric name
_QUARTERLY_ITEMS = {
    'saleq': 'revenue',
    'cogsq': 'cost_of_goods_sold',
    'oiadpq': 'operating_income',
    'niq': 'net_income',
    'atq': 'total_assets',
    'ltq': 'total_liabilities',
    'seqq': 'shareholders_equity',
    'cheq': 'cash_and_equivalents',
    'actq': 'current_assets',
    'lctq': 'current_liabilities',
    'ppentq': 'property_plant_equipment',
    'dlttq': 'long_term_debt',
    'dlcq': 'short_term_debt',
}

# Compustat cash-flow items are reported year-to-date within each fiscal year
_YTD_ITEMS = {
    'oancfy': 'operating_cash_flow',
    'capxy': 'capital_expenditures',
    'dpy': 'depreciation_amortization',
    'sstky': 'stock_issuance',
    'prstkcy': 'stock_repurchases',
}

METRICS = [
    'revenue', 'cost_of_goods_sold', 'gross_profit', 'operating_income', 'net_income',
    'total_assets', 'total_liabilities', 'shareholders_equity', 'cash_and_equivalents',
    'current_assets', 'current_liabilities', 'property_plant_equipment', 'long_term_debt',
    'short_term_debt', 'operating_cash_flow', 'capital_expenditures',
    'depreciation_amortization', 'stock_issuance', 'stock_repurchases',
]


def load_fundamentals(
    tickers: list[str],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    history_quarters: int = 5,
) -> pd.DataFrame:
    """Load discrete quarterly fundamentals and availability dates from Compustat.

    Tickers map to Compustat ``gvkey`` through the CRSP-Compustat link table
    (primary links only, valid on the fiscal period end) and CRSP ``dsenames``.

    Point-in-time handling:
        ``filing_date`` is the earnings report date (``rdq``) plus one day, since
        no acceptance time is available and many reports come after the close.
        Quarters with no ``rdq`` use period end plus 90 days, are flagged in
        ``filing_date_imputed``, and are appended to ``data/logs/imputed_filing_dates.csv``.
        Values are Compustat's current (restated) figures, not an as-first-reported
        snapshot.

    Args:
        tickers (list[str]): CRSP ticker symbols to query.
        start_date (str | date | pd.Timestamp): Inclusive analysis start date.
        end_date (str | date | pd.Timestamp): Inclusive analysis end date.
        history_quarters (int): Observations retained before ``start_date``.

    Returns:
        pd.DataFrame: ``ticker, permno, period_end, filing_date, filing_timestamp``
            ``filing_date_imputed`` plus every metric in ``METRICS``;
            ``filing_timestamp`` is always null.
    """
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError('start_date must be on or before end_date')
    if history_quarters < 1:
        raise ValueError('history_quarters must be at least 1')

    # Extra year of lookback so year-to-date items can be differenced.
    query_start = start - pd.DateOffset(months=3 * (history_quarters + 4))
    select_items = ', '.join(f'f.{item}' for item in [*_QUARTERLY_ITEMS, *_YTD_ITEMS])
    raw = connection().raw_sql(
        f"""
        SELECT n.ticker, l.lpermno AS permno, f.gvkey, f.datadate AS period_end,
               f.rdq, f.fyearq, f.fqtr, {select_items}
        FROM comp.fundq AS f
        JOIN crsp.ccmxpf_linktable AS l
          ON f.gvkey = l.gvkey
         AND l.linktype IN ('LC', 'LU')
         AND l.linkprim IN ('P', 'C')
         AND f.datadate >= l.linkdt
         AND (f.datadate <= l.linkenddt OR l.linkenddt IS NULL)
        JOIN crsp.dsenames AS n
          ON n.permno = l.lpermno
         AND f.datadate BETWEEN n.namedt AND n.nameendt
        WHERE n.ticker IN %(tickers)s
          AND f.datadate BETWEEN %(start)s AND %(end)s
          AND f.indfmt = 'INDL' AND f.datafmt = 'STD'
          AND f.popsrc = 'D' AND f.consol = 'C'
        """,
        params={
            'tickers': tuple(tickers),
            'start': query_start.date(),
            'end': end.date(),
        },
        date_cols=['period_end', 'rdq'],
    )
    columns = [
        'ticker', 'permno', 'period_end', 'filing_date', 'filing_timestamp',
        'filing_date_imputed', *METRICS,
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    raw = (
        raw.sort_values(['ticker', 'period_end', 'rdq'])
        .drop_duplicates(['ticker', 'period_end'], keep='last')
        .rename(columns={**_QUARTERLY_ITEMS})
        .reset_index(drop=True)
    )
    raw = _ytd_to_discrete(raw)
    raw = raw.rename(columns=_YTD_ITEMS)

    raw['gross_profit'] = raw['revenue'] - raw['cost_of_goods_sold']
    raw['shareholders_equity'] = raw['shareholders_equity'].fillna(
        raw['total_assets'] - raw['total_liabilities']
    )

    raw['filing_date_imputed'] = raw['rdq'].isna()
    report_date = raw['rdq'].fillna(
        raw['period_end'] + pd.Timedelta(days=_FALLBACK_REPORT_LAG_DAYS)
    )
    raw['filing_date'] = report_date.dt.normalize() + pd.Timedelta(days=1)
    raw['filing_timestamp'] = pd.NaT

    raw = raw.loc[raw['filing_date'].le(end)]
    parts = []
    for _, group in raw.groupby('ticker', sort=False):
        group = group.sort_values('period_end')
        history = group.loc[group['filing_date'].le(start)].tail(history_quarters)
        analysis = group.loc[group['filing_date'].gt(start)]
        parts.append(pd.concat([history, analysis]))
    result = pd.concat(parts, ignore_index=True) if parts else raw.iloc[0:0]
    result = (
        result.reindex(columns=columns)
        .sort_values(['ticker', 'period_end'])
        .reset_index(drop=True)
    )
    _log_imputed_filing_dates(result)
    return result


def _log_imputed_filing_dates(result: pd.DataFrame) -> None:
    """Append quarters whose filing date was imputed to ``data/logs``."""
    imputed = result.loc[
        result['filing_date_imputed'].astype(bool),
        ['ticker', 'permno', 'period_end', 'filing_date'],
    ]
    if imputed.empty:
        return
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    imputed.to_csv(
        _LOG_PATH, mode='a', index=False, header=not _LOG_PATH.exists()
    )


def _ytd_to_discrete(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert year-to-date cash-flow items to discrete quarters.

    A quarter is differenced only against the immediately preceding fiscal
    quarter of the same year; if that quarter is missing the value is ``NaN``
    rather than a multi-quarter total.
    """
    frame = frame.sort_values(['ticker', 'fyearq', 'fqtr']).copy()
    group = frame.groupby(['ticker', 'fyearq'], sort=False)
    consecutive = group['fqtr'].diff().eq(1)
    first_quarter = frame['fqtr'].eq(1)
    for item in _YTD_ITEMS:
        discrete = group[item].diff()
        frame[item] = np.where(first_quarter, frame[item], discrete.where(consecutive))
    return frame

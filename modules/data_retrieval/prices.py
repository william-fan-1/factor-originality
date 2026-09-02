"""Retrieve price-related data."""

import os
from datetime import date

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from edgar import Company, set_identity

def retrieve_prices(
    ticker: str,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    lookback_months: int = 13,
) -> pd.DataFrame:
    """Retrieve daily prices, volume, and shares outstanding for a ticker.

    Args:
        ticker (str): Yahoo Finance ticker symbol to query.
        start_date (str | date | pd.Timestamp): Inclusive analysis start date.
        end_date (str | date | pd.Timestamp): Exclusive analysis end date.
        lookback_months (int): Calendar months of price history to retrieve before
            ``start_date``; 13 supports characteristics requiring 12-month history.

    Returns:
        pd.DataFrame: Daily observations from the warm-up start through the exclusive
            end date with ticker, prices, volume, and shares outstanding.
    """
    if lookback_months < 0:
        raise ValueError('lookback_months must be non-negative')
    analysis_start = pd.Timestamp(start_date).normalize()
    retrieval_end = pd.Timestamp(end_date).normalize()
    if analysis_start >= retrieval_end:
        raise ValueError('start_date must be before end_date')
    retrieval_start = analysis_start - pd.DateOffset(months=lookback_months)

    security = yf.Ticker(ticker)
    history = security.history(
        start=retrieval_start,
        end=retrieval_end,
        auto_adjust=False,
        actions=False,
    )

    result = history.rename(
        columns={'Close': 'close', 'Adj Close': 'adjusted_close', 'Volume': 'volume'}
    )[['close', 'adjusted_close', 'volume']].copy()

    result.index = pd.to_datetime(result.index).tz_localize(None)
    result.index = result.index.normalize()
    result['shares_outstanding'] = _get_shares_outstanding(
        security,
        result.index,
    )
    result.insert(0, 'date', result.index)
    result.insert(0, 'ticker', ticker)
    return result.reset_index(drop=True)

def _get_shares_outstanding(
    security: yf.Ticker,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Align yfinance share-count data with the retrieved trading dates.

    Args:
        security (yf.Ticker): Yahoo Finance ticker object to query.
        dates (pd.DatetimeIndex): Trading dates to which share counts are aligned.

    Returns:
        pd.Series: Shares outstanding indexed by the requested trading dates.
    """
    if dates.empty:
        return pd.Series(index=dates, dtype='float64')

    yahoo_shares = security.get_shares_full(
        start=dates.min() - pd.Timedelta(days=365),
        end=dates.max() + pd.Timedelta(days=1),
    )
    if yahoo_shares is None:
        yahoo_shares = _empty_share_series()
    else:
        yahoo_shares = _normalize_share_series(yahoo_shares)

    yahoo_aligned = yahoo_shares.reindex(dates, method='ffill')
    if yahoo_aligned.notna().all():
        return yahoo_aligned

    sec_shares = _get_sec_shares_outstanding(
        ticker=security.ticker,
        start_date=dates.min() - pd.Timedelta(days=365),
        end_date=dates.max(),
    )
    # SEC fills gaps in Yahoo's historical coverage; Yahoo wins when both
    # sources report an observation on the same date.
    historical_shares = pd.concat([sec_shares, yahoo_shares]).sort_index()
    historical_shares = historical_shares[
        ~historical_shares.index.duplicated(keep='last')
    ]
    return historical_shares.reindex(dates, method='ffill')

def _get_sec_shares_outstanding(
    ticker: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.Series:
    """Retrieve historical shares outstanding from SEC company XBRL facts.

    Args:
        ticker (str): SEC ticker symbol to query.
        start_date (str | pd.Timestamp): Inclusive start date for XBRL observations.
        end_date (str | pd.Timestamp): Inclusive end date for XBRL observations.

    Returns:
        pd.Series: Historical SEC share counts indexed by their reported instant dates.
    """
    load_dotenv()
    identity = os.getenv('EDGAR_IDENTITY') or os.getenv('IDENTITY')
    if identity:
        set_identity(identity)

    entity_facts = Company(ticker).get_facts()
    if entity_facts is None:
        return _empty_share_series()
    facts = entity_facts.to_dataframe(pit_mode=True)
    if facts.empty:
        return _empty_share_series()

    concepts = facts['concept'].astype(str)
    dei = facts.loc[concepts.eq('dei:EntityCommonStockSharesOutstanding')].copy()
    if dei.empty:
        dei = facts.loc[concepts.eq('us-gaap:CommonStockSharesOutstanding')].copy()
    if dei.empty:
        return _empty_share_series()

    dei['period_end'] = pd.to_datetime(dei['period_end'], errors='coerce').dt.normalize()
    dei['numeric_value'] = pd.to_numeric(dei['numeric_value'], errors='coerce')
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    dei = dei.loc[
        dei['period_end'].between(start, end)
        & dei['numeric_value'].notna()
        & dei['numeric_value'].gt(0)
    ]
    if dei.empty:
        return _empty_share_series()
    if 'filing_date' in dei:
        dei['filing_date'] = pd.to_datetime(dei['filing_date'], errors='coerce')
        dei = dei.sort_values(['period_end', 'filing_date'])
    series = dei.drop_duplicates('period_end', keep='last').set_index('period_end')['numeric_value']
    return _normalize_share_series(series)

def _normalize_share_series(shares: pd.Series) -> pd.Series:
    """Normalize a historical share-count series to unique timezone-naive dates.

    Args:
        shares (pd.Series): Share counts indexed by date-like values.

    Returns:
        pd.Series: Numeric share counts indexed by sorted normalized dates.
    """
    if shares.empty:
        return _empty_share_series()
    result = shares.copy()
    index = pd.to_datetime(result.index, errors='coerce', utc=True)
    result.index = index.tz_convert(None).normalize()
    result = pd.to_numeric(result, errors='coerce').dropna().sort_index()
    return result[~result.index.duplicated(keep='last')]

def _empty_share_series() -> pd.Series:
    """Create an empty share-count series with a date-compatible index.

    Args:
        None.

    Returns:
        pd.Series: Empty floating-point series with a ``DatetimeIndex``.
    """
    return pd.Series(dtype='float64', index=pd.DatetimeIndex([], name='date'))

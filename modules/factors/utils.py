"""Load and transform cached data for factor construction."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw_data'


def load_fundamental_data(
    tickers: Sequence[str],
    year: int,
) -> pd.DataFrame:
    """Load fundamentals.

    Args:
        tickers (Sequence[str]): Tickers whose cached fundamentals should be loaded.
        year (int): First year of the five-year cache period.

    Returns:
        pd.DataFrame: Long-form quarterly fundamentals sorted by ticker and period end.
    """
    data = _load_cached_frames(tickers, year, 'fundamentals')
    if data.empty:
        return data

    data['period_end'] = pd.to_datetime(
        data['period_end'], errors='raise'
    ).astype('datetime64[ns]')
    data['filing_date'] = pd.to_datetime(
        data['filing_date'], errors='raise'
    ).astype('datetime64[ns]')
    if 'filing_timestamp' in data:
        data['filing_timestamp'] = pd.to_datetime(
            data['filing_timestamp'], errors='coerce', utc=True
        )
    return data.sort_values(['ticker', 'period_end', 'filing_date']).reset_index(drop=True)

def load_price_data(
    tickers: Sequence[str],
    year: int,
) -> pd.DataFrame:
    """Load prices.

    Args:
        tickers (Sequence[str]): Tickers whose cached prices should be loaded.
        year (int): First year of the five-year cache period.

    Returns:
        pd.DataFrame: Long-form daily prices with market caps and stock returns.
    """
    data = _load_cached_frames(tickers, year, 'prices')
    if data.empty:
        return data

    data['date'] = pd.to_datetime(
        data['date'], errors='raise'
    ).astype('datetime64[ns]')
    data = data.sort_values(['ticker', 'date']).reset_index(drop=True)
    return _process_data(data)

def align_data(
    fundamental_data: pd.DataFrame,
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """Align fundamentals.

    Args:
        fundamental_data (pd.DataFrame): Quarterly data containing ``ticker``,
            ``period_end``, ``filing_date``, and the exact SEC acceptance time in
            ``filing_timestamp`` when available.
        price_data (pd.DataFrame): Daily data containing ``ticker`` and ``date``.

    Returns:
        pd.DataFrame: Daily price data matched to the latest available filing for
            each ticker, including its effective ``available_date`` and leaving
            pre-filing fundamental values as ``NaN``.
    """
    require_columns(
        fundamental_data,
        {'ticker', 'period_end', 'filing_date', 'filing_timestamp'},
    )
    require_columns(price_data, {'ticker', 'date'})
    if price_data.empty or fundamental_data.empty:
        return price_data.copy()

    fundamentals = fundamental_data.copy()
    prices = price_data.copy()
    fundamentals['period_end'] = pd.to_datetime(
        fundamentals['period_end'], errors='raise'
    ).astype('datetime64[ns]')
    fundamentals['filing_date'] = pd.to_datetime(
        fundamentals['filing_date'], errors='raise'
    ).astype('datetime64[ns]')
    fundamentals['filing_timestamp'] = pd.to_datetime(
        fundamentals['filing_timestamp'], errors='coerce', utc=True
    )
    prices['date'] = pd.to_datetime(
        prices['date'], errors='raise'
    ).astype('datetime64[ns]')
    fundamentals['available_date'] = _effective_filing_dates(
        fundamentals=fundamentals,
        prices=prices,
    )
    fundamentals = fundamentals.dropna(subset=['available_date'])

    # If a later comparative filing supplies an older missing quarter, prefer
    # the most recent fiscal period that became available on that filing date.
    fundamentals = (
        fundamentals
        .sort_values(
            ['ticker', 'available_date', 'filing_timestamp', 'period_end'],
            na_position='first',
        )
        .drop_duplicates(['ticker', 'available_date'], keep='last')
        .sort_values(['available_date', 'ticker'])
    )
    prices = prices.sort_values(['date', 'ticker'])

    aligned_data = pd.merge_asof(
        prices,
        fundamentals,
        by='ticker',
        left_on='date',
        right_on='available_date',
        direction='backward',
        allow_exact_matches=True,
    )
    return aligned_data.sort_values(['ticker', 'date']).reset_index(drop=True)

def _effective_filing_dates(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Series:
    """Assign filing availability dates.

    Args:
        fundamentals (pd.DataFrame): Accounting observations with filing dates and
            optional UTC acceptance timestamps.
        prices (pd.DataFrame): Daily security observations defining trading dates.

    Returns:
        pd.Series: First eligible trading date for every accounting observation.
    """
    timestamps = fundamentals['filing_timestamp'].dt.tz_convert(
        'America/New_York'
    )
    after_close = timestamps.notna() & (
        timestamps.dt.hour.mul(60).add(timestamps.dt.minute) > 16 * 60
    )
    candidate_dates = fundamentals['filing_date'].dt.normalize().where(
        timestamps.isna(),
        timestamps.dt.tz_localize(None).dt.normalize(),
    )
    candidate_dates = candidate_dates + pd.to_timedelta(
        after_close.astype('int64'), unit='D'
    )

    available = pd.Series(pd.NaT, index=fundamentals.index, dtype='datetime64[ns]')
    trading_dates = {
        ticker: pd.DatetimeIndex(group['date'].dropna().unique()).sort_values()
        for ticker, group in prices.groupby('ticker', sort=False)
    }
    for ticker, group in fundamentals.groupby('ticker', sort=False):
        dates = trading_dates.get(ticker, pd.DatetimeIndex([]))
        if dates.empty:
            continue
        targets = pd.DatetimeIndex(candidate_dates.loc[group.index])
        positions = dates.searchsorted(targets, side='left')
        valid = positions < len(dates)
        values = np.full(len(group), np.datetime64('NaT'), dtype='datetime64[ns]')
        values[valid] = dates.to_numpy()[positions[valid]]
        available.loc[group.index] = values
    return available

def calculate_market_cap(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate capitalization.

    Args:
        data (pd.DataFrame): Price data containing ``close`` and
            ``shares_outstanding`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a ``market_cap`` column.
    """
    require_columns(data, {'close', 'shares_outstanding'})
    result = data.copy()
    close = pd.to_numeric(result['close'], errors='coerce')
    shares = pd.to_numeric(result['shares_outstanding'], errors='coerce')
    result['market_cap'] = close * shares
    return result

def winsorize(
    data: pd.DataFrame,
    percentile: int | float = 80,
) -> pd.DataFrame:
    """Winsorize capitalization.

    Args:
        data (pd.DataFrame): Data containing ``date`` and ``market_cap`` columns.
        percentile (int | float): Cross-sectional percentile from 0 through 100
            at which market capitalization is capped.

    Returns:
        pd.DataFrame: Copy of the data with a ``winsorized_market_cap`` column.
    """
    require_columns(data, {'date', 'market_cap'})
    if not 0 <= percentile <= 100:
        raise ValueError('percentile must be between 0 and 100')

    result = data.copy()
    caps = pd.to_numeric(result['market_cap'], errors='coerce')
    upper_bounds = caps.groupby(result['date']).transform(
        lambda values: values.quantile(percentile / 100)
    )
    result['winsorized_market_cap'] = caps.clip(upper=upper_bounds)
    return result

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns.

    Args:
        data (pd.DataFrame): Price data containing ``ticker``, ``date``, and
            ``adjusted_close`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a ticker-level daily ``return`` column.
    """
    require_columns(data, {'ticker', 'date', 'adjusted_close'})
    result = data.sort_values(['ticker', 'date']).copy()
    prices = pd.to_numeric(result['adjusted_close'], errors='coerce')
    result['return'] = prices.groupby(result['ticker'], sort=False).pct_change(
        fill_method=None
    )
    return result

def _process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process prices.

    Args:
        data (pd.DataFrame): Concatenated long-form daily price data.

    Returns:
        pd.DataFrame: Price data with market caps, winsorized market caps, and returns.
    """
    data = calculate_market_cap(data)
    data = winsorize(data)
    return calculate_returns(data)

def _load_cached_frames(
    tickers: Sequence[str],
    year: int,
    dataset: str,
) -> pd.DataFrame:
    """Concatenate caches.

    Args:
        tickers (Sequence[str]): Tickers whose cache files should be loaded.
        year (int): First year of the five-year cache period.
        dataset (str): Cache suffix, either ``'prices'`` or ``'fundamentals'``.

    Returns:
        pd.DataFrame: Vertically concatenated cache contents.
    """
    if dataset not in {'prices', 'fundamentals'}:
        raise ValueError("dataset must be either 'prices' or 'fundamentals'")

    cache_dir = RAW_DATA_DIR / f'{year}_{year + 4}'
    paths = [cache_dir / f'{ticker}_{dataset}.parquet' for ticker in tickers]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ', '.join(path.name for path in missing[:10])
        remainder = len(missing) - 10
        suffix = f' (and {remainder} more)' if remainder > 0 else ''
        raise FileNotFoundError(f'Missing {dataset} caches: {names}{suffix}')
    if not paths:
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
        frames = executor.map(pd.read_parquet, paths)
        return pd.concat(frames, ignore_index=True)

def drop_invalid_factor(data: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Drop invalid observations.

    Args:
        data (pd.DataFrame): Data containing the calculated factor column.
        factor (str): Factor column whose invalid observations should be removed.

    Returns:
        pd.DataFrame: Data containing only finite, non-null factor observations.
    """
    require_columns(data, {factor})
    result = data.copy()
    result[factor] = result[factor].replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=[factor]).reset_index(drop=True)

def require_columns(data: pd.DataFrame, columns: set[str]) -> None:
    """Validate columns.

    Args:
        data (pd.DataFrame): DataFrame whose schema should be checked.
        columns (set[str]): Required column names.

    Returns:
        None.
    """
    missing = columns.difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {", ".join(sorted(missing))}')

__all__ = [
    'align_data',
    'calculate_market_cap',
    'calculate_returns',
    'drop_invalid_factor',
    'load_fundamental_data',
    'load_price_data',
    'require_columns',
    'winsorize',
]

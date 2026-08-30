"""Load and transform cached data for factor construction."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

    data['period_end'] = pd.to_datetime(data['period_end'], errors='raise')
    data['filing_date'] = pd.to_datetime(data['filing_date'], errors='raise')
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

    data['date'] = pd.to_datetime(data['date'], errors='raise')
    data = data.sort_values(['ticker', 'date']).reset_index(drop=True)
    return _process_data(data)


def calculate_market_cap(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate capitalization.

    Args:
        data (pd.DataFrame): Price data containing ``close`` and
            ``shares_outstanding`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a ``market_cap`` column.
    """
    _require_columns(data, {'close', 'shares_outstanding'})
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
    _require_columns(data, {'date', 'market_cap'})
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
    _require_columns(data, {'ticker', 'date', 'adjusted_close'})
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


def _require_columns(data: pd.DataFrame, columns: set[str]) -> None:
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
    'calculate_market_cap',
    'calculate_returns',
    'load_fundamental_data',
    'load_price_data',
    'winsorize',
]

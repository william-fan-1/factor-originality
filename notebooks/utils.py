"""Provide reusable data preparation helpers for factor-analysis notebooks."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from modules.factors.factor import Factor
from modules.factors.mappings import FACTOR_MAPPINGS
from modules.factors.utils import align_data, load_fundamental_data, load_price_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'


def load_tickers(year: int) -> list[str]:
    """Load the unique ticker universe for a cohort year.

    Args:
        year (int): First year of the five-year cohort period.

    Returns:
        list[str]: Tickers in source-file order with duplicates removed.
    """
    path = DATA_DIR / 'tickers' / f'tickers_{year}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing ticker universe: {path}')
    tickers = pd.read_csv(path)['Ticker'].dropna().astype(str)
    return list(dict.fromkeys(tickers))


def load_aligned_factor_data(year: int) -> pd.DataFrame:
    """Load and point-in-time align cached price and fundamental data for a cohort.

    Args:
        year (int): First year of the five-year cache period.

    Returns:
        pd.DataFrame: Daily stock panel containing prices and aligned fundamentals.
    """
    tickers = load_tickers(year)
    fundamentals = load_fundamental_data(tickers=tickers, year=year)
    prices = load_price_data(tickers=tickers, year=year)
    return align_data(fundamental_data=fundamentals, price_data=prices)


def calculate_factor_return_data(
    data: pd.DataFrame,
    year: int,
    factors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate and merge daily portfolio returns for selected registered factors.

    Args:
        data (pd.DataFrame): Daily stock panel accepted by the registered factors.
        year (int): First year of the factor-analysis period.
        factors (Sequence[str] | None): Factor names to calculate, or ``None`` for all.

    Returns:
        pd.DataFrame: Date-indexed panel with one factor-return column per factor.
    """
    selected = list(FACTOR_MAPPINGS) if factors is None else list(dict.fromkeys(factors))
    unknown = set(selected).difference(FACTOR_MAPPINGS)
    if unknown:
        names = ', '.join(sorted(unknown))
        raise ValueError(f'Unknown factors: {names}')
    if not selected:
        return pd.DataFrame(columns=['date'])

    return_series: list[pd.Series] = []
    for name in selected:
        factor_returns = Factor(factor=name, year=year, data=data).data
        series = factor_returns.set_index('date')['factor_return'].rename(name)
        return_series.append(series)

    return (
        pd.concat(return_series, axis=1, join='outer')
        .sort_index()
        .rename_axis('date')
        .reset_index()
        .loc[lambda frame: frame['date'].ge(pd.Timestamp(year=year, month=1, day=1))]
        .reset_index(drop=True)
    )


def concatenate_factor_return_periods(
    periods: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate non-overlapping factor-return periods into one time series.

    Args:
        periods (Sequence[pd.DataFrame]): Factor-return panels containing a unique
            ``date`` column and one or more factor columns.

    Returns:
        pd.DataFrame: Chronologically sorted factor returns spanning every period.
    """
    if not periods:
        return pd.DataFrame(columns=['date'])

    frames: list[pd.DataFrame] = []
    for index, period in enumerate(periods):
        if 'date' not in period:
            raise ValueError(f'Factor-return period {index} is missing date')
        frame = period.copy()
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        if frame['date'].duplicated().any():
            raise ValueError(f'Factor-return period {index} contains duplicate dates')
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_dates = result.loc[result['date'].duplicated(keep=False), 'date']
    if not duplicate_dates.empty:
        examples = ', '.join(
            timestamp.strftime('%Y-%m-%d')
            for timestamp in duplicate_dates.drop_duplicates().sort_values().head(5)
        )
        raise ValueError(f'Factor-return periods overlap on dates including {examples}')
    return result.sort_values('date').reset_index(drop=True)


__all__ = [
    'calculate_factor_return_data',
    'concatenate_factor_return_periods',
    'load_aligned_factor_data',
    'load_tickers',
]

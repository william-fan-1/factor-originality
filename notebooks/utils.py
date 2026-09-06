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


__all__ = [
    'calculate_factor_return_data',
    'load_aligned_factor_data',
    'load_tickers',
]

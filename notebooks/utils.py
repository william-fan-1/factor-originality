"""Provide reusable data preparation helpers for factor-analysis notebooks."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from modules.factors.factor import Factor
from modules.factors.mappings import FACTOR_MAPPINGS
from modules.factors.utils import (
    add_index_membership,
    align_data,
    load_fundamental_data,
    load_price_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'


def load_universe(start_year: int, end_year: int) -> pd.DataFrame:
    """Load the S&P 500 membership table built by ``scripts/build_crsp_universe.py``.

    Args:
        start_year (int): First year of the period.
        end_year (int): Last year of the period.

    Returns:
        pd.DataFrame: One row per security with ``permno``, ``ticker``, ``name``,
            ``start``, and ``end``.
    """
    path = DATA_DIR / 'sp500_universe' / f'crsp_universe_{start_year}_{end_year}.parquet'
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing universe: {path}\n'
            f'Run: python scripts/build_crsp_universe.py --start {start_year} --end {end_year}'
        )
    return pd.read_parquet(path)


def load_aligned_factor_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Load cached prices and fundamentals, align them point-in-time, and flag membership.

    Args:
        start_year (int): First year of the cache period.
        end_year (int): Last year of the cache period.

    Returns:
        pd.DataFrame: Daily stock panel with prices, aligned fundamentals, and an
            ``in_index`` column marking dates on which each stock was an index member.
    """
    universe = load_universe(start_year, end_year)
    tickers = universe['ticker'].tolist()
    fundamentals = load_fundamental_data(tickers, start_year, end_year)
    prices = load_price_data(tickers, start_year, end_year)
    aligned = align_data(fundamental_data=fundamentals, price_data=prices)
    return add_index_membership(aligned, universe)


def calculate_factor_return_data(
    data: pd.DataFrame,
    start_year: int,
    factors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate and merge daily portfolio returns for selected registered factors.

    Args:
        data (pd.DataFrame): Daily stock panel accepted by the registered factors.
        start_year (int): First year of the analysis; earlier rows only supply
            warm-up history and are dropped from the returns.
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
        factor_returns = Factor(factor=name, data=data).data
        series = factor_returns.set_index('date')['factor_return'].rename(name)
        return_series.append(series)

    return (
        pd.concat(return_series, axis=1, join='outer')
        .sort_index()
        .rename_axis('date')
        .reset_index()
        .loc[lambda frame: frame['date'].ge(pd.Timestamp(year=start_year, month=1, day=1))]
        .reset_index(drop=True)
    )


__all__ = [
    'calculate_factor_return_data',
    'load_aligned_factor_data',
    'load_universe',
]

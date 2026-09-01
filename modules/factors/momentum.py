"""Implement momentum factors."""

import numpy as np
import pandas as pd

from modules.factors.utils import drop_invalid_factor, require_columns


def twelve_one_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical 12-1 momentum factor.

    Calculates the return over the previous 12 months while omitting the most
    recent month:

    ``MOM = RI*(t - 1) / RI*(t - 12) - 1``

    Args:
        data (pd.DataFrame): Daily price data containing ``ticker``, ``date``,
            and ``adjusted_close`` columns.

    Returns:
        pd.DataFrame: Data sorted by ticker and date with non-null ``mom_12_1`` values.
    """
    return _calculate_momentum(
        data=data,
        numerator_months=1,
        denominator_months=12,
        factor='mom_12_1',
    )

def six_zero_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical six-month momentum factor.

    Calculates the return over the previous six months through the current date:

    ``MOM = RI*(t) / RI*(t - 6) - 1``

    Args:
        data (pd.DataFrame): Daily price data containing ``ticker``, ``date``,
            and ``adjusted_close`` columns.

    Returns:
        pd.DataFrame: Data sorted by ticker and date with non-null ``mom_6_0`` values.
    """
    return _calculate_momentum(
        data=data,
        numerator_months=0,
        denominator_months=6,
        factor='mom_6_0',
    )

def short_term_reversal(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical short-term reversal factor.

    Calculates the return over the previous month for a strategy that sells recent
    winners and buys recent losers:

    ``MOM = RI*(t) / RI*(t - 1) - 1``

    Args:
        data (pd.DataFrame): Daily price data containing ``ticker``, ``date``,
            and ``adjusted_close`` columns.

    Returns:
        pd.DataFrame: Data sorted by ticker and date with non-null ``st_rev`` values.
    """
    return _calculate_momentum(
        data=data,
        numerator_months=0,
        denominator_months=1,
        factor='st_rev',
    )

def _calculate_momentum(
    data: pd.DataFrame,
    numerator_months: int,
    denominator_months: int,
    factor: str,
) -> pd.DataFrame:
    """Calculate momentum.

    Args:
        data (pd.DataFrame): Daily price data containing ticker, date, and price.
        numerator_months (int): Calendar months by which to lag the numerator.
        denominator_months (int): Calendar months by which to lag the denominator.
        factor (str): Name of the resulting factor column.

    Returns:
        pd.DataFrame: Data with finite, non-null values for the requested factor.
    """
    require_columns(data, {'ticker', 'date', 'adjusted_close'})
    result = data.copy()
    result['date'] = pd.to_datetime(result['date'], errors='raise')
    result['adjusted_close'] = pd.to_numeric(
        result['adjusted_close'], errors='coerce'
    )
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

    numerator = _lagged_prices(result, numerator_months)
    denominator = _lagged_prices(result, denominator_months)
    result[factor] = numerator / denominator - 1
    return drop_invalid_factor(result, factor)

def _lagged_prices(data: pd.DataFrame, months: int) -> pd.Series:
    """Find lagged prices.

    Args:
        data (pd.DataFrame): Ticker-sorted data containing dates and adjusted prices.
        months (int): Number of calendar months by which observations are lagged.

    Returns:
        pd.Series: Prices from the latest trading date on or before each lagged date.
    """
    lagged = pd.Series(np.nan, index=data.index, dtype='float64')
    offset = pd.DateOffset(months=months)
    for _, group in data.groupby('ticker', sort=False):
        dates = pd.DatetimeIndex(group['date'])
        target_dates = dates - offset
        positions = dates.searchsorted(target_dates, side='right') - 1
        valid = positions >= 0
        values = np.full(len(group), np.nan, dtype='float64')
        prices = group['adjusted_close'].to_numpy(dtype='float64')
        values[valid] = prices[positions[valid]]
        lagged.loc[group.index] = values
    return lagged

__all__ = ['short_term_reversal', 'six_zero_momentum', 'twelve_one_momentum']

"""Implement momentum factors."""

import numpy as np
import pandas as pd

from modules.factors.utils import drop_invalid_factor, require_columns


def twelve_one_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate price momentum from t-12 months to t-1 month.

    Construction:
        ``MOM = RI(t-1) / RI(t-12) - 1``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        Prices are selected from the latest trading date on or before each
        calendar-month lag, and the most recent month is omitted.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``mom_12_1`` values.
    """
    return _calculate_momentum(data, 1, 12, 'mom_12_1')


def six_zero_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate price momentum from t-6 months through the measurement date.

    Construction:
        ``MOM = RI(t) / RI(t-6) - 1``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        Prices are selected from the latest trading date on or before each
        calendar-month lag.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``mom_6_0`` values.
    """
    return _calculate_momentum(data, 0, 6, 'mom_6_0')


def short_term_reversal(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the prior-month return used as a short-term reversal signal.

    Construction:
        ``STREV = RI(t) / RI(t-1) - 1``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        Prices are selected from the latest trading date on or before the
        one-month calendar lag.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``st_rev`` values.
    """
    return _calculate_momentum(data, 0, 1, 'st_rev')


def price_to_high_252d(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate current price relative to its trailing 252-trading-day high.

    Construction:
        ``Price to high = RI(t) / max(RI(t-251), ..., RI(t))``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        The rolling maximum includes only the current and preceding 251 trading
        observations and requires the complete 252-day window.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``prc_high_252`` values.
    """
    require_columns(data, {'ticker', 'date', 'adjusted_close'})
    result = _prepare_prices(data)
    trailing_high = (
        result.groupby('ticker', sort=False)['adjusted_close']
        .rolling(window=252, min_periods=252)
        .max()
        .reset_index(level=0, drop=True)
    )
    result['prc_high_252'] = result['adjusted_close'] / trailing_high
    return drop_invalid_factor(result, 'prc_high_252')


def three_one_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate price momentum from t-3 months to t-1 month.

    Construction:
        ``MOM = RI(t-1) / RI(t-3) - 1``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        Prices are selected from the latest trading date on or before each
        calendar-month lag, and the most recent month is omitted.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``mom_3_1`` values.
    """
    return _calculate_momentum(data, 1, 3, 'mom_3_1')


def six_one_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate price momentum from t-6 months to t-1 month.

    Construction:
        ``MOM = RI(t-1) / RI(t-6) - 1``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        Prices are selected from the latest trading date on or before each
        calendar-month lag, and the most recent month is omitted.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``mom_6_1`` values.
    """
    return _calculate_momentum(data, 1, 6, 'mom_6_1')


def size_one_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate six-one momentum through its legacy misspelled public name.

    Construction:
        ``MOM = RI(t-1) / RI(t-6) - 1``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        This compatibility wrapper delegates to ``six_one_momentum``.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``mom_6_1`` values.
    """
    return six_one_momentum(data)


def nine_one_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate price momentum from t-9 months to t-1 month.

    Construction:
        ``MOM = RI(t-1) / RI(t-9) - 1``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        Prices are selected from the latest trading date on or before each
        calendar-month lag, and the most recent month is omitted.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``mom_9_1`` values.
    """
    return _calculate_momentum(data, 1, 9, 'mom_9_1')


def eleven_one_seasonality_nonannual(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate average nonannual seasonality over monthly lags one through eleven.

    Construction:
        ``Seasonality = mean(RI(t-k) / RI(t-k-1) - 1 for k=1,...,11)``

    Required cached columns:
        ``ticker``, ``date``, and ``adjusted_close``.

    Point-in-time handling:
        Each price uses the latest trading date on or before its calendar-month
        lag, and all eleven monthly returns are required.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel.

    Returns:
        pd.DataFrame: Input data with non-null ``seas_11_1na`` values.
    """
    require_columns(data, {'ticker', 'date', 'adjusted_close'})
    result = _prepare_prices(data)
    lagged_prices = [_lagged_prices(result, month) for month in range(1, 13)]
    monthly_returns = [
        lagged_prices[index] / lagged_prices[index + 1] - 1
        for index in range(11)
    ]
    result['seas_11_1na'] = pd.concat(monthly_returns, axis=1).mean(
        axis=1, skipna=False
    )
    return drop_invalid_factor(result, 'seas_11_1na')


def _calculate_momentum(
    data: pd.DataFrame,
    numerator_months: int,
    denominator_months: int,
    factor: str,
) -> pd.DataFrame:
    """Calculate a calendar-month price-momentum signal.

    Args:
        data (pd.DataFrame): Daily price data containing ticker, date, and price.
        numerator_months (int): Calendar months by which to lag the numerator.
        denominator_months (int): Calendar months by which to lag the denominator.
        factor (str): Name of the resulting factor column.

    Returns:
        pd.DataFrame: Data with finite, non-null values for the requested factor.
    """
    if numerator_months < 0 or denominator_months <= numerator_months:
        raise ValueError(
            'denominator_months must be greater than non-negative numerator_months'
        )
    result = _prepare_prices(data)
    numerator = _lagged_prices(result, numerator_months)
    denominator = _lagged_prices(result, denominator_months)
    result[factor] = numerator / denominator - 1
    return drop_invalid_factor(result, factor)


def _prepare_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the price columns used by momentum calculations.

    Args:
        data (pd.DataFrame): Daily data containing ticker, date, and adjusted price.

    Returns:
        pd.DataFrame: Ticker-date-sorted copy with normalized price fields.
    """
    require_columns(data, {'ticker', 'date', 'adjusted_close'})
    result = data.copy()
    result['date'] = pd.to_datetime(result['date'], errors='raise')
    result['adjusted_close'] = pd.to_numeric(
        result['adjusted_close'], errors='coerce'
    )
    return result.sort_values(['ticker', 'date']).reset_index(drop=True)


def _lagged_prices(data: pd.DataFrame, months: int) -> pd.Series:
    """Find prices at calendar-month lags without using future observations.

    Args:
        data (pd.DataFrame): Ticker-sorted data containing dates and adjusted prices.
        months (int): Number of calendar months by which observations are lagged.

    Returns:
        pd.Series: Prices from the latest trading date on or before each lagged date.
    """
    if months < 0:
        raise ValueError('months must be non-negative')
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


__all__ = [
    'eleven_one_seasonality_nonannual',
    'nine_one_momentum',
    'price_to_high_252d',
    'short_term_reversal',
    'six_one_momentum',
    'six_zero_momentum',
    'three_one_momentum',
    'twelve_one_momentum',
]

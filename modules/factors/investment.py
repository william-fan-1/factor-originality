"""Implement investment factors."""

import pandas as pd

from modules.factors.utils import (
    calculate_trailing_sum,
    drop_invalid_factor,
    require_columns,
)


def asset_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical asset growth factor. 

    Calculates the yearly change in total assets divided by total assets 
    from the previous period. The direction is reversed due to low asset 
    growth historically outperforming higher asset growth:

    ``Asset Growth = A*(t) / A*(t - 4) - 1``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, and ``total_assets`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``asst_grwth`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'total_assets'})
    result = data.copy()
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    prior_assets = _lag_quarterly_values(result, assets, quarters=4)
    result['asst_grwth'] = assets / prior_assets - 1
    return drop_invalid_factor(result, 'asst_grwth')


def capex_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical CapEx growth factor. The direction 
    is reversed due to low CapEx growth historically outperforming 
    higher CapEx growth:

    Calculates the yearly change in TTM CapEx divided by TTM CapEx 
    from the previous period:

    ``CapEx Growth = TTM CE*(t) / TTM CE*(t - 4) - 1``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, and ``capital_expenditures`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``capex_gr1`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'capital_expenditures'})
    result = data.copy()
    result['_capex_magnitude'] = pd.to_numeric(
        result['capital_expenditures'], errors='coerce'
    ).abs()
    trailing_capex = calculate_trailing_sum(
        result, '_capex_magnitude', quarters=4
    )
    prior_capex = _lag_quarterly_values(result, trailing_capex, quarters=4)
    result['capex_gr1'] = trailing_capex / prior_capex - 1
    result = result.drop(columns='_capex_magnitude')
    return drop_invalid_factor(result, 'capex_gr1')


def capex_change(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical CapEx change factor. The direction 
    is reversed due to low CapEx change historically outperforming 
    higher CapEx change:

    Calculates the yearly change in TTM CapEx scaled by total assets for the period:

    ``CapEx Change = (TTM CE*(t) - TTM CE*(t - 4)) / A*(t)``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``capital_expenditures``, and ``total_assets`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``capex_gr1_a`` factor column.
    """
    require_columns(
        data,
        {'ticker', 'period_end', 'capital_expenditures', 'total_assets'},
    )
    result = data.copy()
    result['_capex_magnitude'] = pd.to_numeric(
        result['capital_expenditures'], errors='coerce'
    ).abs()
    trailing_capex = calculate_trailing_sum(
        result, '_capex_magnitude', quarters=4
    )
    prior_capex = _lag_quarterly_values(result, trailing_capex, quarters=4)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['capex_gr1_a'] = (trailing_capex - prior_capex) / assets
    result = result.drop(columns='_capex_magnitude')
    return drop_invalid_factor(result, 'capex_gr1_a')


def noa_change(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical net operating asset (NOA) change factor. 
    The direction is reversed due to low NOA change historically outperforming 
    higher NOA change:

    Calculates the yearly change in NOA scaled by total assets for the period:

    ``NOA Change = (NOA*(t) - NOA*(t - 4)) / A*(t)``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``total_assets``, ``total_liabilities``,
            ``cash_and_equivalents``, ``short_term_debt``, and
            ``long_term_debt`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``noa_gr1_a`` factor column.
    """
    require_columns(
        data,
        {
            'ticker',
            'period_end',
            'total_assets',
            'total_liabilities',
            'cash_and_equivalents',
            'short_term_debt',
            'long_term_debt',
        },
    )
    result = data.copy()
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    liabilities = pd.to_numeric(result['total_liabilities'], errors='coerce')
    cash = pd.to_numeric(result['cash_and_equivalents'], errors='coerce')
    short_debt = pd.to_numeric(result['short_term_debt'], errors='coerce')
    long_debt = pd.to_numeric(result['long_term_debt'], errors='coerce')
    net_operating_assets = (
        (assets - cash) - (liabilities - short_debt - long_debt)
    )
    prior_noa = _lag_quarterly_values(
        result, net_operating_assets, quarters=4
    )
    result['noa_gr1_a'] = (net_operating_assets - prior_noa) / assets
    return drop_invalid_factor(result, 'noa_gr1_a')


def _lag_quarterly_values(
    data: pd.DataFrame,
    values: pd.Series,
    quarters: int,
) -> pd.Series:
    """Lag values across unique fiscal-quarter observations for each ticker.

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker`` and ``period_end``.
        values (pd.Series): Values aligned to the rows of ``data``.
        quarters (int): Number of unique fiscal quarters by which to lag values.

    Returns:
        pd.Series: Lagged quarterly values aligned to the original DataFrame index.
    """
    require_columns(data, {'ticker', 'period_end'})
    if quarters < 1:
        raise ValueError('quarters must be at least 1')
    if not values.index.equals(data.index):
        raise ValueError('values must have the same index as data')

    observations = data[['ticker', 'period_end']].copy()
    observations['period_end'] = pd.to_datetime(
        observations['period_end'], errors='coerce'
    )
    observations['_value'] = pd.to_numeric(values, errors='coerce')
    observations = (
        observations
        .dropna(subset=['ticker', 'period_end'])
        .sort_values(['ticker', 'period_end'])
        .drop_duplicates(['ticker', 'period_end'], keep='last')
    )
    observations['_lagged'] = observations.groupby(
        'ticker', sort=False
    )['_value'].shift(quarters)
    lagged = observations.set_index(['ticker', 'period_end'])['_lagged']
    original_period_end = pd.to_datetime(data['period_end'], errors='coerce')
    original_keys = pd.MultiIndex.from_arrays(
        [data['ticker'], original_period_end], names=['ticker', 'period_end']
    )
    return pd.Series(lagged.reindex(original_keys).to_numpy(), index=data.index)


__all__ = ['asset_growth', 'capex_change', 'capex_growth', 'noa_change']

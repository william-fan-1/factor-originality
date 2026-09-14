"""Implement investment factors."""

import pandas as pd

from modules.factors.utils import (
    calculate_trailing_sum,
    drop_invalid_factor,
    lag_quarterly_values,
    require_columns,
)


def asset_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical annual asset-growth factor.

    Construction:
        ``Asset growth = A(t) / A(t-4) - 1``

    Required cached columns:
        ``ticker``, ``period_end``, and ``total_assets``.

    Point-in-time handling:
        Total assets are point-in-time balance-sheet values, and ``t-4`` refers
        to four unique fiscal-quarter observations earlier.

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, and ``total_assets`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``asst_grwth`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'total_assets'})
    result = data.copy()
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    prior_assets = lag_quarterly_values(result, assets, quarters=4)
    result['asst_grwth'] = assets / prior_assets - 1
    return drop_invalid_factor(result, 'asst_grwth')


def capex_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical annual capital-expenditure growth factor.

    Construction:
        ``CapEx growth = TTM CapEx(t) / TTM CapEx(t-4) - 1``

    Required cached columns:
        ``ticker``, ``period_end``, and ``capital_expenditures``.

    Point-in-time handling:
        Quarterly CapEx magnitudes are summed over four unique fiscal quarters,
        and the comparison value is lagged four fiscal quarters.

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
    prior_capex = lag_quarterly_values(result, trailing_capex, quarters=4)
    result['capex_gr1'] = trailing_capex / prior_capex - 1
    result = result.drop(columns='_capex_magnitude')
    return drop_invalid_factor(result, 'capex_gr1')


def capex_change(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical asset-scaled capital-expenditure change factor.

    Construction:
        ``CapEx change = (TTM CapEx(t) - TTM CapEx(t-4)) / A(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``capital_expenditures``, and
        ``total_assets``.

    Point-in-time handling:
        Quarterly CapEx magnitudes are summed over four unique fiscal quarters,
        compared with the TTM value four fiscal quarters earlier, and scaled by
        current point-in-time assets.

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
    prior_capex = lag_quarterly_values(result, trailing_capex, quarters=4)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['capex_gr1_a'] = (trailing_capex - prior_capex) / assets
    result = result.drop(columns='_capex_magnitude')
    return drop_invalid_factor(result, 'capex_gr1_a')


def noa_change(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical asset-scaled net-operating-assets change factor.

    Construction:
        ``NOA change = (NOA(t) - NOA(t-4)) / A(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``total_assets``, ``total_liabilities``,
        ``cash_and_equivalents``, ``short_term_debt``, and ``long_term_debt``.

    Point-in-time handling:
        All inputs are point-in-time balance-sheet values, and the comparison
        value is lagged four unique fiscal-quarter observations.

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
    prior_noa = lag_quarterly_values(
        result, net_operating_assets, quarters=4
    )
    result['noa_gr1_a'] = (net_operating_assets - prior_noa) / assets
    return drop_invalid_factor(result, 'noa_gr1_a')


__all__ = ['asset_growth', 'capex_change', 'capex_growth', 'noa_change']

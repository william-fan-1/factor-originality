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

def common_equity_change(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate change in common equity scaled by assets.

    Construction:
        ``(shareholders_equity_t - shareholders_equity_t-4q) / total_assets_t``

    Required cached columns:
        ticker, period_end, shareholders_equity, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``shareholders_equity``, and ``total_assets``.

    Returns:
        pd.DataFrame: Input data with a non-null ``be_gr1_a`` factor column.
    """
    require_columns(
        data, {'ticker', 'period_end', 'shareholders_equity', 'total_assets'}
    )
    result = data.copy()
    equity = pd.to_numeric(result['shareholders_equity'], errors='coerce')
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    lagged_equity = lag_quarterly_values(result, equity, quarters=4)
    result['be_gr1_a'] = (equity - lagged_equity) / assets
    return drop_invalid_factor(result, 'be_gr1_a')


def coa_change(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate change in current operating assets.

    Construction:
        ``(COA_t - COA_t-4q) / total_assets_t, where COA = current_assets - cash_and_equivalents``

    Required cached columns:
        ticker, period_end, current_assets, cash_and_equivalents, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``current_assets``, ``cash_and_equivalents``, and
            ``total_assets``.

    Returns:
        pd.DataFrame: Input data with a non-null ``coa_gr1_a`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'current_assets',
            'cash_and_equivalents', 'total_assets',
        },
    )
    result = data.copy()
    current_assets = pd.to_numeric(result['current_assets'], errors='coerce')
    cash = pd.to_numeric(result['cash_and_equivalents'], errors='coerce')
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    current_operating_assets = current_assets - cash
    lagged_coa = lag_quarterly_values(
        result, current_operating_assets, quarters=4
    )
    result['coa_gr1_a'] = (current_operating_assets - lagged_coa) / assets
    return drop_invalid_factor(result, 'coa_gr1_a')


def col_change(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate change in current operating liabilities.

    Construction:
        ``(COL_t - COL_t-4q) / total_assets_t, where COL = current_liabilities - short_term_debt``

    Required cached columns:
        ticker, period_end, current_liabilities, short_term_debt, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``current_liabilities``, ``short_term_debt``, and
            ``total_assets``.

    Returns:
        pd.DataFrame: Input data with a non-null ``col_gr1_a`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'current_liabilities',
            'short_term_debt', 'total_assets',
        },
    )
    result = data.copy()
    current_liabilities = pd.to_numeric(
        result['current_liabilities'], errors='coerce'
    )
    short_term_debt = pd.to_numeric(
        result['short_term_debt'], errors='coerce'
    )
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    current_operating_liabilities = current_liabilities - short_term_debt
    lagged_col = lag_quarterly_values(
        result, current_operating_liabilities, quarters=4
    )
    result['col_gr1_a'] = (current_operating_liabilities - lagged_col) / assets
    return drop_invalid_factor(result, 'col_gr1_a')


def sales_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate one-year sales growth.

    Construction:
        ``TTM revenue_t / TTM revenue_t-4q - 1``

    Required cached columns:
        ticker, period_end, revenue

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, and ``revenue``.

    Returns:
        pd.DataFrame: Input data with a non-null ``sale_grwth`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'revenue'})
    result = data.copy()
    revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    lagged_revenue = lag_quarterly_values(result, revenue, quarters=4)
    result['sale_grwth'] = revenue / lagged_revenue - 1
    return drop_invalid_factor(result, 'sale_grwth')


def quarterly_sales_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate one-quarter sales growth.

    Construction:
        ``revenue_q,t / revenue_q,t-1q - 1``

    Required cached columns:
        ticker, period_end, revenue

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Implementation notes:
        Use raw quarterly revenue rather than a TTM sum.

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, and quarterly ``revenue``.

    Returns:
        pd.DataFrame: Input data with a non-null ``sale_grwth_q`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'revenue'})
    result = data.copy()
    revenue = pd.to_numeric(result['revenue'], errors='coerce')
    lagged_revenue = lag_quarterly_values(result, revenue, quarters=1)
    result['sale_grwth_q'] = revenue / lagged_revenue - 1
    return drop_invalid_factor(result, 'sale_grwth_q')

__all__ = [
    'asset_growth',
    'capex_change',
    'capex_growth',
    'coa_change',
    'col_change',
    'common_equity_change',
    'noa_change',
    'quarterly_sales_growth',
    'sales_growth',
]

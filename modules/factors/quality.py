"""Implement quality factors."""

import pandas as pd

from modules.factors.utils import (
    calculate_trailing_sum,
    drop_invalid_factor,
    require_columns,
)

def gross_profit_margin(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical gross profit margin factor.

    Calculates the ratio between TTM gross profit (revenue - cogs)
    and sales:

    ``GP/Sales = (revenue - cogs) / sales``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``revenue``, and ``cost_of_goods_sold`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``gp_sales`` factor column.
    """
    require_columns(
        data,
        {'ticker', 'period_end', 'revenue', 'cost_of_goods_sold'},
    )
    result = data.copy()
    trailing_revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    trailing_cost = calculate_trailing_sum(
        result, 'cost_of_goods_sold', quarters=4
    )
    result['gp_sales'] = (trailing_revenue - trailing_cost) / trailing_revenue
    return drop_invalid_factor(result, 'gp_sales')

def gross_profitability(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical gross profitability factor.

    Calculates the ratio between TTM gross profit (revenue - cogs)
    and assets:

    ``GP/Assets = (revenue - cogs) / total assets``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``revenue``, ``cost_of_goods_sold``, and
            ``total_assets`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``gp_assts`` factor column.
    """
    require_columns(
        data,
        {
            'ticker',
            'period_end',
            'revenue',
            'cost_of_goods_sold',
            'total_assets',
        },
    )
    result = data.copy()
    trailing_revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    trailing_cost = calculate_trailing_sum(
        result, 'cost_of_goods_sold', quarters=4
    )
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['gp_assts'] = (trailing_revenue - trailing_cost) / assets
    return drop_invalid_factor(result, 'gp_assts')

def operating_profitability(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical operating profitability factor.

    Calculates the ratio between TTM operating profit (EBIT)
    and book equity:

    ``OP/B = operating income / book equity``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``operating_income``, ``total_assets``, and
            ``total_liabilities`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``op_be`` factor column.
    """
    require_columns(
        data,
        {
            'ticker',
            'period_end',
            'operating_income',
            'total_assets',
            'total_liabilities',
        },
    )
    result = data.copy()
    trailing_operating_income = calculate_trailing_sum(
        result, 'operating_income', quarters=4
    )
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    liabilities = pd.to_numeric(result['total_liabilities'], errors='coerce')
    result['op_be'] = trailing_operating_income / (assets - liabilities)
    return drop_invalid_factor(result, 'op_be')

def return_on_equity(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical return on equity factor.

    Calculates the ratio between the TTM net income
    and book equity:

    ``NI/B = net income / book equity``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``net_income``, ``total_assets``, and
            ``total_liabilities`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``roe`` factor column.
    """
    require_columns(
        data,
        {
            'ticker',
            'period_end',
            'net_income',
            'total_assets',
            'total_liabilities',
        },
    )
    result = data.copy()
    trailing_net_income = calculate_trailing_sum(result, 'net_income', quarters=4)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    liabilities = pd.to_numeric(result['total_liabilities'], errors='coerce')
    result['roe'] = trailing_net_income / (assets - liabilities)
    return drop_invalid_factor(result, 'roe')


__all__ = [
    'gross_profit_margin',
    'gross_profitability',
    'operating_profitability',
    'return_on_equity',
]

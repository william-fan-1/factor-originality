"""Implement value factors."""

import pandas as pd

from modules.factors.utils import (
    calculate_trailing_sum,
    drop_invalid_factor,
    require_columns,
)


def book_to_market(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical book-equity-to-market-equity factor.

    Calculates the ratio between book equity value (total assets - total liabilities)
    and market equity value (market capitalization):

    ``B/M = (total assets - total liabilities) / market cap``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``total_assets``,
            ``total_liabilities``, and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``be_me`` factor column.
    """
    require_columns(data, {'total_assets', 'total_liabilities', 'market_cap'})
    result = data.copy()
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    liabilities = pd.to_numeric(result['total_liabilities'], errors='coerce')
    market_cap = pd.to_numeric(result['market_cap'], errors='coerce')
    result['be_me'] = (assets - liabilities) / market_cap
    return drop_invalid_factor(result, 'be_me')


def earnings_to_price(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate earnings yield using trailing-four-quarter net income.

    Calculates a company's earnings yield:

    ``E/P = trailing-four-quarter net income / market cap``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``net_income``, and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``e_pe`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'net_income', 'market_cap'})
    result = data.copy()
    trailing_net_income = calculate_trailing_sum(result, 'net_income', quarters=4)
    market_cap = pd.to_numeric(result['market_cap'], errors='coerce')
    result['e_pe'] = trailing_net_income / market_cap
    return drop_invalid_factor(result, 'e_pe')


def fcf_to_price(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical free-cash-flow-to-price factor.

    Calculates the ratio between free cash flow and market capitalization:

    ``FCF/P = (operating cash flow - abs(CapEx)) / market cap``

    Args:
        data (pd.DataFrame): Point-in-time data containing ``operating_cash_flow``,
            ``capital_expenditures``, and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``fcf_me`` factor column.
    """
    require_columns(
        data,
        {'operating_cash_flow', 'capital_expenditures', 'market_cap'},
    )
    result = data.copy()
    operating_cash_flow = pd.to_numeric(
        result['operating_cash_flow'], errors='coerce'
    )
    capital_expenditures = pd.to_numeric(
        result['capital_expenditures'], errors='coerce'
    ).abs()
    market_cap = pd.to_numeric(result['market_cap'], errors='coerce')
    result['fcf_me'] = (
        operating_cash_flow - capital_expenditures
    ) / market_cap
    return drop_invalid_factor(result, 'fcf_me')

__all__ = ['book_to_market', 'earnings_to_price', 'fcf_to_price']

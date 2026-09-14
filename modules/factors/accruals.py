"""Implement accrual factors."""

import pandas as pd

from modules.factors.utils import (
    calculate_trailing_sum,
    drop_invalid_factor,
    lag_quarterly_values,
    require_columns,
)


def cowc_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate change in current operating working capital.

    Construction:
        ``(COWC_t - COWC_t-4q) / total_assets_t, where COWC = (current_assets - cash) - (current_liabilities - short_term_debt)``

    Required cached columns:
        ticker, period_end, current_assets, cash_and_equivalents, current_liabilities, short_term_debt, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``current_assets``, ``cash_and_equivalents``,
            ``current_liabilities``, ``short_term_debt``, and ``total_assets``.

    Returns:
        pd.DataFrame: Input data with a non-null ``cowc_grwth`` factor column.
    """
    required = {
        'ticker',
        'period_end',
        'current_assets',
        'cash_and_equivalents',
        'current_liabilities',
        'short_term_debt',
        'total_assets',
    }
    require_columns(data, required)
    result = data.copy()
    current_assets = pd.to_numeric(result['current_assets'], errors='coerce')
    cash = pd.to_numeric(result['cash_and_equivalents'], errors='coerce')
    current_liabilities = pd.to_numeric(
        result['current_liabilities'], errors='coerce'
    )
    short_term_debt = pd.to_numeric(
        result['short_term_debt'], errors='coerce'
    )
    total_assets = pd.to_numeric(result['total_assets'], errors='coerce')
    cowc = (
        current_assets - cash
    ) - (
        current_liabilities - short_term_debt
    )
    lagged_cowc = lag_quarterly_values(result, cowc, quarters=4)
    result['cowc_grwth'] = (cowc - lagged_cowc) / total_assets
    return drop_invalid_factor(result, 'cowc_grwth')


def operating_accruals_to_assets(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate operating accruals scaled by assets.

    Construction:
        ``(TTM net_income - TTM operating_cash_flow) / total_assets_t``

    Required cached columns:
        ticker, period_end, net_income, operating_cash_flow, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Implementation notes:
        JKP prefers NI - operating cash flow when cash-flow statement data are available, which matches the cached fields.

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``net_income``, ``operating_cash_flow``, and
            ``total_assets``.

    Returns:
        pd.DataFrame: Input data with a non-null ``op_accrl_assts`` factor column.
    """
    require_columns(
        data,
        {
            'ticker',
            'period_end',
            'net_income',
            'operating_cash_flow',
            'total_assets',
        },
    )
    result = data.copy()
    accruals, _ = _trailing_operating_accruals(result)
    total_assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['op_accrl_assts'] = accruals / total_assets
    return drop_invalid_factor(result, 'op_accrl_assts')


def operating_accruals_percent(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate percent operating accruals.

    Construction:
        ``(TTM net_income - TTM operating_cash_flow) / abs(TTM net_income)``

    Required cached columns:
        ticker, period_end, net_income, operating_cash_flow

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``net_income``, and ``operating_cash_flow``.

    Returns:
        pd.DataFrame: Input data with a non-null ``op_accrls_pct`` factor column.
    """
    require_columns(
        data,
        {'ticker', 'period_end', 'net_income', 'operating_cash_flow'},
    )
    result = data.copy()
    accruals, net_income = _trailing_operating_accruals(result)
    result['op_accrls_pct'] = accruals / net_income.abs()
    return drop_invalid_factor(result, 'op_accrls_pct')


def _trailing_operating_accruals(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Calculate TTM operating accruals and their net-income component.

    Args:
        data (pd.DataFrame): Quarterly panel containing net income and operating
            cash flow.

    Returns:
        tuple[pd.Series, pd.Series]: TTM operating accruals and TTM net income,
            both aligned to ``data``.
    """
    net_income = calculate_trailing_sum(data, 'net_income', quarters=4)
    operating_cash_flow = calculate_trailing_sum(
        data, 'operating_cash_flow', quarters=4
    )
    return net_income - operating_cash_flow, net_income


__all__ = [
    'cowc_growth',
    'operating_accruals_percent',
    'operating_accruals_to_assets',
]

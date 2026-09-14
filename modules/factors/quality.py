"""Implement quality factors."""

import numpy as np
import pandas as pd

from modules.factors.utils import (
    calculate_trailing_sum,
    drop_invalid_factor,
    lag_quarterly_values,
    require_columns,
)

def gross_profit_margin(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical gross profit margin factor.

    Construction:
        ``Gross margin = (TTM revenue - TTM COGS) / TTM revenue``

    Required cached columns:
        ``ticker``, ``period_end``, ``revenue``, and ``cost_of_goods_sold``.

    Point-in-time handling:
        Revenue and COGS are summed over the latest four available unique fiscal
        quarters before the ratio is calculated.

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

    Construction:
        ``Gross profitability = (TTM revenue - TTM COGS) / total_assets``

    Required cached columns:
        ``ticker``, ``period_end``, ``revenue``, ``cost_of_goods_sold``, and
        ``total_assets``.

    Point-in-time handling:
        Revenue and COGS are summed over four unique fiscal quarters and divided
        by current point-in-time assets.

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

    Construction:
        ``Operating profitability = TTM operating income / book equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``operating_income``, ``total_assets``, and
        ``total_liabilities``.

    Point-in-time handling:
        Operating income is summed over four unique fiscal quarters and divided
        by current point-in-time book equity.

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

    Construction:
        ``ROE = TTM net income / book equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``net_income``, ``total_assets``, and
        ``total_liabilities``.

    Point-in-time handling:
        Net income is summed over four unique fiscal quarters and divided by
        current point-in-time book equity.

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

def capital_turnover(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate capital turnover.

    Construction:
        ``TTM revenue / total_assets``

    Required cached columns:
        ticker, period_end, revenue, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``cap_to`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'revenue', 'total_assets'})
    result = data.copy()
    revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['cap_to'] = revenue / assets
    return drop_invalid_factor(result, 'cap_to')

def cbop_to_assets(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate cash-based operating profits-to-assets.

    Construction:
        ``(EBITDA - operating_accruals) / total_assets``

    Required cached columns:
        ticker, period_end, operating_income, depreciation_amortization, net_income, operating_cash_flow, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Implementation notes:
        Construct EBITDA as operating_income + depreciation_amortization. JKP adds R&D to operating profit when available; R&D is not in the cache.

    Returns:
        pd.DataFrame: Input data with a non-null ``cbop_assts`` factor column.
    """
    required = {
        'ticker', 'period_end', 'operating_income',
        'depreciation_amortization', 'net_income',
        'operating_cash_flow', 'total_assets',
    }
    require_columns(data, required)
    result = data.copy()
    cbop = _cash_based_operating_profit(result)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['cbop_assts'] = cbop / assets
    return drop_invalid_factor(result, 'cbop_assts')

def cbop_to_lagged_assets(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate cash-based operating profits-to-lagged assets.

    Construction:
        ``(EBITDA - operating_accruals)_t / total_assets_t-4q``

    Required cached columns:
        ticker, period_end, operating_income, depreciation_amortization, net_income, operating_cash_flow, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``cbop_assts_l1`` factor column.
    """
    required = {
        'ticker', 'period_end', 'operating_income',
        'depreciation_amortization', 'net_income',
        'operating_cash_flow', 'total_assets',
    }
    require_columns(data, required)
    result = data.copy()
    cbop = _cash_based_operating_profit(result)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    lagged_assets = lag_quarterly_values(result, assets, quarters=4)
    result['cbop_assts_l1'] = cbop / lagged_assets
    return drop_invalid_factor(result, 'cbop_assts_l1')

def d_gross_margin_minus_d_sales(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate change in gross margin minus change in sales.

    Construction:
        ``year-over-year change in gross_profit margin minus year-over-year change in revenue growth signal``

    Required cached columns:
        ticker, period_end, gross_profit, revenue

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Implementation notes:
        Implement the Abarbanell-Bushee change-to-expectations transform consistently for both gross profit and sales before differencing.

    Returns:
        pd.DataFrame: Input data with a non-null ``dgp_dsale`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'gross_profit', 'revenue'})
    result = data.copy()
    gross_profit = calculate_trailing_sum(result, 'gross_profit', quarters=4)
    revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    lagged_gross_profit = lag_quarterly_values(
        result, gross_profit, quarters=4
    )
    lagged_revenue = lag_quarterly_values(result, revenue, quarters=4)
    gross_profit_change = gross_profit / lagged_gross_profit - 1
    sales_change = revenue / lagged_revenue - 1
    result['dgp_dsale'] = gross_profit_change - sales_change
    return drop_invalid_factor(result, 'dgp_dsale')

def gross_profitability_lagged(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate gross profits-to-lagged assets.

    Construction:
        ``TTM gross_profit_t / total_assets_t-4q``

    Required cached columns:
        ticker, period_end, gross_profit, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``gp_assts_l1`` factor column.
    """
    require_columns(
        data, {'ticker', 'period_end', 'gross_profit', 'total_assets'}
    )
    result = data.copy()
    gross_profit = calculate_trailing_sum(result, 'gross_profit', quarters=4)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    lagged_assets = lag_quarterly_values(result, assets, quarters=4)
    result['gp_assts_l1'] = gross_profit / lagged_assets
    return drop_invalid_factor(result, 'gp_assts_l1')

def consecutive_earnings_increases(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate consecutive quarters with earnings increases.

    Construction:
        ``count consecutive positive quarter-over-quarter net_income changes, capped at 8 quarters``

    Required cached columns:
        ticker, period_end, net_income

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``ni_inc`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'net_income'})
    result = data.copy()
    observations = result[['ticker', 'period_end', 'net_income']].copy()
    observations['period_end'] = pd.to_datetime(
        observations['period_end'], errors='coerce'
    )
    observations['net_income'] = pd.to_numeric(
        observations['net_income'], errors='coerce'
    )
    observations = (
        observations.dropna(subset=['ticker', 'period_end'])
        .sort_values(['ticker', 'period_end'])
        .drop_duplicates(['ticker', 'period_end'], keep='last')
    )

    def count_increases(values: pd.Series) -> pd.Series:
        prior = values.shift(1)
        valid = values.notna() & prior.notna()
        increases = values.gt(prior) & valid
        groups = (~increases).cumsum()
        counts = increases.groupby(groups).cumsum().clip(upper=8).astype(float)
        return counts.where(valid)

    observations['_ni_inc'] = observations.groupby(
        'ticker', group_keys=False, sort=False
    )['net_income'].apply(count_increases)
    signal = observations.set_index(['ticker', 'period_end'])['_ni_inc']
    keys = pd.MultiIndex.from_arrays(
        [result['ticker'], pd.to_datetime(result['period_end'], errors='coerce')],
        names=['ticker', 'period_end'],
    )
    result['ni_inc'] = signal.reindex(keys).to_numpy()
    return drop_invalid_factor(result, 'ni_inc')

def quarterly_return_on_assets(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate quarterly return on assets.

    Construction:
        ``quarterly net_income / lagged total_assets``

    Required cached columns:
        ticker, period_end, net_income, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``ni_assts_q`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'net_income', 'total_assets'})
    result = data.copy()
    net_income = pd.to_numeric(result['net_income'], errors='coerce')
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    lagged_assets = lag_quarterly_values(result, assets, quarters=1)
    result['ni_assts_q'] = net_income / lagged_assets
    return drop_invalid_factor(result, 'ni_assts_q')

def operating_profit_to_assets(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate operating profits-to-assets (JKP ``op_at``).

    Construction:
        ``TTM (operating_income + depreciation_amortization) / total_assets``

    Required cached columns:
        ticker, period_end, operating_income, depreciation_amortization, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Implementation notes:
        JKP operating profit adds R&D to EBITDA when R&D is available; the cache has no R&D, so EBITDA is the available numerator.

    Returns:
        pd.DataFrame: Input data with a non-null ``op_assts`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'operating_income',
            'depreciation_amortization', 'total_assets',
        },
    )
    result = data.copy()
    operating_profit = _operating_profit(result)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['op_assts'] = operating_profit / assets
    return drop_invalid_factor(result, 'op_assts')

def operating_profit_to_lagged_assets(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate operating profits-to-lagged assets.

    Construction:
        ``TTM (operating_income + depreciation_amortization)_t / total_assets_t-4q``

    Required cached columns:
        ticker, period_end, operating_income, depreciation_amortization, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``op_assts_l1`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'operating_income',
            'depreciation_amortization', 'total_assets',
        },
    )
    result = data.copy()
    operating_profit = _operating_profit(result)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    lagged_assets = lag_quarterly_values(result, assets, quarters=4)
    result['op_assts_l1'] = operating_profit / lagged_assets
    return drop_invalid_factor(result, 'op_assts_l1')

def operating_leverage(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate operating leverage.

    Construction:
        ``TTM operating_expenses / total_assets, with operating_expenses = revenue - EBITDA``

    Required cached columns:
        ticker, period_end, revenue, operating_income, depreciation_amortization, total_assets

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``opex_assts`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'revenue', 'operating_income',
            'depreciation_amortization', 'total_assets',
        },
    )
    result = data.copy()
    revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    operating_profit = _operating_profit(result)
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    result['opex_assts'] = (revenue - operating_profit) / assets
    return drop_invalid_factor(result, 'opex_assts')

def qmj_profitability(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Quality-Minus-Junk profitability score.

    Construction:
        ``cross-sectional z-score aggregate of GP/assets, NI/equity, NI/assets, OCF/assets, GP/sales, and negative operating accruals/assets``

    Required cached columns:
        all cached accounting fields needed by the six component characteristics

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Implementation notes:
        At each measurement date, rank/z-score components cross-sectionally across the investable universe, orient accruals so lower is better, average the standardized components, then standardize the aggregate again.

    Returns:
        pd.DataFrame: Input data with a non-null ``qmj_prof`` factor column.
    """
    required = {
        'ticker', 'date', 'period_end', 'gross_profit', 'revenue',
        'net_income', 'operating_cash_flow', 'total_assets',
        'shareholders_equity',
    }
    require_columns(data, required)
    result = data.copy()
    result['date'] = pd.to_datetime(result['date'], errors='raise')
    gross_profit = calculate_trailing_sum(result, 'gross_profit', quarters=4)
    revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    net_income = calculate_trailing_sum(result, 'net_income', quarters=4)
    operating_cash_flow = calculate_trailing_sum(
        result, 'operating_cash_flow', quarters=4
    )
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    equity = pd.to_numeric(result['shareholders_equity'], errors='coerce')
    operating_accruals = net_income - operating_cash_flow
    components = pd.DataFrame(
        {
            'gp_assets': gross_profit / assets,
            'ni_equity': net_income / equity,
            'ni_assets': net_income / assets,
            'ocf_assets': operating_cash_flow / assets,
            'gp_sales': gross_profit / revenue,
            'negative_accruals_assets': -operating_accruals / assets,
        },
        index=result.index,
    ).replace([np.inf, -np.inf], np.nan)
    standardized = components.groupby(result['date']).transform(_zscore)
    aggregate = standardized.mean(axis=1, skipna=False)
    result['qmj_prof'] = aggregate.groupby(result['date']).transform(_zscore)
    return drop_invalid_factor(result, 'qmj_prof')

def assets_turnover(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate assets turnover.

    Construction:
        ``TTM revenue / book_enterprise_value, where BEV = equity + total_debt - cash``

    Required cached columns:
        ticker, period_end, revenue, shareholders_equity, short_term_debt, long_term_debt, cash_and_equivalents

    Point-in-time handling:
        Use only financial statements whose ``filing_timestamp`` is available by the
        measurement date. Quarterly accounting lags refer to unique fiscal quarters;
        flow variables are TTM sums when the JKP characteristic is annual. Market
        equity is ``close * shares_outstanding`` (or the PIT-equivalent market-cap
        field after the price/fundamental merge).

    Returns:
        pd.DataFrame: Input data with a non-null ``assts_to`` factor column.
    """
    required = {
        'ticker', 'period_end', 'revenue', 'shareholders_equity',
        'short_term_debt', 'long_term_debt', 'cash_and_equivalents',
    }
    require_columns(data, required)
    result = data.copy()
    revenue = calculate_trailing_sum(result, 'revenue', quarters=4)
    equity = pd.to_numeric(result['shareholders_equity'], errors='coerce')
    short_debt = pd.to_numeric(result['short_term_debt'], errors='coerce')
    long_debt = pd.to_numeric(result['long_term_debt'], errors='coerce')
    cash = pd.to_numeric(result['cash_and_equivalents'], errors='coerce')
    book_enterprise_value = equity + short_debt + long_debt - cash
    result['assts_to'] = revenue / book_enterprise_value
    return drop_invalid_factor(result, 'assts_to')


def _operating_profit(data: pd.DataFrame) -> pd.Series:
    """Calculate trailing EBITDA from the accounting fields currently available.

    Args:
        data (pd.DataFrame): Quarterly panel containing operating income and D&A.

    Returns:
        pd.Series: Trailing-four-quarter EBITDA aligned to ``data``.
    """
    operating_income = calculate_trailing_sum(
        data, 'operating_income', quarters=4
    )
    depreciation = calculate_trailing_sum(
        data, 'depreciation_amortization', quarters=4
    )
    return operating_income + depreciation


def _cash_based_operating_profit(data: pd.DataFrame) -> pd.Series:
    """Calculate trailing cash-based operating profit.

    Args:
        data (pd.DataFrame): Quarterly panel containing EBITDA and accrual fields.

    Returns:
        pd.Series: Trailing cash-based operating profit aligned to ``data``.
    """
    operating_profit = _operating_profit(data)
    net_income = calculate_trailing_sum(data, 'net_income', quarters=4)
    operating_cash_flow = calculate_trailing_sum(
        data, 'operating_cash_flow', quarters=4
    )
    operating_accruals = net_income - operating_cash_flow
    return operating_profit - operating_accruals


def _zscore(values: pd.Series) -> pd.Series:
    """Standardize a cross section while retaining undefined observations.

    Args:
        values (pd.Series): Cross-sectional characteristic observations.

    Returns:
        pd.Series: Population z-scores, or ``NaN`` when dispersion is unavailable.
    """
    numeric = pd.to_numeric(values, errors='coerce')
    deviation = numeric.std(ddof=0)
    if pd.isna(deviation) or deviation == 0:
        return pd.Series(np.nan, index=values.index, dtype='float64')
    return (numeric - numeric.mean()) / deviation

__all__ = [
    'assets_turnover',
    'capital_turnover',
    'cbop_to_assets',
    'cbop_to_lagged_assets',
    'consecutive_earnings_increases',
    'd_gross_margin_minus_d_sales',
    'gross_profit_margin',
    'gross_profitability',
    'gross_profitability_lagged',
    'operating_leverage',
    'operating_profit_to_assets',
    'operating_profit_to_lagged_assets',
    'operating_profitability',
    'qmj_profitability',
    'quarterly_return_on_assets',
    'return_on_equity',
]

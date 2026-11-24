"""Implement value factors."""

from typing import Literal

import numpy as np
import pandas as pd

from modules.factors.utils import (
    calculate_trailing_sum,
    drop_invalid_factor,
    lag_quarterly_values,
    require_columns,
)

def assets_to_market(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canoncial total-assets-to-market-equity factor - Fama and French (1992).

    Construction:
        ``Assets to market = A(t) / market equity(t)``

    Required cached columns:
        ``total_assets`` and ``market_cap``.

    Point-in-time handling:
        Assets uses the latest available balance sheet, while market equity
        uses the contemporaneous trading-day market capitalization.
    
    Args:
        data (pd.DataFrame): Point-in-time data containing ``total_assets`` 
            and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``asst_me`` factor column.
    """
    require_columns(data, {'total_assets', 'market_cap'})
    result = data.copy()
    assets = pd.to_numeric(result['total_assets'], errors='coerce')
    market_cap = pd.to_numeric(result['market_cap'], errors='coerce')
    result['asst_me'] = assets / market_cap
    return drop_invalid_factor(result, 'asst_me')

def book_to_market(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical book-equity-to-market-equity factor - Rosenberg et al. (1985).

    Construction:
        ``Book to market = (A(t) - L(t)) / market equity(t)``

    Required cached columns:
        ``total_assets``, ``total_liabilities``, and ``market_cap``.

    Point-in-time handling:
        Book equity uses the latest available balance sheet, while market equity
        uses the contemporaneous trading-day market capitalization.

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

def book_to_mev(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical book-equity-to-market-enterprise-value factor - Penman et al. (2007).

    Construction:
        ``Book to market EV = (A(t) - L(t)) / market EV(t), where MEV = market cap + net debt``

    Required cached columns:
        ``total_assets``, ``total_liabilities``, ``long_term_debt``,
        ``short_term_debt``, ``cash_and_equivalents``, ``preferred_stock``,
        ``minority_interest``, and ``market_cap``.

    Point-in-time handling:
        Book equity and the balance-sheet components of MEV (debt, cash,
        preferred stock, minority interest) use the latest available balance
        sheet, while market equity uses the contemporaneous trading-day market
        capitalization. Missing preferred stock or minority interest is
        treated as zero.

    Args:
        data (pd.DataFrame): Point-in-time data containing the columns above.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``be_mev`` factor column.
    """
    require_columns(
        data,
        {
            'total_assets', 'total_liabilities', 'long_term_debt',
            'short_term_debt', 'cash_and_equivalents', 'preferred_stock',
            'minority_interest', 'market_cap',
        },
    )
    result = data.copy()
    book_equity = _numeric(result, 'total_assets') - _numeric(result, 'total_liabilities')
    result['be_mev'] = book_equity / _market_enterprise_value(result)
    return drop_invalid_factor(result, 'be_mev')

def net_stock_issues(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical net-stock-issues factor - Pontiff and Woodgate (2008).

    Construction:
        ``Share Growth = (Shares(t) x Adj(t)) / (Shares(t-12) x Adj(t-12)) - 1``

    Required cached columns:
        ``ticker``, ``date``, ``shares_outstanding``, and
        ``share_adjustment_factor`` (CRSP ``cfacshr``).

    Point-in-time handling:
        Shares are multiplied by CRSP's cumulative share adjustment factor so
        splits and stock dividends do not register as issuance. The comparison
        uses the latest trading date on or before the calendar date 12 months
        earlier.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel containing the
            columns above.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``nsi_12m`` factor column.
    """
    require_columns(
        data,
        {'ticker', 'date', 'shares_outstanding', 'share_adjustment_factor'},
    )
    result = data.copy()
    result['date'] = pd.to_datetime(result['date'], errors='raise')
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)
    adjusted_shares = (
        _numeric(result, 'shares_outstanding')
        * _numeric(result, 'share_adjustment_factor')
    )
    prior_shares = _lag_by_months(result, adjusted_shares, months=12)
    result['nsi_12m'] = adjusted_shares / prior_shares - 1
    return drop_invalid_factor(result, 'nsi_12m')

def debt_to_market(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canoncial total-debt-to-market-equity factor - Bhandari (1988).

    Construction:
        ``Debt to market = Total debt(t) / market equity(t)``

    Required cached columns:
        ``long_term_debt``, ``short_term_debt``, and ``market_cap``.

    Point-in-time handling:
        Total debts uses the latest available balance sheet, while market equity
        uses the contemporaneous trading-day market capitalization.
    
    Args:
        data (pd.DataFrame): Point-in-time data containing ``long_term_debt``, 
            ``short_term_debt``, and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``debt_me`` factor column.
    """
    require_columns(data, {'long_term_debt', 'short_term_debt', 'market_cap'})
    result = data.copy()
    lt_debt = pd.to_numeric(result['long_term_debt'], errors='coerce')
    st_debt = pd.to_numeric(result['short_term_debt'], errors='coerce')
    market_cap = pd.to_numeric(result['market_cap'], errors='coerce')
    result['debt_me'] = (lt_debt + st_debt) / market_cap
    return drop_invalid_factor(result, 'debt_me')

def dividend_yield(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical dividend-yield factor - Litzenberger and Ramaswamy (1979).

    Construction:
        ``D/P = TTM total dividends(t) / market equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``dividends``, and ``market_cap``.

    Point-in-time handling:
        Cash dividends are summed over four unique fiscal quarters and divided
        by contemporaneous trading-day market capitalization. ``dividends`` is
        already in total dollars, so no share count is applied.

    Args:
        data (pd.DataFrame): Point-in-time data containing the columns above.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``div12m_me`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'dividends', 'market_cap'})
    result = data.copy()
    result['_dividends'] = _numeric(result, 'dividends').abs()
    trailing_dividends = calculate_trailing_sum(result, '_dividends', quarters=4)
    result = result.drop(columns='_dividends')
    result['div12m_me'] = trailing_dividends / _numeric(result, 'market_cap')
    return drop_invalid_factor(result, 'div12m_me')

def ebitda_to_mev(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canoncial ebitda-to-market-enterprise-value factor - Loughran and Wellman (2011).

    Construction:
        ``EBTIDA to market EV = EBITDA(t) / market EV(t), where MEV = market cap + net debt``

    Required cached columns:
        ``ticker``, ``period_end``, ``operating_income``,
        ``depreciation_amortization``, ``long_term_debt``, ``short_term_debt``,
        ``cash_and_equivalents``, ``preferred_stock``, ``minority_interest``,
        and ``market_cap``.

    Point-in-time handling:
        EBITDA (``operating_income + depreciation_amortization``) is summed over
        four unique fiscal quarters. The balance-sheet components of MEV use the
        latest balance sheet and market equity uses the contemporaneous
        trading-day market capitalization. Missing preferred stock or minority
        interest is treated as zero.

    Args:
        data (pd.DataFrame): Point-in-time data containing the columns above.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``ebitda_mev`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'operating_income',
            'depreciation_amortization', 'long_term_debt', 'short_term_debt',
            'cash_and_equivalents', 'preferred_stock', 'minority_interest',
            'market_cap',
        },
    )
    result = data.copy()
    result['_ebitda'] = (
        _numeric(result, 'operating_income')
        + _numeric(result, 'depreciation_amortization')
    )
    trailing_ebitda = calculate_trailing_sum(result, '_ebitda', quarters=4)
    result = result.drop(columns='_ebitda')
    result['ebitda_mev'] = trailing_ebitda / _market_enterprise_value(result)
    return drop_invalid_factor(result, 'ebitda_mev')

def net_equity_issuance(
    data: pd.DataFrame,
    method: Literal['balance_sheet', 'cash_flow'] = 'balance_sheet',
) -> pd.DataFrame:
    """Calculate the canonical net equity issuance factor - Bradshaw et al. (2006).

    Construction:
        ``Net Equity Issuance = NEI / total assets(t)``, where NEI is one of:

        - ``'balance_sheet'`` (default): ``ΔBook Equity - ΔRetained Earnings``
          over four fiscal quarters.
        - ``'cash_flow'``: TTM sale of common stock minus TTM purchase of
          common stock (sstk - prstkc).

    Required cached columns:
        ``balance_sheet``: ``ticker``, ``period_end``, ``shareholders_equity``,
        ``retained_earnings``, and ``total_assets``.
        ``cash_flow``: ``ticker``, ``period_end``, ``stock_issuance``,
        ``stock_repurchases``, and ``total_assets``.

    Point-in-time handling:
        Balance-sheet changes compare the latest balance sheet with the one four
        unique fiscal quarters earlier. Cash-flow amounts are summed over four
        unique fiscal quarters (repurchases are cached as positive magnitudes).
        Either way NEI is scaled by the latest available total assets.

    Args:
        data (pd.DataFrame): Point-in-time data containing the columns above.
        method (Literal['balance_sheet', 'cash_flow']): NEI definition to use.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``nei_assts`` factor column.
    """
    if method not in {'balance_sheet', 'cash_flow'}:
        raise ValueError("method must be 'balance_sheet' or 'cash_flow'")
    result = data.copy()
    if method == 'balance_sheet':
        require_columns(
            data,
            {
                'ticker', 'period_end', 'shareholders_equity',
                'retained_earnings', 'total_assets',
            },
        )
        equity = _numeric(result, 'shareholders_equity')
        retained = _numeric(result, 'retained_earnings')
        net_issuance = (
            (equity - lag_quarterly_values(result, equity, quarters=4))
            - (retained - lag_quarterly_values(result, retained, quarters=4))
        )
    else:
        require_columns(
            data,
            {'ticker', 'period_end', 'stock_issuance', 'stock_repurchases', 'total_assets'},
        )
        result['_net_issuance'] = (
            _numeric(result, 'stock_issuance')
            - _numeric(result, 'stock_repurchases').abs()
        )
        net_issuance = calculate_trailing_sum(result, '_net_issuance', quarters=4)
        result = result.drop(columns='_net_issuance')
    result['nei_assts'] = net_issuance / _numeric(result, 'total_assets')
    return drop_invalid_factor(result, 'nei_assts')

def equity_net_payout(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canoncial equity-net-payout factor - Daniel and Titman (2006).

    Measures the net cash flow between a firm and its equity holders by accounting for 
    both share issuance and capital returns. Firms that issue new equity tend to underperform 
    in the future, whereas firms that return capital to shareholders via buybacks and 
    dividends tend to outperform.

    Construction:
        ``Equity Net Payout = log(cumulative return(t)/cumulative return(t-12)) - log(market equity(t)/market equity(t-12))``

    Implementation notes:
        Implemented as a difference rather than a ratio of logs: a ratio blows
        up whenever market equity is flat over the year. The difference is the
        negative of Daniel and Titman's composite issuance.

    Required cached columns:
        ``ticker``, ``date``, ``adjusted_close``, and ``market_cap``.

    Point-in-time handling:
        Cumulative return is the ratio of dividend- and split-adjusted prices.
        Both it and market cap are compared with the latest trading date on or
        before the calendar date 12 months earlier.

    Args:
        data (pd.DataFrame): Daily point-in-time security panel containing the
            columns above.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``ne_payout`` factor column.
    """
    require_columns(data, {'ticker', 'date', 'adjusted_close', 'market_cap'})
    result = data.copy()
    result['date'] = pd.to_datetime(result['date'], errors='raise')
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)
    adjusted_close = _numeric(result, 'adjusted_close')
    market_cap = _numeric(result, 'market_cap')
    prior_close = _lag_by_months(result, adjusted_close, months=12)
    prior_cap = _lag_by_months(result, market_cap, months=12)
    result['ne_payout'] = (
        np.log(adjusted_close / prior_close) - np.log(market_cap / prior_cap)
    )
    return drop_invalid_factor(result, 'ne_payout')

def net_payout_yield(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates the canonical net-payout-yield factor - Boudoukh (2007).

    Construction:
        ``Net Payout Yield = TTM (Dividends + Share Repurchases - Share Issuances) / Market Equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``dividends``, ``stock_repurchases``,
        ``stock_issuance``, and ``market_cap``.

    Point-in-time handling:
        Dividends, repurchases and issuances are summed over four unique fiscal
        quarters (all cached as positive magnitudes) and divided by
        contemporaneous trading-day market capitalization.

    Args:
        data (pd.DataFrame): Point-in-time data containing the columns above.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``npayout_yld`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'dividends', 'stock_repurchases',
            'stock_issuance', 'market_cap',
        },
    )
    result = data.copy()
    result['_net_payout'] = (
        _numeric(result, 'dividends').abs()
        + _numeric(result, 'stock_repurchases').abs()
        - _numeric(result, 'stock_issuance')
    )
    trailing_payout = calculate_trailing_sum(result, '_net_payout', quarters=4)
    result = result.drop(columns='_net_payout')
    result['npayout_yld'] = trailing_payout / _numeric(result, 'market_cap')
    return drop_invalid_factor(result, 'npayout_yld')

def fcf_to_price(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical free-cash-flow-to-price factor.

    Construction:
        ``FCF/P = TTM (operating cash flow - abs(CapEx)) / market equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``operating_cash_flow``,
        ``capital_expenditures``, and ``market_cap``.

    Point-in-time handling:
        Quarterly free cash flow is summed over four unique fiscal quarters and
        divided by contemporaneous trading-day market capitalization.

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``operating_cash_flow``, ``capital_expenditures``,
            and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``fcf_me`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'operating_cash_flow',
            'capital_expenditures', 'market_cap',
        },
    )
    result = data.copy()
    result['_fcf'] = (
        _numeric(result, 'operating_cash_flow')
        - _numeric(result, 'capital_expenditures').abs()
    )
    trailing_fcf = calculate_trailing_sum(result, '_fcf', quarters=4)
    result = result.drop(columns='_fcf')
    result['fcf_me'] = trailing_fcf / _numeric(result, 'market_cap')
    return drop_invalid_factor(result, 'fcf_me')

def net_total_issuance(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical net-total-issuance factor - Bradshaw et al. (1998).

    Construction:
        ``NTI / A = Net Issuance(t) / total assets(t), NTI = Equity Issuance + Debt Issuance``

        Both legs are net: ``(sstk - prstkc) + (dltis - dltr)``.

    Required cached columns:
        ``ticker``, ``period_end``, ``stock_issuance``, ``stock_repurchases``,
        ``debt_issuance``, ``debt_repayment``, and ``total_assets``.

    Point-in-time handling:
        Quarterly net equity and net debt issuance are summed over four unique
        fiscal quarters (all flows cached as positive magnitudes) and scaled by
        the latest available total assets.

    Args:
        data (pd.DataFrame): Point-in-time data containing the columns above.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``nti_assts`` factor column.
    """
    require_columns(
        data,
        {
            'ticker', 'period_end', 'stock_issuance', 'stock_repurchases',
            'debt_issuance', 'debt_repayment', 'total_assets',
        },
    )
    result = data.copy()
    result['_net_issuance'] = (
        _numeric(result, 'stock_issuance')
        - _numeric(result, 'stock_repurchases').abs()
        + _numeric(result, 'debt_issuance')
        - _numeric(result, 'debt_repayment').abs()
    )
    trailing_issuance = calculate_trailing_sum(result, '_net_issuance', quarters=4)
    result = result.drop(columns='_net_issuance')
    result['nti_assts'] = trailing_issuance / _numeric(result, 'total_assets')
    return drop_invalid_factor(result, 'nti_assts')

def earnings_to_price(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate earnings yield using trailing-four-quarter net income.

    Construction:
        ``Earnings yield = TTM net income / market equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``net_income``, and ``market_cap``.

    Point-in-time handling:
        Net income is summed over four unique fiscal quarters and divided by
        contemporaneous trading-day market capitalization.

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

def op_cash_flow_to_market(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical operating-cash-flow-to-market-equity factor - Desai et al. (2004).

    Construction:
        ``TTM Operating Cash Flow(t) / Market Equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``operating_cash_flow``, and ``market_cap``.

    Point-in-time handling:
        Operating cash flow is summed over four unique fiscal quarters and
        divided by contemporaneous trading-day market capitalization.

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``operating_cash_flow``, and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``ocf_me`` factor column.
    """
    require_columns(
        data, {'ticker', 'period_end', 'operating_cash_flow', 'market_cap'}
    )
    result = data.copy()
    trailing_ocf = calculate_trailing_sum(result, 'operating_cash_flow', quarters=4)
    result['ocf_me'] = trailing_ocf / _numeric(result, 'market_cap')
    return drop_invalid_factor(result, 'ocf_me')

def sales_to_market(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical sales-to-market factor - Barbee Jr et al. (1996).

    Construction:
        ``TTM Sales(t) / market equity(t)``

    Required cached columns:
        ``ticker``, ``period_end``, ``revenue``, and ``market_cap``.

    Point-in-time handling:
        Revenue is summed over four unique fiscal quarters and divided by
        contemporaneous trading-day market capitalization.

    Args:
        data (pd.DataFrame): Point-in-time data containing ``ticker``,
            ``period_end``, ``revenue``, and ``market_cap`` columns.

    Returns:
        pd.DataFrame: Copy of the data with a non-null ``sale_me`` factor column.
    """
    require_columns(data, {'ticker', 'period_end', 'revenue', 'market_cap'})
    result = data.copy()
    trailing_sales = calculate_trailing_sum(result, 'revenue', quarters=4)
    result['sale_me'] = trailing_sales / _numeric(result, 'market_cap')
    return drop_invalid_factor(result, 'sale_me')

def _numeric(data: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to numeric values.

    Args:
        data (pd.DataFrame): Data containing ``column``.
        column (str): Column to convert.

    Returns:
        pd.Series: Numeric column with unparseable entries set to ``NaN``.
    """
    return pd.to_numeric(data[column], errors='coerce')

def _market_enterprise_value(data: pd.DataFrame) -> pd.Series:
    """Calculate market enterprise value.

    ``MEV = market cap + debt + preferred stock + minority interest - cash``.

    Args:
        data (pd.DataFrame): Data containing ``market_cap``, ``long_term_debt``,
            ``short_term_debt``, ``cash_and_equivalents``, ``preferred_stock``,
            and ``minority_interest``.

    Returns:
        pd.Series: Market enterprise value, ``NaN`` where market cap, debt or
            cash is missing. Missing preferred stock and minority interest
            count as zero.
    """
    claims = (
        _numeric(data, 'long_term_debt')
        + _numeric(data, 'short_term_debt')
        + _numeric(data, 'preferred_stock').fillna(0.0)
        + _numeric(data, 'minority_interest').fillna(0.0)
        - _numeric(data, 'cash_and_equivalents')
    )
    return _numeric(data, 'market_cap') + claims

def _lag_by_months(
    data: pd.DataFrame,
    values: pd.Series,
    months: int,
) -> pd.Series:
    """Look up values at a calendar-month lag without using future observations.

    Args:
        data (pd.DataFrame): Ticker-date-sorted data with a fresh index.
        values (pd.Series): Values aligned to the rows of ``data``.
        months (int): Calendar months by which observations are lagged.

    Returns:
        pd.Series: Values from the latest trading date on or before each lagged
            date, ``NaN`` where history does not reach back far enough.
    """
    lagged = pd.Series(np.nan, index=data.index, dtype='float64')
    offset = pd.DateOffset(months=months)
    for _, group in data.groupby('ticker', sort=False):
        dates = pd.DatetimeIndex(group['date'])
        positions = dates.searchsorted(dates - offset, side='right') - 1
        valid = positions >= 0
        group_values = values.loc[group.index].to_numpy(dtype='float64')
        out = np.full(len(group), np.nan, dtype='float64')
        out[valid] = group_values[positions[valid]]
        lagged.loc[group.index] = out
    return lagged

__all__ = [
    'assets_to_market',
    'book_to_market',
    'book_to_mev',
    'debt_to_market',
    'dividend_yield',
    'earnings_to_price',
    'ebitda_to_mev',
    'equity_net_payout',
    'fcf_to_price',
    'net_equity_issuance',
    'net_payout_yield',
    'net_stock_issues',
    'net_total_issuance',
    'op_cash_flow_to_market',
    'sales_to_market',
]

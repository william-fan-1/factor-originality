"""Map factor names to their signal functions and portfolio metadata."""

from collections.abc import Callable
from typing import Literal, TypeAlias, TypedDict

import pandas as pd

from modules.factors.accruals import *
from modules.factors.investment import *
from modules.factors.momentum import *
from modules.factors.quality import *
from modules.factors.value import *

FactorFunction = Callable[[pd.DataFrame], pd.DataFrame]
FactorDirection: TypeAlias = Literal['high', 'low']


class FactorDefinition(TypedDict):
    """Describe the callable, signal column, and portfolio direction of a factor."""

    column: str
    func: FactorFunction
    direction: FactorDirection


FACTOR_MAPPINGS: dict[str, FactorDefinition] = {
    'cowc_growth': {
        'column': 'cowc_grwth',
        'func': cowc_growth,
        'direction': 'low',
    },
    'operating_accruals_to_assets': {
        'column': 'op_accrl_assts',
        'func': operating_accruals_to_assets,
        'direction': 'low',
    },
    'operating_accruals_percent': {
        'column': 'op_accrls_pct',
        'func': operating_accruals_percent,
        'direction': 'low',
    },
    'asset_growth': {
        'column': 'asst_grwth',
        'func': asset_growth,
        'direction': 'low',
    },
    'capex_growth': {
        'column': 'capex_gr1',
        'func': capex_growth,
        'direction': 'low',
    },
    'capex_change': {
        'column': 'capex_gr1_a',
        'func': capex_change,
        'direction': 'low',
    },
    'noa_change': {
        'column': 'noa_gr1_a',
        'func': noa_change,
        'direction': 'low',
    },
    'common_equity_change': {
        'column': 'be_gr1_a',
        'func': common_equity_change,
        'direction': 'low',
    },
    'coa_change': {
        'column': 'coa_gr1_a',
        'func': coa_change,
        'direction': 'low',
    },
    'col_change': {
        'column': 'col_gr1_a',
        'func': col_change,
        'direction': 'high',
    },
    'sales_growth': {
        'column': 'sale_grwth',
        'func': sales_growth,
        'direction': 'low',
    },
    'quarterly_sales_growth': {
        'column': 'sale_grwth_q',
        'func': quarterly_sales_growth,
        'direction': 'high',
    },
    'book_to_market': {
        'column': 'be_me',
        'func': book_to_market,
        'direction': 'high',
    },
    'earnings_to_price': {
        'column': 'e_pe',
        'func': earnings_to_price,
        'direction': 'high',
    },
    'fcf_to_price': {
        'column': 'fcf_me',
        'func': fcf_to_price,
        'direction': 'high',
    },
    'gross_profit_margin': {
        'column': 'gp_sales',
        'func': gross_profit_margin,
        'direction': 'high',
    },
    'gross_profitability': {
        'column': 'gp_assts',
        'func': gross_profitability,
        'direction': 'high',
    },
    'operating_profitability': {
        'column': 'op_be',
        'func': operating_profitability,
        'direction': 'high',
    },
    'return_on_equity': {
        'column': 'roe',
        'func': return_on_equity,
        'direction': 'high',
    },
    'capital_turnover': {
        'column': 'cap_to',
        'func': capital_turnover,
        'direction': 'high',
    },
    'cbop_to_assets': {
        'column': 'cbop_assts',
        'func': cbop_to_assets,
        'direction': 'high',
    },
    'cbop_to_lagged_assets': {
        'column': 'cbop_assts_l1',
        'func': cbop_to_lagged_assets,
        'direction': 'high',
    },
    'd_gross_margin_minus_d_sales': {
        'column': 'dgp_dsale',
        'func': d_gross_margin_minus_d_sales,
        'direction': 'high',
    },
    'gross_profitability_lagged': {
        'column': 'gp_assts_l1',
        'func': gross_profitability_lagged,
        'direction': 'high',
    },
    'consecutive_earnings_increases': {
        'column': 'ni_inc',
        'func': consecutive_earnings_increases,
        'direction': 'high',
    },
    'quarterly_return_on_assets': {
        'column': 'ni_assts_q',
        'func': quarterly_return_on_assets,
        'direction': 'high',
    },
    'operating_profit_to_assets': {
        'column': 'op_assts',
        'func': operating_profit_to_assets,
        'direction': 'high',
    },
    'operating_profit_to_lagged_assets': {
        'column': 'op_assts_l1',
        'func': operating_profit_to_lagged_assets,
        'direction': 'high',
    },
    'operating_leverage': {
        'column': 'opex_assts',
        'func': operating_leverage,
        'direction': 'high',
    },
    'qmj_profitability': {
        'column': 'qmj_prof',
        'func': qmj_profitability,
        'direction': 'high',
    },
    'assets_turnover': {
        'column': 'assts_to',
        'func': assets_turnover,
        'direction': 'high',
    },
    'twelve_one_momentum': {
        'column': 'mom_12_1',
        'func': twelve_one_momentum,
        'direction': 'high',
    },
    'six_zero_momentum': {
        'column': 'mom_6_0',
        'func': six_zero_momentum,
        'direction': 'high',
    },
    'short_term_reversal': {
        'column': 'st_rev',
        'func': short_term_reversal,
        'direction': 'low',
    },
    'price_to_high_252d': {
        'column': 'prc_high_252',
        'func': price_to_high_252d,
        'direction': 'high',
    },
    'three_one_momentum': {
        'column': 'mom_3_1',
        'func': three_one_momentum,
        'direction': 'high',
    },
    'six_one_momentum': {
        'column': 'mom_6_1',
        'func': six_one_momentum,
        'direction': 'high',
    },
    'nine_one_momentum': {
        'column': 'mom_9_1',
        'func': nine_one_momentum,
        'direction': 'high',
    },
    'eleven_one_seasonality_nonannual': {
        'column': 'seas_11_1na',
        'func': eleven_one_seasonality_nonannual,
        'direction': 'high',
    },
}


def get_factor_mapping(name: str) -> FactorDefinition:
    """Retrieve factor metadata.

    Args:
        name (str): Registered factor name, such as ``'book_to_market'``.

    Returns:
        FactorDefinition: Copy of the factor's callable, column, and direction.
    """
    try:
        return FACTOR_MAPPINGS[name].copy()
    except KeyError as exc:
        available = ', '.join(sorted(FACTOR_MAPPINGS))
        raise KeyError(
            f"Unknown factor '{name}'; available factors: {available}"
        ) from exc


__all__ = [
    'FACTOR_MAPPINGS',
    'FactorDefinition',
    'FactorDirection',
    'FactorFunction',
    'get_factor_mapping',
]

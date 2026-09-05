"""Map factor names to their signal functions and portfolio metadata."""

from collections.abc import Callable
from typing import Literal, TypeAlias, TypedDict

import pandas as pd

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

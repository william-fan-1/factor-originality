"""Provide a common interface for constructing long-short factor portfolios."""

from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from modules.factors.utils import require_columns

FactorDirection: TypeAlias = Literal['high', 'low']


class Factor:
    """Construct daily value-weighted tercile portfolios for one factor signal."""

    def __init__(
        self,
        factor: str,
        year: int,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        data: pd.DataFrame,
        direction: FactorDirection = 'high',
    ) -> None:
        """Initialize a factor and calculate its portfolio returns.

        Args:
            factor (str): Signal column produced by ``func``.
            year (int): First year of the factor's source-data period.
            func (Callable[[pd.DataFrame], pd.DataFrame]): Function that adds the
                named factor signal to the supplied data.
            data (pd.DataFrame): Daily security panel required by the factor function.
            direction (FactorDirection): ``'high'`` when the highest signal tercile
                is long, or ``'low'`` when the lowest signal tercile is long.

        Returns:
            None.
        """
        if not callable(func):
            raise TypeError('func must be callable')
        if direction not in {'high', 'low'}:
            raise ValueError("direction must be either 'high' or 'low'")

        self.factor = factor
        self.year = year
        self.func = func
        self.direction = direction
        self.signals = pd.DataFrame()
        self.portfolios = pd.DataFrame()
        self.portfolio_returns = pd.DataFrame()
        self.data = self.get_factor_data(data=data)

    def get_factor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Construct factor returns.

        Args:
            data (pd.DataFrame): Daily security panel used to calculate signals,
                portfolios, weights, and subsequent returns.

        Returns:
            pd.DataFrame: Daily long, short, and long-short factor returns.
        """
        self.signals = self.calculate_factor(data=data)
        self.portfolios = self.create_portfolios(data=self.signals)
        self.portfolio_returns = self._calculate_portfolio_returns(
            portfolios=self.portfolios
        )
        return self.portfolio_returns.copy()

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate signals.

        Args:
            data (pd.DataFrame): Security panel accepted by the factor function.

        Returns:
            pd.DataFrame: Factor-function output containing the configured signal.
        """
        result = self.func(data.copy())
        if not isinstance(result, pd.DataFrame):
            raise TypeError('factor function must return a pandas DataFrame')
        require_columns(result, {'ticker', 'date', self.factor})
        result = result.copy()
        result['date'] = pd.to_datetime(result['date'], errors='raise')
        result[self.factor] = pd.to_numeric(result[self.factor], errors='coerce')
        return (
            result
            .dropna(subset=[self.factor])
            .sort_values(['ticker', 'date'])
            .reset_index(drop=True)
        )

    def create_portfolios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create portfolios.

        Args:
            data (pd.DataFrame): Daily factor observations containing ticker, date,
                signal, and market-capitalization columns.

        Returns:
            pd.DataFrame: Factor observations with tercile assignments and weights.
        """
        weight_column = self._weight_column(data)
        require_columns(data, {'ticker', 'date', self.factor, weight_column})

        result = data.copy()
        result['date'] = pd.to_datetime(result['date'], errors='raise')
        signal = pd.to_numeric(result[self.factor], errors='coerce')
        group = signal.groupby(result['date'])
        percentile_rank = group.rank(method='average', pct=True)
        observation_count = group.transform('count')

        result['portfolio'] = 'no portfolio'
        eligible = observation_count.ge(3) & signal.notna()
        low_tercile = eligible & percentile_rank.le(1 / 3)
        high_tercile = eligible & percentile_rank.gt(2 / 3)
        low_portfolio = 'long' if self.direction == 'low' else 'short'
        high_portfolio = 'short' if self.direction == 'low' else 'long'
        result.loc[low_tercile, 'portfolio'] = low_portfolio
        result.loc[high_tercile, 'portfolio'] = high_portfolio
        result['weight'] = np.nan

        selected = result['portfolio'].isin({'long', 'short'})
        if selected.any():
            result.loc[selected] = self._calculate_weights(
                result.loc[selected].copy(),
                weight_column=weight_column,
            )
        return result.sort_values(['ticker', 'date']).reset_index(drop=True)

    def _calculate_weights(
        self,
        subset: pd.DataFrame,
        weight_column: str | None = None,
    ) -> pd.DataFrame:
        """Calculate weights.

        Args:
            subset (pd.DataFrame): Long and short portfolio observations containing
                date, portfolio, and market capitalization.
            weight_column (str | None): Capitalization column to use, or ``None`` to
                select winsorized capitalization when available.

        Returns:
            pd.DataFrame: Portfolio observations with within-portfolio weights.
        """
        column = weight_column or self._weight_column(subset)
        require_columns(subset, {'date', 'portfolio', column})
        result = subset.copy()
        capitalization = pd.to_numeric(result[column], errors='coerce')
        capitalization = capitalization.where(capitalization.gt(0))
        totals = capitalization.groupby(
            [result['date'], result['portfolio']]
        ).transform('sum')
        result['weight'] = capitalization / totals.where(totals.gt(0))
        return result

    def calculate_portfolio_return(self, portfolio: pd.DataFrame) -> float:
        """Calculate return.

        Args:
            portfolio (pd.DataFrame): One date-portfolio subset containing security
                ``return`` and lagged ``weight`` columns.

        Returns:
            float: Renormalized weighted portfolio return, or ``NaN`` if unavailable.
        """
        require_columns(portfolio, {'return', 'weight'})
        returns = pd.to_numeric(portfolio['return'], errors='coerce')
        weights = pd.to_numeric(portfolio['weight'], errors='coerce')
        valid = returns.notna() & weights.notna() & weights.gt(0)
        if not valid.any():
            return float('nan')
        return float((returns[valid] * weights[valid]).sum() / weights[valid].sum())

    def _calculate_portfolio_returns(
        self,
        portfolios: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate return series.

        Args:
            portfolios (pd.DataFrame): Daily portfolio assignments and weights.

        Returns:
            pd.DataFrame: Daily long, short, and long-short factor returns.
        """
        require_columns(portfolios, {'ticker', 'date', 'portfolio', 'weight', 'return'})
        holdings = portfolios.sort_values(['ticker', 'date']).copy()
        holdings[['portfolio', 'weight']] = (
            holdings.groupby('ticker', sort=False)[['portfolio', 'weight']].shift(1)
        )
        holdings = holdings.loc[
            holdings['portfolio'].isin({'long', 'short'})
        ].copy()
        if holdings.empty:
            return pd.DataFrame(
                columns=['date', 'long_return', 'short_return', 'factor_return']
            )

        returns = pd.to_numeric(holdings['return'], errors='coerce')
        weights = pd.to_numeric(holdings['weight'], errors='coerce')
        valid = returns.notna() & weights.notna() & weights.gt(0)
        holdings = holdings.loc[valid].copy()
        holdings['_weighted_return'] = returns[valid] * weights[valid]
        grouped = (
            holdings
            .groupby(['date', 'portfolio'], as_index=False)
            .agg(
                weighted_return=('_weighted_return', 'sum'),
                available_weight=('weight', 'sum'),
            )
        )
        grouped['portfolio_return'] = (
            grouped['weighted_return'] / grouped['available_weight']
        )
        result = grouped.pivot(
            index='date', columns='portfolio', values='portfolio_return'
        )
        for portfolio in ('long', 'short'):
            if portfolio not in result:
                result[portfolio] = np.nan
        result = result.rename(
            columns={'long': 'long_return', 'short': 'short_return'}
        )
        result['factor_return'] = result['long_return'] - result['short_return']
        return (
            result[['long_return', 'short_return', 'factor_return']]
            .reset_index()
            .sort_values('date')
            .reset_index(drop=True)
        )

    @staticmethod
    def _weight_column(data: pd.DataFrame) -> str:
        """Select capitalization.

        Args:
            data (pd.DataFrame): Data containing possible weighting columns.

        Returns:
            str: Preferred market-capitalization column name.
        """
        if 'winsorized_market_cap' in data:
            return 'winsorized_market_cap'
        if 'market_cap' in data:
            return 'market_cap'
        raise ValueError(
            'Missing required weighting column: winsorized_market_cap or market_cap'
        )


__all__ = ['Factor', 'FactorDirection']

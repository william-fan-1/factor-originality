"""Provide a common interface for constructing long-short factor portfolios."""

import hashlib
import inspect
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from modules.factors.mappings import (
    FactorDirection,
    get_factor_mapping,
)
from modules.factors.utils import require_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTOR_CACHE_DIR = PROJECT_ROOT / 'data' / 'factors'
_CACHED_FRAMES = ('signals', 'portfolios', 'portfolio_returns')


def hash_data(data: pd.DataFrame) -> str:
    """Fingerprint a security panel by its contents.

    Args:
        data (pd.DataFrame): Daily security panel.

    Returns:
        str: Hex digest that changes whenever values, columns, or dtypes change.
    """
    digest = hashlib.sha256()
    digest.update(repr((data.shape, list(data.columns), list(data.dtypes))).encode())
    digest.update(pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes())
    return digest.hexdigest()


class Factor:
    """Construct daily value-weighted tercile portfolios for one factor signal."""

    def __init__(
        self,
        factor: str,
        data: pd.DataFrame,
        use_cache: bool = True,
        refresh: bool = False,
        cache_dir: Path | str = FACTOR_CACHE_DIR,
        data_key: str | None = None,
    ) -> None:
        """Initialize a factor and calculate its portfolio returns.

        Results are cached on disk under a key built from the factor definition
        (name, direction, and source of the factor function) and a fingerprint of
        ``data``. Changing either one produces a new cache entry. Edits to helper
        functions called by the factor function are not detected; use ``refresh``.

        Args:
            factor (str): Registered factor name from ``FACTOR_MAPPINGS``.
            data (pd.DataFrame): Daily security panel required by the factor function.
                If it has a boolean ``in_index`` column, only rows where it is true
                are ranked into portfolios; other rows still supply signal history.
            use_cache (bool): Whether to read from and write to the on-disk cache.
            refresh (bool): Recalculate and overwrite any existing cache entry.
            cache_dir (Path | str): Root directory of the factor cache.
            data_key (str | None): Precomputed ``hash_data(data)``, to avoid
                re-hashing a large panel for every factor.

        Returns:
            None.
        """
        definition = get_factor_mapping(factor)
        self.name = factor
        self.factor = definition['column']
        self.func = definition['func']
        self.direction: FactorDirection = definition['direction']
        self.signals = pd.DataFrame()
        self.portfolios = pd.DataFrame()
        self.portfolio_returns = pd.DataFrame()
        self.cache_path: Path | None = None
        if use_cache:
            key = self._cache_key(data_key or hash_data(data))
            self.cache_path = Path(cache_dir) / self.name / key
            if not refresh and self._load_cache():
                self.data = self.portfolio_returns.copy()
                return
        self.data = self.get_factor_data(data=data)
        if self.cache_path is not None:
            self._save_cache()

    def _cache_key(self, data_key: str) -> str:
        """Build the cache key.

        Args:
            data_key (str): Fingerprint of the input panel.

        Returns:
            str: Short hex digest identifying this factor definition and input.
        """
        try:
            source = inspect.getsource(self.func)
        except (OSError, TypeError):
            source = getattr(self.func, '__qualname__', repr(self.func))
        payload = '|'.join([self.name, self.factor, self.direction, source, data_key])
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def _load_cache(self) -> bool:
        """Load cached frames if a complete entry exists.

        Returns:
            bool: ``True`` if the cache was loaded, ``False`` on a miss or bad entry.
        """
        assert self.cache_path is not None
        paths = {n: self.cache_path / f'{n}.parquet' for n in _CACHED_FRAMES}
        if not all(path.is_file() for path in paths.values()):
            return False
        try:
            frames = {n: pd.read_parquet(path) for n, path in paths.items()}
        except Exception:
            return False
        self.signals = frames['signals']
        self.portfolios = frames['portfolios']
        self.portfolio_returns = frames['portfolio_returns']
        return True

    def _save_cache(self) -> None:
        """Write frames to a temporary directory, then move it into place."""
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(f'{self.cache_path.name}.{uuid.uuid4().hex}.tmp')
        tmp.mkdir()
        try:
            for name in _CACHED_FRAMES:
                getattr(self, name).to_parquet(tmp / f'{name}.parquet', index=False)
            shutil.rmtree(self.cache_path, ignore_errors=True)
            tmp.rename(self.cache_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
        if 'in_index' in result:
            signal = signal.where(result['in_index'].fillna(False).astype(bool))
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
            .rename_axis(columns=None)
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


__all__ = ['Factor', 'FactorDirection', 'hash_data']

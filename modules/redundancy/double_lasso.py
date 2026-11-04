"""Module for running double LASSO regressions.

Implements the double-selection test of Feng, Giglio and Xiu (2020), "Taming the
Factor Zoo". A factor's marginal contribution is its SDF loading, estimated from
cross-sectional regressions of test-asset average returns on covariances with
factors, with LASSO selecting which control factors to include.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import numpy as np
from celer import Lasso, celer_path
from pathlib import Path
from modules.data_retrieval.utils import connection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ASSETS_PATH = PROJECT_ROOT / 'data' / 'test_assets' / 'ff25.csv'
START_DATE = '2004-01-01'
END_DATE = '2024-12-31'
SEED = 42
TOLERANCE = 1e-6


@dataclass
class DoubleSelectionResult:
    """Outcome of the double-selection test of one factor.

    Attributes:
        loading (float): Estimated SDF loading of the test factor.
        scaled_loading (float): Loading times the factor's variance: the average
            monthly excess return of a portfolio with unit univariate beta.
        std_error (float): Newey-West standard error of ``loading``.
        t_stat (float): ``loading / std_error``.
        first_selection (set[str]): Controls kept by the first LASSO.
        second_selection (set[str]): Controls kept by the second LASSO.
    """

    loading: float
    scaled_loading: float
    std_error: float
    t_stat: float
    first_selection: set[str]
    second_selection: set[str]


def _load_test_assets() -> pd.DataFrame:
    """Load the Fama-French 25 size-book-to-market portfolios.

    Reads ``data/test_assets/ff25.csv`` if it exists. Otherwise downloads the
    CRSP-based value-weighted 5x5 portfolios from WRDS (``ff.portfolios25``) for
    ``START_DATE`` through ``END_DATE``, subtracts the one-month T-bill rate, and
    saves the result to that path.

    Returns:
        pd.DataFrame: Monthly excess returns as decimals, indexed by month-end date,
            with one column per portfolio named ``s{size}b{book_to_market}``.
    """
    if TEST_ASSETS_PATH.is_file():
        return pd.read_csv(TEST_ASSETS_PATH, index_col='date', parse_dates=['date'])

    columns = ', '.join(
        f'p.s{size}b{value}_vwret AS s{size}b{value}'
        for size in range(1, 6)
        for value in range(1, 6)
    )
    raw = connection().raw_sql(
        f"""
        SELECT p.date, {columns}, f.rf
        FROM ff.portfolios25 AS p
        LEFT JOIN ff.factors_monthly AS f
          ON date_trunc('month', f.date) = date_trunc('month', p.date)
        WHERE p.date BETWEEN %(start)s AND %(end)s
        ORDER BY p.date
        """,
        params={'start': START_DATE, 'end': END_DATE},
        date_cols=['date'],
    ).set_index('date')
    if raw['rf'].isna().any():
        raise ValueError('Missing risk-free rate for some test-asset months')

    portfolios = raw.drop(columns='rf').sub(raw['rf'], axis=0)
    TEST_ASSETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    portfolios.to_csv(TEST_ASSETS_PATH)
    return portfolios

def _align_monthly(*frames: pd.DataFrame | pd.Series) -> list:
    """Restrict monthly data to the months they all share.

    Test assets are dated on a month's last trading day while factor returns are
    dated on the calendar month-end, so rows are matched on year and month.

    Args:
        *frames (pd.DataFrame | pd.Series): Monthly data with a datetime index.

    Returns:
        list: The inputs, indexed by month period and limited to the shared months.
    """
    shifted = [
        frame if isinstance(frame.index, pd.PeriodIndex)
        else frame.set_axis(frame.index.to_period('M'))
        for frame in frames
    ]
    common = shifted[0].index
    for frame in shifted[1:]:
        common = common.intersection(frame.index)
    if common.empty:
        raise ValueError('inputs share no months')
    return [frame.loc[common] for frame in shifted]

def _covariances(returns: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Sample covariances between each asset and each factor.

    Args:
        returns (np.ndarray): Asset returns, ``T x n``.
        factors (np.ndarray): Factor returns, ``T x p``.

    Returns:
        np.ndarray: ``n x p`` matrix ``T^-1 (R - mean)' (F - mean)``.
    """
    returns = returns - returns.mean(axis=0)
    factors = factors - factors.mean(axis=0)
    return returns.T @ factors / len(returns)

def _penalty_scale(returns: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Column scales that turn the L1 penalty into the paper's weighted penalty.

    Each factor's penalty weight is proportional to the inverse of the norm of its
    vector of univariate betas, so factors with small exposures are penalized more.
    A weighted LASSO is the same as an unweighted LASSO on columns multiplied by
    the inverse weights, that is, by the beta norms.

    Args:
        returns (np.ndarray): Asset returns, ``T x n``.
        factors (np.ndarray): Factor returns, ``T x p``.

    Returns:
        np.ndarray: Length-``p`` multipliers for the columns of the covariance matrix.
    """
    variances = factors.var(axis=0)
    betas = _covariances(returns, factors) / variances
    return np.linalg.norm(betas, axis=0)

def _rescale(
    statistics: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    n_obs: int,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Divide regressors and response by one full-sample scalar each.

    Covariances and returns are tiny numbers, which makes the solver's relative
    tolerance unreachable. A single scalar per side leaves the selected set unchanged
    because the relative weights of the columns are untouched. Penalties are then
    on this standardized scale.

    Args:
        statistics (Callable): Maps month positions to ``(X, y)``.
        n_obs (int): Number of months.

    Returns:
        Callable: ``statistics`` with its outputs rescaled.
    """
    X, y = statistics(np.arange(n_obs))
    x_scale, y_scale = X.std(), y.std()

    def rescaled(months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_months, y_months = statistics(months)
        return X_months / x_scale, y_months / y_scale

    return rescaled

def _alpha_grid(X: np.ndarray, y: np.ndarray, size: int = 60) -> np.ndarray:
    """Descending penalty grid from the smallest penalty that zeroes every
    coefficient down to a hundredth of it. Smaller penalties fit the handful of
    test assets almost exactly and are slow to solve without being selected.

    Args:
        X (np.ndarray): Regressors.
        y (np.ndarray): Response.
        size (int): Number of penalty values.

    Returns:
        np.ndarray: Penalties in celer's parameterization.
    """
    centered_X = X - X.mean(axis=0)
    centered_y = y - y.mean()
    largest = np.abs(centered_X.T @ centered_y).max() / len(y)
    return np.geomspace(largest, largest * 1e-2, size)

def _lasso_path(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Solve the LASSO along a penalty grid, leaving the intercept unpenalized.

    Args:
        X (np.ndarray): Regressors.
        y (np.ndarray): Response.
        alphas (np.ndarray): Descending penalties.

    Returns:
        np.ndarray: ``p x len(alphas)`` coefficients.
    """
    # An unpenalized intercept is the same as centering both sides first
    return celer_path(
        X - X.mean(axis=0), y - y.mean(), 'lasso', alphas=alphas, tol=TOLERANCE
    )[1]

def _select_alpha(
    statistics: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    n_obs: int,
    n_folds: int = 10,
    n_seeds: int = 20,
) -> float:
    """Choose the LASSO penalty by repeated k-fold cross-validation over months.

    The months are split at random into ``n_folds`` disjoint subsamples. For each
    fold, the LASSO is fit on statistics from the other months and scored by squared
    error on statistics from the held-out months. As in the paper, the penalties
    chosen under many random splits are averaged.

    Args:
        statistics (Callable): Maps an array of month positions to the ``(X, y)``
            built from just those months.
        n_obs (int): Number of months.
        n_folds (int): Number of folds.
        n_seeds (int): Number of random splits to average over.

    Returns:
        float: Average of the penalties minimizing validation error.
    """
    alphas = _alpha_grid(*statistics(np.arange(n_obs)))
    everything = np.arange(n_obs)
    chosen = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(SEED + seed)
        errors = np.zeros(len(alphas))
        for fold in np.array_split(rng.permutation(n_obs), n_folds):
            X_train, y_train = statistics(np.setdiff1d(everything, fold))
            X_valid, y_valid = statistics(fold)
            coefficients = _lasso_path(X_train, y_train, alphas)
            intercepts = y_train.mean() - X_train.mean(axis=0) @ coefficients
            predictions = X_valid @ coefficients + intercepts
            errors += ((y_valid[:, None] - predictions) ** 2).mean(axis=0)
        chosen.append(alphas[errors.argmin()])
    return float(np.mean(chosen))

def _survivors(X: np.ndarray, y: np.ndarray, alpha: float, names: pd.Index) -> set[str]:
    """Names of the regressors with a nonzero LASSO coefficient.

    Args:
        X (np.ndarray): Regressors, one column per name.
        y (np.ndarray): Response.
        alpha (float): Penalty strength.
        names (pd.Index): Column names of ``X``.

    Returns:
        set[str]: Names of the surviving regressors.
    """
    lasso = Lasso(alpha=alpha, tol=TOLERANCE, fit_intercept=True)
    lasso.fit(X, y)
    return set(names[np.asarray(lasso.coef_) != 0])

def _first_stage_inputs(
    test_assets: np.ndarray, factors: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build the statistics of the first LASSO for a subset of months.

    Args:
        test_assets (np.ndarray): Test-asset excess returns, ``T x n``.
        factors (np.ndarray): Control factor returns, ``T x p``.

    Returns:
        Callable: Maps month positions to the weighted covariances with the controls
            and the average returns of the test assets.
    """
    scale = _penalty_scale(test_assets, factors)

    def statistics(months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        covariances = _covariances(test_assets[months], factors[months])
        return covariances * scale, test_assets[months].mean(axis=0)

    return _rescale(statistics, len(factors))

def _second_stage_inputs(
    test_assets: np.ndarray, test_factor: np.ndarray, factors: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build the statistics of the second LASSO for a subset of months.

    Args:
        test_assets (np.ndarray): Test-asset excess returns, ``T x n``.
        test_factor (np.ndarray): Returns of the factor being tested, length ``T``.
        factors (np.ndarray): Control factor returns, ``T x p``.

    Returns:
        Callable: Maps month positions to the covariances of the test assets with
            the controls and with the test factor.
    """
    def statistics(months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        covariances = _covariances(test_assets[months], factors[months])
        target = _covariances(test_assets[months], test_factor[months, None])[:, 0]
        return covariances, target

    return _rescale(statistics, len(factors))

def lasso_one(
    alpha: float | None,
    factors: pd.DataFrame,
    test_assets: pd.DataFrame | None = None,
    n_seeds: int = 20,
) -> set[str]:
    """First selection. Runs a cross-sectional LASSO of the test assets' average
    returns on their covariances with the control factors, and returns the factors
    with nonzero coefficients. Factors whose univariate betas are small are
    penalized more heavily.

    Args:
        alpha (float | None): Penalty strength (celer's ``alpha``) on standardized
            covariances and returns, or ``None`` to choose it by cross-validation.
        factors (pd.DataFrame): Monthly returns of the control factors, one column
            per factor.
        test_assets (pd.DataFrame | None): Monthly excess returns of the test assets,
            or ``None`` to load the Fama-French 25.
        n_seeds (int): Cross-validation splits to average over when ``alpha`` is
            ``None``.

    Returns:
        set[str]: Names of the selected control factors.
    """
    if test_assets is None:
        test_assets = _load_test_assets()
    factors, test_assets = _align_monthly(factors, test_assets)
    statistics = _first_stage_inputs(test_assets.to_numpy(), factors.to_numpy())
    if alpha is None:
        alpha = _select_alpha(statistics, len(factors), n_seeds=n_seeds)
    return _survivors(*statistics(np.arange(len(factors))), alpha, factors.columns)

def lasso_two(
    alpha: float | None,
    test_factor: pd.Series,
    factors: pd.DataFrame,
    test_assets: pd.DataFrame | None = None,
    n_seeds: int = 20,
) -> set[str]:
    """Second selection. Runs a cross-sectional LASSO of the test assets' covariances
    with the test factor on their covariances with the control factors, and returns
    the controls with nonzero coefficients. These are the controls whose exposures
    resemble the test factor's, which would bias its estimate if left out.

    Args:
        alpha (float | None): Penalty strength (celer's ``alpha``) on standardized
            covariances and returns, or ``None`` to choose it by cross-validation.
        test_factor (pd.Series): Monthly returns of the factor being tested.
        factors (pd.DataFrame): Control factor returns, not including the test factor.
        test_assets (pd.DataFrame | None): Monthly excess returns of the test assets,
            or ``None`` to load the Fama-French 25.
        n_seeds (int): Cross-validation splits to average over when ``alpha`` is
            ``None``.

    Returns:
        set[str]: Names of the selected control factors.
    """
    if test_assets is None:
        test_assets = _load_test_assets()
    factors, test_factor, test_assets = _align_monthly(factors, test_factor, test_assets)
    statistics = _second_stage_inputs(
        test_assets.to_numpy(), test_factor.to_numpy(), factors.to_numpy()
    )
    if alpha is None:
        alpha = _select_alpha(statistics, len(factors), n_seeds=n_seeds)
    return _survivors(*statistics(np.arange(len(factors))), alpha, factors.columns)

def _inference_controls(
    test_factor: pd.Series,
    factors: pd.DataFrame,
    alpha: float | None,
    n_seeds: int,
) -> set[str]:
    """Controls used to orthogonalize the test factor for standard errors.

    Runs a time-series LASSO of the test factor on the control factors (equation 10
    of the paper) and returns the factors with nonzero coefficients.

    Args:
        test_factor (pd.Series): Monthly returns of the factor being tested.
        factors (pd.DataFrame): Control factor returns, aligned to ``test_factor``.
        alpha (float | None): Penalty strength, or ``None`` to cross-validate.
        n_seeds (int): Cross-validation splits to average over.

    Returns:
        set[str]: Names of the selected control factors.
    """
    X = factors.to_numpy()
    y = test_factor.to_numpy()

    def raw_statistics(months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X[months], y[months]

    statistics = _rescale(raw_statistics, len(y))
    if alpha is None:
        alpha = _select_alpha(statistics, len(y), n_seeds=n_seeds)
    return _survivors(*statistics(np.arange(len(y))), alpha, factors.columns)

def post_selection_estimate(
    test_assets: pd.DataFrame,
    test_factor: pd.Series,
    factors: pd.DataFrame,
    selected: set[str],
    orthogonalizing: set[str],
    lags: int | None = None,
) -> tuple[float, float, float]:
    """Post-selection estimation. Regresses the test assets' average returns on a
    constant, their covariances with the test factor, and their covariances with
    the selected controls (equation 9 of the paper). The coefficient on the test
    factor is its SDF loading; its standard error is the Newey-West-type estimator
    of the paper's Theorem 2.

    Args:
        test_assets (pd.DataFrame): Monthly excess returns of the test assets.
        test_factor (pd.Series): Monthly returns of the factor being tested.
        factors (pd.DataFrame): Control factor returns, not including the test factor.
        selected (set[str]): Controls kept by the two selections, in the union.
        orthogonalizing (set[str]): Controls kept by the time-series LASSO of the
            test factor on the controls, used only for the standard error.
        lags (int | None): Newey-West lags, or ``None`` for ``floor(4 (T/100)^(2/9))``.

    Returns:
        tuple[float, float, float]: SDF loading of the test factor, that loading
            times the factor's variance, and its standard error.
    """
    factors, test_factor, test_assets = _align_monthly(factors, test_factor, test_assets)
    controls = [name for name in factors.columns if name in selected]
    T, n = len(factors), test_assets.shape[1]
    if len(controls) + 2 > n:
        raise ValueError(
            f'{len(controls)} selected controls leave too few of the {n} test assets '
            'for the cross-sectional regression; use a stronger penalty'
        )

    R = test_assets.to_numpy()
    g = test_factor.to_numpy() - test_factor.mean()
    H = factors[controls].to_numpy()
    H = H - H.mean(axis=0)

    # Cross-sectional OLS of average returns on covariances (equation 9)
    regressors = np.column_stack([
        np.ones(n),
        _covariances(R, g[:, None])[:, 0],
        _covariances(R, H),
    ])
    coefficients = np.linalg.lstsq(regressors, R.mean(axis=0), rcond=None)[0]
    loading = float(coefficients[1])

    # Newey-West variance of the loading (Theorem 2, one tested factor)
    pricing_kernel = 1 - coefficients[1] * g - H @ coefficients[2:]
    other = factors[[name for name in factors.columns if name in orthogonalizing]]
    other = other.to_numpy() - other.to_numpy().mean(axis=0)
    residual = g - other @ np.linalg.lstsq(other, g, rcond=None)[0] if other.size else g
    score = pricing_kernel * residual / np.mean(residual ** 2)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    variance = np.mean(score ** 2)
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * np.sum(score[lag:] * score[:-lag]) / T
    std_error = float(np.sqrt(variance / T))
    return loading, loading * float(test_factor.var(ddof=0)), std_error

def double_selection_test(
    test_factor: str,
    factors: pd.DataFrame,
    test_assets: pd.DataFrame | None = None,
    alpha_one: float | None = None,
    alpha_two: float | None = None,
    alpha_three: float | None = None,
    n_seeds: int = 20,
    lags: int | None = None,
) -> DoubleSelectionResult:
    """Test whether a factor adds pricing information beyond all the others.

    Runs both selections, takes the union of the controls they keep, and estimates
    the test factor's SDF loading with those controls. A loading that is
    statistically indistinguishable from zero means the other factors already price
    whatever the test factor prices.

    Args:
        test_factor (str): Column of ``factors`` to test.
        factors (pd.DataFrame): Monthly returns of all factors, one column per factor.
        test_assets (pd.DataFrame | None): Monthly excess returns of the test assets,
            or ``None`` to load the Fama-French 25.
        alpha_one (float | None): Penalty of the first selection, or ``None`` to
            cross-validate.
        alpha_two (float | None): Penalty of the second selection, or ``None`` to
            cross-validate.
        alpha_three (float | None): Penalty of the time-series LASSO used for the
            standard error, or ``None`` to cross-validate.
        n_seeds (int): Cross-validation splits to average over for each penalty.
        lags (int | None): Newey-West lags, or ``None`` for the default.

    Returns:
        DoubleSelectionResult: Loading, standard error, t-statistic, and the sets of
            controls kept by each selection.
    """
    if test_assets is None:
        test_assets = _load_test_assets()
    controls = factors.drop(columns=test_factor)
    target = factors[test_factor]
    controls, target, test_assets = _align_monthly(controls, target, test_assets)

    first = lasso_one(alpha_one, controls, test_assets, n_seeds)
    second = lasso_two(alpha_two, target, controls, test_assets, n_seeds)
    orthogonalizing = _inference_controls(target, controls, alpha_three, n_seeds)
    loading, scaled, std_error = post_selection_estimate(
        test_assets, target, controls, first | second, orthogonalizing, lags
    )
    return DoubleSelectionResult(
        loading=loading,
        scaled_loading=scaled,
        std_error=std_error,
        t_stat=loading / std_error,
        first_selection=first,
        second_selection=second,
    )

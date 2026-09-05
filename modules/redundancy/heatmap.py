"""Generate correlation heatmaps for merged factor-return data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from modules.factors.mappings import FACTOR_MAPPINGS

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def select_factor_returns(
    data: pd.DataFrame,
    theme: str | None = None,
) -> pd.DataFrame:
    """Select registered factor-return columns belonging to an optional theme.

    Args:
        data (pd.DataFrame): Merged return data with factor names as columns.
        theme (str | None): Factor module name to select, or ``None`` for all factors.

    Returns:
        pd.DataFrame: Numeric returns for the registered factors in the requested theme.
    """
    normalized_theme = _normalize_theme(theme)
    columns = [
        name
        for name, definition in FACTOR_MAPPINGS.items()
        if name in data.columns
        and (
            normalized_theme is None
            or _factor_theme(definition['func']) == normalized_theme
        )
    ]
    if not columns:
        scope = 'any registered factors' if normalized_theme is None else (
            f"factors from the '{normalized_theme}' theme"
        )
        raise ValueError(f'Merged return data does not contain {scope}')

    return data.loc[:, columns].apply(pd.to_numeric, errors='coerce')


def calculate_factor_correlations(
    data: pd.DataFrame,
    theme: str | None = None,
) -> pd.DataFrame:
    """Calculate Pearson correlations for factor returns in an optional theme.

    Args:
        data (pd.DataFrame): Merged return data with factor names as columns.
        theme (str | None): Factor module name to select, or ``None`` for all factors.

    Returns:
        pd.DataFrame: Symmetric factor-return correlation matrix.
    """
    returns = select_factor_returns(data=data, theme=theme)
    return returns.corr(method='pearson')


def create_correlation_heatmap(
    data: pd.DataFrame,
    theme: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Create a red Pearson-correlation heatmap for an optional factor theme.

    Args:
        data (pd.DataFrame): Merged return data with factor names as columns.
        theme (str | None): Factor module name to select, or ``None`` for all factors.
        ax (Axes | None): Matplotlib axes to draw on, or ``None`` to create new axes.

    Returns:
        Axes: Matplotlib axes containing the correlation heatmap.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Creating a correlation heatmap requires matplotlib and seaborn'
        ) from exc

    correlations = calculate_factor_correlations(data=data, theme=theme)
    if ax is None:
        size = max(6.0, 0.8 * len(correlations.columns) + 2.0)
        _, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        correlations,
        ax=ax,
        annot=True,
        cmap='Reds',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson correlation'},
    )
    title_theme = 'All' if theme is None else _normalize_theme(theme).title()
    ax.set_title(f'{title_theme} Factor Return Correlations')
    ax.set_xlabel('Factor')
    ax.set_ylabel('Factor')
    return ax


def _normalize_theme(theme: str | None) -> str | None:
    """Normalize and validate an optional factor-module theme name.

    Args:
        theme (str | None): Factor module name, optionally ending in ``.py``.

    Returns:
        str | None: Normalized module name, or ``None`` when no theme was supplied.
    """
    if theme is None:
        return None
    normalized = theme.strip().lower().removesuffix('.py')
    themes = {_factor_theme(item['func']) for item in FACTOR_MAPPINGS.values()}
    if normalized not in themes:
        available = ', '.join(sorted(themes))
        raise ValueError(
            f"Unknown factor theme '{theme}'; available themes: {available}"
        )
    return normalized


def _factor_theme(func: object) -> str:
    """Return the source-module name for a registered factor function.

    Args:
        func (object): Registered factor function with a ``__module__`` attribute.

    Returns:
        str: Final component of the function's source-module path.
    """
    return str(getattr(func, '__module__', '')).rsplit('.', maxsplit=1)[-1]


__all__ = [
    'calculate_factor_correlations',
    'create_correlation_heatmap',
    'select_factor_returns',
]

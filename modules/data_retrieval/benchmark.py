"""Retrieve benchmark market returns and risk-free rates from FRED."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

_FRED_OBSERVATIONS_URL = 'https://api.stlouisfed.org/fred/series/observations'

def retrieve_benchmark(
    start_date: str | date,
    end_date: str | date,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve daily S&P 500 returns and 10-year Treasury rates from FRED.

    Args:
        start_date (str | date): Inclusive start date in ``YYYY-MM-DD`` format.
        end_date (str | date): Inclusive end date in ``YYYY-MM-DD`` format.
        api_key (str | None): FRED API key, or ``None`` to use ``FRED_API_KEY`` from the environment or ``.env``.

    Returns:
        pd.DataFrame: Daily observations with date, market_return, and risk_free_rate columns expressed as decimals.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError('start_date must be on or before end_date')

    key = api_key or _load_api_key()
    # Fetch a short lookback so the first requested date can have a return.
    observation_start = start - pd.Timedelta(days=10)
    sp500 = _retrieve_fred_series('SP500', observation_start, end, key)
    treasury = _retrieve_fred_series('DGS10', observation_start, end, key)

    result = pd.DataFrame({
        'market_return': sp500.dropna().pct_change(fill_method=None),
        'risk_free_rate': treasury / 100.0,
    })
    # Treasury observations can be absent on market holidays; carry forward the
    # latest published yield and retain only S&P 500 observation dates.
    result['risk_free_rate'] = result['risk_free_rate'].ffill()
    result = result.loc[sp500.notna() & result.index.to_series().between(start, end)]
    result.index.name = 'date'
    return result.reset_index()[['date', 'market_return', 'risk_free_rate']]

def _load_api_key() -> str:
    """Load the FRED API key from the environment or the project's ``.env`` file.

    Args:
        None.

    Returns:
        str: Configured FRED API key.
    """
    load_dotenv()
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError('FRED_API_KEY is not set; add it to .env or pass api_key explicitly')
    return api_key

def _retrieve_fred_series(
    series_id: str,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    api_key: str,
) -> pd.Series:
    """Retrieve and numerically parse one FRED series for an inclusive date range.

    Args:
        series_id (str): FRED series identifier, such as ``'SP500'`` or ``'DGS10'``.
        start_date (str | date | pd.Timestamp): Inclusive observation start date.
        end_date (str | date | pd.Timestamp): Inclusive observation end date.
        api_key (str): FRED API key used to authenticate the request.

    Returns:
        pd.Series: Numeric observations indexed by normalized dates, with missing values represented by ``NaN``.
    """
    response = requests.get(
        _FRED_OBSERVATIONS_URL,
        params={
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'observation_start': str(pd.Timestamp(start_date).date()),
            'observation_end': str(pd.Timestamp(end_date).date()),
        },
        timeout=30,
    )
    response.raise_for_status()
    observations = response.json().get('observations', [])
    series = pd.Series(
        (observation.get('value') for observation in observations),
        index=pd.to_datetime([observation['date'] for observation in observations]),
        name=series_id,
        dtype='object',
    )
    return pd.to_numeric(series, errors='coerce').sort_index()

__all__ = ['retrieve_benchmark']

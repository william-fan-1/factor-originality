"""Retrieve price-related data"""

import pandas as pd
import yfinance as yf

def retrieve_prices(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Retrieve daily prices, volume, and shares outstanding for a ticker.

    Args:
        ticker (str): Yahoo Finance ticker symbol to query.
        start_date (str): Inclusive start date in '%Y-%m-%d' format.
        end_date (str): Exclusive end date in a '%Y-%m-%d' format.

    Returns:
        pd.DataFrame: Daily observations with ticker, date, close,
            adjusted_close, volume, and shares_outstanding columns.
    """
    security = yf.Ticker(ticker)
    history = security.history(
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
    )

    result = history.rename(
        columns={'Close': 'close', 'Adj Close': 'adjusted_close', 'Volume': 'volume'}
    )[['close', 'adjusted_close', 'volume']].copy()

    result.index = pd.to_datetime(result.index).tz_localize(None)
    result.index = result.index.normalize()
    result['shares_outstanding'] = _get_shares_outstanding(
        security,
        result.index,
    )
    result.insert(0, 'date', result.index)
    result.insert(0, 'ticker', ticker)
    return result.reset_index(drop=True)

def _get_shares_outstanding(
    security: yf.Ticker,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Align yfinance share-count data with the retrieved trading dates.

    Args:
        security (yf.Ticker): Yahoo Finance ticker object to query.
        dates (pd.DatetimeIndex): Trading dates to which share counts are aligned.

    Returns:
        pd.Series: Shares outstanding indexed by the requested trading dates.
    """
    historical_shares = security.get_shares_full()
    if historical_shares is not None and not historical_shares.empty:
        historical_shares.index = pd.to_datetime(historical_shares.index)
        historical_shares.index = historical_shares.index.tz_localize(None).normalize()
        historical_shares = historical_shares.sort_index()
        historical_shares = historical_shares[
            ~historical_shares.index.duplicated(keep='last')
        ]
        return historical_shares.reindex(dates, method='ffill')

    current_shares = security.info.get('sharesOutstanding')
    return pd.Series(current_shares, index=dates, dtype='float64')
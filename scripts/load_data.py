"""Script to retrieve data requried to create factors

Of the initial 20 factors, there were 3 major themes of data that could
be inferred:
- Market (Prices + Shares)
- Fundamentals/Financial Statements
- Benchmarks

This script is designed to load in tickers from 2015, 2020, 2025
then save data for each ticker to a cache within the data directory
for future use (and to not have to recompute every call)
"""
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from modules.data_retrieval.benchmark import retrieve_benchmark
from modules.data_retrieval.prices import retrieve_prices
from modules.data_retrieval.fundamentals import load_fundamentals

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

def check_cache(
    cache_path: Path,
    func: Callable[..., pd.DataFrame],
    **kwargs: object,
) -> pd.DataFrame:
    """Load a cached DataFrame or call a loader with the supplied keyword arguments and cache its result.

    Args:
        cache_path (Path): Parquet file to read from or write to.
        func (Callable[..., pd.DataFrame]): Data loader called when the cache does not exist.
        **kwargs (object): Keyword arguments forwarded unchanged to ``func``.

    Returns:
        pd.DataFrame: Data loaded from the existing cache or returned by ``func``.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = func(**kwargs)
    data.to_parquet(cache_path, index=False)
    return data

def main() -> None:
    for year in range(2015, 2026, 5):
        ticker_path = DATA_DIR / 'tickers' / f'tickers_{year}.csv'
        tickers = pd.read_csv(ticker_path)['Ticker']

        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year + 4, month=12, day=31)
        print(f'Processing {start_date.date()} -> {end_date.date()}')

        cache_dir = DATA_DIR / 'raw_data' / f'{year}_{year + 4}'
        check_cache(
            cache_path=cache_dir / 'benchmark.parquet',
            func=retrieve_benchmark,
            start_date=start_date,
            end_date=end_date,
        )

        for ticker in tqdm(tickers, desc=f'{year}-{year + 4}'):
            check_cache(
                cache_path=cache_dir / f'{ticker}_prices.parquet',
                func=retrieve_prices,
                ticker=ticker,
                start_date=start_date,
                # yfinance uses an exclusive end date.
                end_date=end_date + pd.Timedelta(days=1),
            )

            # Omitting metrics requests every supported fundamental.
            check_cache(
                cache_path=cache_dir / f'{ticker}_fundamentals.parquet',
                func=load_fundamentals,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

if __name__ == '__main__':
    main()

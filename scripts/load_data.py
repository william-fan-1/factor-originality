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
import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

# Direct execution adds ``scripts`` to sys.path, so add the repository root
# before importing sibling top-level modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_retrieval.benchmark import retrieve_benchmark
from modules.data_retrieval.prices import retrieve_prices
from modules.data_retrieval.fundamentals import load_fundamentals

DATA_DIR = PROJECT_ROOT / 'data'

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
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f'{func.__name__} returned no rows')
    temporary_path = cache_path.with_suffix(f'{cache_path.suffix}.tmp')
    try:
        data.to_parquet(temporary_path, index=False)
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return data

def _cache_ticker(
    ticker: str,
    cache_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    """Populate price and fundamental caches for one ticker.

    Args:
        ticker (str): Security ticker to retrieve.
        cache_dir (Path): Directory in which the ticker caches are stored.
        start_date (pd.Timestamp): Inclusive analysis start date.
        end_date (pd.Timestamp): Inclusive analysis end date.

    Returns:
        str: The successfully cached ticker.
    """
    check_cache(
        cache_path=cache_dir / f'{ticker}_prices.parquet',
        func=retrieve_prices,
        ticker=ticker,
        start_date=start_date,
        # yfinance uses an exclusive end date.
        end_date=end_date + pd.Timedelta(days=1),
    )
    check_cache(
        cache_path=cache_dir / f'{ticker}_fundamentals.parquet',
        func=load_fundamentals,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    return ticker

def main(years: Sequence[int] = (2015,), workers: int = 1) -> None:
    """Populate five-year raw-data caches for requested cohort years.

    Args:
        years (Sequence[int]): Starting years of five-year analysis periods.
        workers (int): Number of tickers to retrieve concurrently.

    Returns:
        None.
    """
    for year in years:
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

        failed_tickers: list[str] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _cache_ticker, ticker, cache_dir, start_date, end_date
                ): ticker
                for ticker in tickers
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f'{year}-{year + 4}'
            ):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failed_tickers.append(ticker)
                    tqdm.write(
                        f'FAILED {ticker}: {type(error).__name__}: {error}'
                    )

        if failed_tickers:
            print(f'Failed tickers for {year}-{year + 4}: {", ".join(failed_tickers)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'years',
        nargs='*',
        type=int,
        default=[2015],
        help='Starting years of five-year periods to populate (default: 2015).',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        choices=range(1, 9),
        metavar='1-8',
        help='Tickers to retrieve concurrently (default: 1).',
    )
    arguments = parser.parse_args()
    main(years=arguments.years, workers=arguments.workers)

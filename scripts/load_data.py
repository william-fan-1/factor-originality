"""Script to retrieve data required to create factors

Of the initial 20 factors, there were 3 major themes of data that could
be inferred:
- Market (Prices + Shares)             -> CRSP via WRDS
- Fundamentals/Financial Statements    -> Compustat via WRDS
- Benchmarks (monthly)                 -> CRSP / Fama-French via WRDS

Pulls every security in the S&P 500 universe file (built by
``build_crsp_universe.py``) for the span from January 1 of the start year to
December 31 of the end year, and saves one price and one fundamentals parquet
per security to ``data/raw_data/{start}_{end}/``. Securities are retrieved by
PERMNO, so ticker changes, share classes and reused tickers cannot mix
companies. Existing files are skipped, so an interrupted run can simply be
started again.
"""
import argparse
from collections.abc import Callable
from pathlib import Path
import sys
import traceback

import pandas as pd
from tqdm import tqdm

# Direct execution adds ``scripts`` to sys.path, so add the repository root
# before importing sibling top-level modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_retrieval.benchmark import retrieve_benchmark
from modules.data_retrieval.fundamentals import load_fundamentals
from modules.data_retrieval.prices import retrieve_prices
from modules.data_retrieval.utils import connection, reset_connection

DATA_DIR = PROJECT_ROOT / 'data'
HISTORY_QUARTERS = 9
MAX_ATTEMPTS = 2


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

    data = func(**kwargs)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f'{func.__name__} returned no rows')
    _write_parquet(data, cache_path)
    return data


def _write_parquet(data: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically so a crash never leaves a partial cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f'{path.suffix}.tmp')
    try:
        data.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_universe(path: Path) -> dict[str, int]:
    """Read ``{ticker label: permno}`` from a universe CSV with Ticker and Permno columns."""
    universe = pd.read_csv(path).dropna(subset=['Ticker', 'Permno'])
    universe['Ticker'] = universe['Ticker'].astype(str).str.strip()
    universe['Permno'] = universe['Permno'].astype('int64')
    universe = universe.drop_duplicates('Ticker')
    return dict(zip(universe['Ticker'], universe['Permno']))


def _is_database_error(error: Exception) -> bool:
    """Whether an error came from the WRDS connection rather than our own code."""
    return type(error).__module__.split('.')[0] in {'sqlalchemy', 'psycopg2', 'psycopg', 'wrds'}


def _cache_chunk(
    securities: dict[str, int],
    cache_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, str]:
    """Query WRDS for one batch of securities and write their per-ticker caches.

    Args:
        securities (dict[str, int]): ``{ticker label: permno}`` to retrieve.
        cache_dir (Path): Directory receiving the per-ticker parquet files.
        start_date (pd.Timestamp): Inclusive analysis start date.
        end_date (pd.Timestamp): Inclusive analysis end date.

    Returns:
        dict[str, str]: Status per ticker: ``complete``, ``missing_prices``,
            ``missing_fundamentals``, or ``missing_both``.
    """
    permnos = list(securities.values())
    prices = retrieve_prices(
        permnos,
        # Two years expose all eight quarters required by year-over-year TTM
        # signals after fundamentals are aligned to daily observations.
        start_date=start_date - pd.DateOffset(years=2),
        # The price query treats the end date as exclusive.
        end_date=end_date + pd.Timedelta(days=1),
    )
    fundamentals = load_fundamentals(
        permnos, start_date, end_date, history_quarters=HISTORY_QUARTERS
    )
    prices_by_permno = dict(tuple(prices.groupby('permno')))
    fundamentals_by_permno = dict(tuple(fundamentals.groupby('permno')))

    status: dict[str, str] = {}
    for ticker, permno in securities.items():
        price_frame = prices_by_permno.get(permno)
        fundamental_frame = fundamentals_by_permno.get(permno)
        if price_frame is not None:
            _write_parquet(
                price_frame.assign(ticker=ticker)[['ticker', *price_frame.columns]],
                cache_dir / f'{ticker}_prices.parquet',
            )
        if fundamental_frame is not None:
            _write_parquet(
                fundamental_frame.assign(ticker=ticker)[['ticker', *fundamental_frame.columns]],
                cache_dir / f'{ticker}_fundamentals.parquet',
            )
        if price_frame is not None and fundamental_frame is not None:
            status[ticker] = 'complete'
        elif price_frame is not None:
            status[ticker] = 'missing_fundamentals'
        elif fundamental_frame is not None:
            status[ticker] = 'missing_prices'
        else:
            status[ticker] = 'missing_both'
    return status


def main(
    start_year: int = 2004,
    end_year: int = 2024,
    universe_path: Path | None = None,
    chunk_size: int = 50,
) -> None:
    """Populate the raw-data cache for every universe security over a year range.

    Args:
        start_year (int): First year; the span starts on January 1.
        end_year (int): Last year; the span ends on December 31.
        universe_path (Path | None): CSV with ``Ticker`` and ``Permno`` columns;
            defaults to ``data/sp500_universe/crsp_universe_{start}_{end}.csv``.
        chunk_size (int): Securities per WRDS query.

    Returns:
        None.
    """
    if start_year > end_year:
        raise ValueError('start_year must not be after end_year')
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)
    if universe_path is None:
        universe_path = DATA_DIR / 'sp500_universe' / f'crsp_universe_{start_year}_{end_year}.csv'
    if not universe_path.exists():
        raise FileNotFoundError(
            f'Universe file not found: {universe_path}\n'
            f'Run: python scripts/build_crsp_universe.py --start {start_year} --end {end_year}'
        )
    universe = _load_universe(universe_path)
    print(f'Processing {start_date.date()} -> {end_date.date()} ({len(universe)} securities)')

    cache_dir = DATA_DIR / 'raw_data' / f'{start_year}_{end_year}'
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        check_cache(
            cache_path=cache_dir / 'benchmark.parquet',
            func=retrieve_benchmark,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as error:
        print(f'WARNING: benchmark not cached ({type(error).__name__}: {error})')

    todo = {
        ticker: permno for ticker, permno in universe.items()
        if not (
            (cache_dir / f'{ticker}_prices.parquet').exists()
            and (cache_dir / f'{ticker}_fundamentals.parquet').exists()
        )
    }
    print(f'{len(universe) - len(todo)} already cached, {len(todo)} to retrieve')
    if not todo:
        return

    connection()  # authenticate once up front so a bad login fails immediately
    results: dict[str, str] = {}
    items = list(todo.items())
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    for chunk in tqdm(chunks, desc=f'{start_year}-{end_year}'):
        first = next(iter(chunk))
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                results.update(_cache_chunk(chunk, cache_dir, start_date, end_date))
                break
            except Exception as error:
                frame = traceback.extract_tb(error.__traceback__)[-1]
                tqdm.write(
                    f'Batch starting {first} failed (attempt {attempt}/{MAX_ATTEMPTS}): '
                    f'{type(error).__name__}: {error} '
                    f'[{Path(frame.filename).name}:{frame.lineno} in {frame.name}]'
                )
                if _is_database_error(error):
                    reset_connection()
                else:
                    break  # a data or code error will fail identically on retry
        if first not in results:
            results.update({ticker: 'failed' for ticker in chunk})

    outcomes = pd.Series(results, name='status')
    print(outcomes.value_counts().to_string())
    problems = outcomes[outcomes.ne('complete')]
    if not problems.empty:
        log_path = DATA_DIR / 'logs' / f'load_data_{start_year}_{end_year}.csv'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        problems.rename_axis('ticker').reset_index().to_csv(log_path, index=False)
        print(f'{len(problems)} securities incomplete; see {log_path.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start_year', nargs='?', type=int, default=2004)
    parser.add_argument('end_year', nargs='?', type=int, default=2024)
    parser.add_argument(
        '--universe',
        type=Path,
        default=None,
        help='Universe CSV with Ticker and Permno columns '
             '(default: data/sp500_universe/crsp_universe_{start}_{end}.csv).',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50,
        help='Securities per WRDS query (default: 50).',
    )
    arguments = parser.parse_args()
    main(
        start_year=arguments.start_year,
        end_year=arguments.end_year,
        universe_path=arguments.universe,
        chunk_size=arguments.chunk_size,
    )

"""Build S&P 500 historical universe from FolioIQ constituent change events.

Replays the full addition/removal event log back to the 1957 founding to
produce point-in-time membership intervals, then filters to a target window.

Outputs (written to data/sp500_universe/):
  constituents_events.parquet          cleaned event log with parsed dates
  universe_{start}_{end}.parquet       membership intervals for the window
  universe_{start}_{end}.csv           unique tickers (Name, Ticker) for the window
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyfolioiq
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'data' / 'sp500_universe'


def fetch_events(client: pyfolioiq.FolioIQClient) -> pd.DataFrame:
    """Fetch and clean the full historical constituent event log."""
    raw = client.get_historical_sp500_constituents()
    rows = [r.model_dump() if hasattr(r, 'model_dump') else r for r in raw]
    df = pd.DataFrame(rows)

    df['date'] = pd.to_datetime(df['date_added'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    df['removed_ticker'] = df['removed_ticker'].replace('', None)
    return df


def build_membership_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Replay the event log and return (ticker, name, date_added, date_removed) rows.

    Simulates index membership from the 1957 founding forward so every
    historical constituent gets a precise entry and exit date.
    """
    intervals: list[dict] = []
    # ticker -> (date_added, name)
    active: dict[str, tuple[pd.Timestamp, str]] = {}

    for _, row in events.iterrows():
        removed = row['removed_ticker']
        if pd.notna(removed) and removed:
            if removed in active:
                add_date, name = active.pop(removed)
                intervals.append(
                    {
                        'ticker': removed,
                        'name': name,
                        'date_added': add_date,
                        'date_removed': row['date'],
                    }
                )
            else:
                # Ticker removed but wasn't in active tracking (data gap)
                intervals.append(
                    {
                        'ticker': removed,
                        'name': None,
                        'date_added': None,
                        'date_removed': row['date'],
                    }
                )

        active[row['symbol']] = (row['date'], row['added_security'])

    # Remaining active tickers are still in the index
    for ticker, (add_date, name) in active.items():
        intervals.append(
            {
                'ticker': ticker,
                'name': name,
                'date_added': add_date,
                'date_removed': pd.NaT,
            }
        )

    return pd.DataFrame(intervals)


def filter_universe(
    intervals: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Keep rows representing any membership overlap with [start, end]."""
    in_window = (
        intervals['date_added'].notna()
        & (intervals['date_added'] <= end)
        & (intervals['date_removed'].isna() | (intervals['date_removed'] >= start))
    )
    return intervals[in_window].copy()


def main(start_year: int = 2005, end_year: int = 2025) -> None:
    start = pd.Timestamp(year=start_year, month=1, day=1)
    end = pd.Timestamp(year=end_year, month=12, day=31)

    client = pyfolioiq.FolioIQClient()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Fetching historical S&P 500 constituents from FolioIQ...')
    events = fetch_events(client)
    print(
        f'  {len(events)} events  '
        f'({events['date'].min().date()} → {events['date'].max().date()})'
    )

    events_path = OUT_DIR / 'constituents_events.parquet'
    events.to_parquet(events_path, index=False)
    print(f'  Event log → {events_path.relative_to(PROJECT_ROOT)}')

    print('Replaying membership intervals...')
    intervals = build_membership_intervals(events)

    print(f'Filtering to {start.date()} – {end.date()}...')
    universe = filter_universe(intervals, start, end)
    unique_tickers = universe['ticker'].nunique()
    print(f'  {unique_tickers} unique tickers in window')

    label = f'{start_year}_{end_year}'

    parquet_path = OUT_DIR / f'universe_{label}.parquet'
    universe.to_parquet(parquet_path, index=False)
    print(f'  Intervals → {parquet_path.relative_to(PROJECT_ROOT)}')

    csv_path = OUT_DIR / f'universe_{label}.csv'
    (
        universe[['name', 'ticker']]
        .drop_duplicates('ticker')
        .rename(columns={'name': 'Name', 'ticker': 'Ticker'})
        .sort_values('Ticker')
        .to_csv(csv_path, index=False)
    )
    print(f'  CSV      → {csv_path.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--start',
        type=int,
        default=2005,
        metavar='YEAR',
        help='First year of the target window (default: 2005).',
    )
    parser.add_argument(
        '--end',
        type=int,
        default=2025,
        metavar='YEAR',
        help='Last year of the target window (default: 2025).',
    )
    args = parser.parse_args()
    main(start_year=args.start, end_year=args.end)

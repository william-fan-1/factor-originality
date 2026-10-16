"""Build the S&P 500 universe for a year range from CRSP's membership list.

Writes to data/sp500_universe/:
  crsp_universe_{start}_{end}.parquet   permno, ticker, name, start, end
  crsp_universe_{start}_{end}.csv       Name, Ticker, Permno (what load_data.py reads)

Membership comes from ``crsp.msp500list`` (PERMNO-keyed, through the end of
CRSP's data), which avoids the ticker mismatches in the FolioIQ event log.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_retrieval.universe import retrieve_universe

OUT_DIR = PROJECT_ROOT / 'data' / 'sp500_universe'


def main(start_year: int = 2004, end_year: int = 2024, min_member_days: int = 5) -> None:
    start = pd.Timestamp(year=start_year, month=1, day=1)
    end = pd.Timestamp(year=end_year, month=12, day=31)
    universe = retrieve_universe(start, end, min_member_days=min_member_days)
    print(f'{len(universe)} securities in the S&P 500 between {start.date()} and {end.date()}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label = f'{start_year}_{end_year}'
    universe.to_parquet(OUT_DIR / f'crsp_universe_{label}.parquet', index=False)
    (
        universe[['name', 'ticker', 'permno']]
        .rename(columns={'name': 'Name', 'ticker': 'Ticker', 'permno': 'Permno'})
        .to_csv(OUT_DIR / f'crsp_universe_{label}.csv', index=False)
    )
    print(f'Wrote crsp_universe_{label}.parquet and .csv to {OUT_DIR.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=2004, metavar='YEAR')
    parser.add_argument('--end', type=int, default=2024, metavar='YEAR')
    parser.add_argument(
        '--min-days',
        type=int,
        default=5,
        help='Drop securities in the index fewer than this many days, e.g. one-day '
             'spin-off stubs, unless membership runs to the end year (default: 5).',
    )
    arguments = parser.parse_args()
    main(start_year=arguments.start, end_year=arguments.end, min_member_days=arguments.min_days)

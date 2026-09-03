# factor-originality
An experiment designed at identifying original factors for asset pricing

## Project Structure
```text
factor-originality/
|-- data/
|   |-- factors/
|   |-- raw_data/
|   |   `-- 2015_2019/
|   |       `-- <ticker and benchmark Parquet caches>
|   `-- tickers/
|       |-- tickers_2015.csv
|       |-- tickers_2020.csv
|       `-- tickers_2025.csv
|-- dev/
|   |-- benchmark.ipynb
|   |-- fundamentals.ipynb
|   `-- prices.ipynb
|-- modules/
|   |-- data_retrieval/
|   |   |-- benchmark.py
|   |   |-- fundamental_mappings.py
|   |   |-- fundamentals.py
|   |   `-- prices.py
|   |-- factors/
|   |   |-- factor.py
|   |   |-- mappings.py
|   |   |-- momentum.py
|   |   |-- utils.py
|   |   `-- value.py
|   `-- redundancy/
|       |-- heatmap.py
|       |-- pca.py
|       `-- spanning_regression.py
|-- notebooks/
|-- scripts/
|   `-- load_data.py
|-- .gitignore
`-- README.md
```

## Usage
All data I used currently sits in the data directory. If you wanted to re-run my script yourself, after cloning, you `cd` into the project root and run the following: `python .\scripts\load_data.py year --workers int`, where `year` is the start of a 5 year period you want to download. You can download multiple years: i.e., `python .\scripts\load_data.py 2015 2020 2025 --workers int` would load all available data from 2015-2030 in 5 year increments. `--workers` can receive an `int` between 1 and 8 to try to speed up the data download process, but I found that anything more than 2 ran into issues.

After downloading data (or using my exisiting data), simply access the notebooks folder and run the notebook corresponding to your desired redundancy measure to see the results.

## Factor construction
Factors were constructued using canonical definitions and in cases of fundamental data where a specific field may not have been queried or retrieved, metrics were recreated using accounting identities to create factors. In the case of "fundamental" factors, if they used a flow variable, such as `net_income`, the 4 most recent quarters would be summed, while stock variables, such as `cash_and_equivalents` were used as is on the balance sheet.

When constructing factors that used fundamental data, I had 2 considerations:
1. I had to align the filing date with the trading date as opposed to simply the "`as_of`" since the market didn't know about the data until the filing was released.
2. For filings that were released after 4 PM ET, I had to align them to the following trading day since traders were unable to "act" on the information until the following day.

In accordance with the procedure followed by Jensen, Kelly, and Pedersen, when constructing factors, I performed the following steps:
- Rank stocks in my universe by factor value
- Construct portfolios of terciles, long the top tercile and short the bottom tercile
- Compute capped-weighted return
- Weight stocks by market cap, winsorizing at 80th percentile to ensure mega-caps do not dominate
- The factor return is then defined as the high-tercile return minus the low-tercile return, corresponding to the excess return of a long-short zero-net-investment strategy. 
# factor-originality
An experiment designed at identifying original factors for asset pricing

## Project Structure
```text
factor_originality/
│
├── modules/
│   └── data_retrieval/
│   │   ├── data_retrieval.py
│   │   ├── prices.py
│   │   ├── fundamentals.py
│   │   ├── benchmark.py
│   │   └── shares.py
│   │
│   ├── preprocessing.py
│   ├── statistical_measures.py
│   └── factors/
│       └── ...
│
├── scripts/
│   └── load_data.py
│
├── data/
│   ├── tickers/
│   │   ├── tickers_2015.csv
│   │   ├── tickers_2020.csv
│   │   └── tickers_2025.csv
│   └── factors/
│       └── ...
│
└── notebooks/
    ├── 01_factor_validation.ipynb
    ├── 02_correlations.ipynb
    ├── 03_spanning_regressions.ipynb
    ├── 04_pca.ipynb
    └── 05_out_of_sample.ipynb
```
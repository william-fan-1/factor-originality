"""Retrieve quarterly fundamental data from SEC filings with edgartools."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Literal, TypeAlias

import pandas as pd
from edgar import Company, set_identity
from dotenv import load_dotenv

Metric: TypeAlias = Literal[
    'revenue', 'cost_of_goods_sold', 'gross_profit', 'operating_income', 'net_income',
    'total_assets', 'total_liabilities', 'shareholders_equity', 'cash_and_equivalents',
    'current_assets', 'current_liabilities', 'property_plant_equipment', 'long_term_debt',
    'short_term_debt', 'operating_cash_flow', 'capital_expenditures',
    'depreciation_amortization', 'stock_issuance', 'stock_repurchases',
]

_METRICS: dict[Metric, tuple[str, tuple[str, ...]]] = {
    'revenue': ('income_statement', ('Revenue', 'Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax')),
    'cost_of_goods_sold': ('income_statement', ('CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold')),
    'gross_profit': ('income_statement', ('GrossProfit',)),
    'operating_income': ('income_statement', ('OperatingIncomeLoss',)),
    'net_income': ('income_statement', ('NetIncomeLoss', 'ProfitLoss')),
    'total_assets': ('balance_sheet', ('Assets',)),
    'total_liabilities': ('balance_sheet', ('Liabilities',)),
    'shareholders_equity': ('balance_sheet', ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')),
    'cash_and_equivalents': ('balance_sheet', ('CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents')),
    'current_assets': ('balance_sheet', ('AssetsCurrent',)),
    'current_liabilities': ('balance_sheet', ('LiabilitiesCurrent',)),
    'property_plant_equipment': ('balance_sheet', ('PropertyPlantAndEquipmentNet',)),
    'long_term_debt': ('balance_sheet', ('LongTermDebtNoncurrent', 'LongTermDebt')),
    'short_term_debt': ('balance_sheet', ('ShortTermBorrowings', 'LongTermDebtCurrent', 'ShortTermDebtCurrent')),
    'operating_cash_flow': ('cash_flow_statement', ('NetCashProvidedByUsedInOperatingActivities',)),
    'capital_expenditures': ('cash_flow_statement', ('PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsForProceedsFromOtherPropertyPlantAndEquipment')),
    'depreciation_amortization': ('cash_flow_statement', ('DepreciationDepletionAndAmortization', 'DepreciationDepletionAndAmortizationPropertyPlantAndEquipment')),
    'stock_issuance': ('cash_flow_statement', ('ProceedsFromStockOptionsExercised', 'ProceedsFromIssuanceOfCommonStock')),
    'stock_repurchases': ('cash_flow_statement', ('PaymentsForRepurchaseOfCommonStock',)),
}

_META_COLUMNS = {'concept', 'label', 'level', 'abstract', 'unit', 'balance', 'weight', 'preferred_sign', 'standard_concept', 'point_in_time', 'dimension'}

def load_fundamentals(
    ticker: str,
    start_date: str | date,
    end_date: str | date,
    metrics: Sequence[Metric] | None = None,
    identity: str | None = None,
) -> pd.DataFrame:
    """Load selected quarterly fundamentals and their SEC filing dates for a ticker.

    Args:
        ticker (str): SEC ticker symbol to query, such as ``'AAPL'``.
        start_date (str | date): Inclusive fiscal-period start date in ``YYYY-MM-DD`` format.
        end_date (str | date): Inclusive fiscal-period end date in ``YYYY-MM-DD`` format.
        metrics (Sequence[Metric] | None): Metric names to return, or ``None`` for all metrics.
        identity (str | None): SEC identity in ``'Name email@example.com'`` form, or ``None`` to use an environment variable.

    Returns:
        pd.DataFrame: One row per fiscal quarter with ticker, period_end, filing_date, and requested metric columns.
    """
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError('start_date must be on or before end_date')
    selected = list(_METRICS) if metrics is None else list(dict.fromkeys(metrics))
    unknown = set(selected).difference(_METRICS)
    if unknown:
        raise ValueError(f'Unknown metrics: {', '.join(sorted(unknown))}')
    load_dotenv()
    sec_identity = identity or os.getenv('EDGAR_IDENTITY') or os.getenv('IDENTITY')
    if sec_identity:
        set_identity(sec_identity)

    filings = Company(ticker.upper()).get_filings(
        form=['10-Q', '10-K'], amendments=False,
        filing_date=(str((start - pd.Timedelta(days=380)).date()), str((end + pd.Timedelta(days=120)).date())),
        sort_by=[('filing_date', 'ascending')],
    )
    observations: list[dict[str, object]] = []
    for filing in filings:
        report = filing.obj()
        period_end = _resolve_period_end(report, filing, ticker.upper())
        row: dict[str, object] = {
            'ticker': ticker.upper(), 'period_end': period_end,
            'filing_date': pd.Timestamp(report.filing_date).normalize(),
            'form': str(getattr(filing, 'form', '')),
        }
        cache: dict[str, pd.DataFrame] = {}
        for metric in selected:
            statement_name, concepts = _METRICS[metric]
            if statement_name not in cache:
                statement = getattr(report, statement_name, None)
                cache[statement_name] = statement.to_dataframe(view='summary', standard=False, presentation=True) if statement is not None else pd.DataFrame()
            row[metric] = _extract_value(cache[statement_name], concepts, period_end)
        observations.append(row)

    result = pd.DataFrame(observations)
    columns = ['ticker', 'period_end', 'filing_date', *selected]
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values(['period_end', 'filing_date']).drop_duplicates('period_end', keep='last')
    result = _to_discrete_quarters(result, selected)
    result = result.loc[result['period_end'].between(start, end)]
    return result.reindex(columns=columns).sort_values('period_end').reset_index(drop=True)

def _resolve_period_end(report: object, filing: object, ticker: str) -> pd.Timestamp:
    """Resolve a filing's fiscal period end from report metadata, SEC XBRL facts, or statement columns.

    Args:
        report (object): Parsed edgartools filing report.
        filing (object): Source edgartools filing containing SEC metadata.
        ticker (str): Ticker used to identify failures in error messages.

    Returns:
        pd.Timestamp: Normalized fiscal period-end date for the filing.
    """
    for owner in (report, filing):
        try:
            source = getattr(owner, 'period_of_report', None)
        except Exception:
            source = None
        timestamp = _valid_timestamp(source)
        if timestamp is not None:
            return timestamp

    financials = getattr(report, 'financials', None)
    xbrl = getattr(financials, 'xb', None)
    facts_view = getattr(xbrl, 'facts_view', None)
    if facts_view is not None:
        try:
            facts = facts_view.get_facts_by_concept('DocumentPeriodEndDate', exact=False)
            for column in ('value', 'numeric_value', 'raw_value'):
                if column in facts:
                    for value in facts[column].dropna():
                        timestamp = _valid_timestamp(value)
                        if timestamp is not None:
                            return timestamp
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    statement_dates: list[pd.Timestamp] = []
    for statement_name in ('balance_sheet', 'income_statement', 'cash_flow_statement'):
        statement = getattr(report, statement_name, None)
        if statement is None:
            continue
        try:
            columns = statement.to_dataframe(view='summary', standard=False).columns
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        for column in columns:
            for match in re.findall(r'\d{4}-\d{2}-\d{2}', str(column)):
                timestamp = _valid_timestamp(match)
                if timestamp is not None:
                    statement_dates.append(timestamp)
    if statement_dates:
        return max(statement_dates)

    accession = getattr(filing, 'accession_no', 'unknown')
    form = getattr(filing, 'form', 'unknown')
    filing_date = getattr(filing, 'filing_date', 'unknown')
    raise ValueError(
        f'Could not resolve period_of_report for {ticker} {form} filing '
        f'{accession} filed {filing_date}'
    )

def _valid_timestamp(value: object) -> pd.Timestamp | None:
    """Convert a candidate date to a normalized timestamp or return ``None`` when invalid.

    Args:
        value (object): Candidate date-like value.

    Returns:
        pd.Timestamp | None: Normalized timestamp when valid, otherwise ``None``.
    """
    timestamp = pd.to_datetime(value, errors='coerce')
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).normalize()

def _to_discrete_quarters(frame: pd.DataFrame, metrics: Sequence[Metric]) -> pd.DataFrame:
    """Convert 10-Q year-to-date and 10-K annual flows into discrete-quarter values."""
    cash_metrics = [metric for metric in metrics if _METRICS[metric][0] == 'cash_flow_statement']
    income_metrics = [metric for metric in metrics if _METRICS[metric][0] == 'income_statement']
    prior_cash: dict[str, float] = {}
    income_since_annual: dict[str, list[float]] = {metric: [] for metric in income_metrics}
    for index in frame.index:
        annual = '10-K' in str(frame.at[index, 'form'])
        for metric in cash_metrics:
            raw = pd.to_numeric(frame.at[index, metric], errors='coerce')
            if pd.notna(raw):
                frame.at[index, metric] = raw - prior_cash.get(metric, 0.0)
                prior_cash[metric] = float(raw)
            if annual:
                prior_cash.pop(metric, None)
        for metric in income_metrics:
            raw = pd.to_numeric(frame.at[index, metric], errors='coerce')
            if annual and pd.notna(raw):
                frame.at[index, metric] = raw - sum(income_since_annual[metric])
                income_since_annual[metric].clear()
            elif pd.notna(raw):
                income_since_annual[metric].append(float(raw))
    return frame.drop(columns='form')

def _extract_value(frame: pd.DataFrame, concepts: Iterable[str], period_end: pd.Timestamp) -> float:
    """Extract the best matching concept value for a filing's report date."""
    if frame.empty or 'concept' not in frame:
        return float('nan')
    names = frame['concept'].astype(str).str.replace('us-gaap_', '', regex=False).str.split(':').str[-1]
    row = None
    for concept in concepts:
        matches = frame.loc[names.eq(concept)]
        if not matches.empty:
            row = matches.iloc[0]
            break
    if row is None:
        return float('nan')
    candidates = [column for column in frame.columns if str(column) not in _META_COLUMNS]
    dated = [column for column in candidates if period_end.strftime('%Y-%m-%d') in str(column)]
    candidates = dated or candidates
    quarter = [column for column in candidates if re.search(r'\bQ[1-4]\b', str(column), re.I)]
    for column in quarter or candidates:
        value = pd.to_numeric(row[column], errors='coerce')
        if pd.notna(value):
            return float(value)
    return float('nan')

def _metric_loader(metric: Metric, ticker: str, start_date: str | date, end_date: str | date, identity: str | None) -> pd.DataFrame:
    """Load one named metric through the shared fundamentals loader."""
    return load_fundamentals(ticker, start_date, end_date, metrics=[metric], identity=identity)

def _make_metric_function(metric: Metric):
    def retrieve(ticker: str, start_date: str | date, end_date: str | date, identity: str | None = None) -> pd.DataFrame:
        return _metric_loader(metric, ticker, start_date, end_date, identity)
    retrieve.__name__ = metric
    retrieve.__doc__ = f"""Load quarterly {metric.replace('_', ' ')} and SEC filing dates for a ticker.

    Args:
        ticker (str): SEC ticker symbol to query, such as ``'AAPL'``.
        start_date (str | date): Inclusive fiscal-period start date in ``YYYY-MM-DD`` format.
        end_date (str | date): Inclusive fiscal-period end date in ``YYYY-MM-DD`` format.
        identity (str | None): SEC identity, or ``None`` to use ``EDGAR_IDENTITY``/``IDENTITY``.

    Returns:
        pd.DataFrame: Quarterly ticker, period_end, filing_date, and {metric} columns.
    """
    return retrieve

for _metric in _METRICS:
    globals()[_metric] = _make_metric_function(_metric)

__all__ = ['Metric', 'load_fundamentals', *_METRICS]

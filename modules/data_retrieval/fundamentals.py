"""Retrieve quarterly fundamental data from SEC filings with edgartools."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from edgar import Company, set_identity

from modules.data_retrieval.fundamental_mappings import (
    COMPONENT_CONCEPT_PATTERNS,
    COMPONENT_LABEL_PATTERNS,
    COMPONENT_STANDARD_CONCEPTS,
    LABEL_EXCLUSIONS,
    LABEL_PATTERNS,
    META_COLUMNS,
    METRIC_DEPENDENCIES,
    METRICS,
    STANDARD_CONCEPTS,
    Metric,
)

def load_fundamentals(
    ticker: str,
    start_date: str | date,
    end_date: str | date,
    metrics: Sequence[Metric] | None = None,
    identity: str | None = None,
    history_quarters: int = 5,
) -> pd.DataFrame:
    """Load selected quarterly fundamentals and their SEC filing dates for a ticker.

    Args:
        ticker (str): SEC ticker symbol to query, such as ``'AAPL'``.
        start_date (str | date): Inclusive fiscal-period start date in ``YYYY-MM-DD`` format.
        end_date (str | date): Inclusive fiscal-period end date in ``YYYY-MM-DD`` format.
        metrics (Sequence[Metric] | None): Metric names to return, or ``None`` for all metrics.
        identity (str | None): SEC identity in ``'Name email@example.com'`` form, or ``None`` to use an environment variable.
        history_quarters (int): Accounting observations to retain at or before
            ``start_date``; five provides the latest available quarter plus four
            preceding observations.

    Returns:
        pd.DataFrame: Point-in-time quarterly observations with fiscal period end,
            filing date, UTC filing timestamp when available, and requested metrics.
    """
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError('start_date must be on or before end_date')
    if history_quarters < 1:
        raise ValueError('history_quarters must be at least 1')
    requested = list(METRICS) if metrics is None else list(dict.fromkeys(metrics))
    unknown = set(requested).difference(METRICS)
    if unknown:
        raise ValueError(f'Unknown metrics: {', '.join(sorted(unknown))}')
    selected = list(requested)
    for metric in requested:
        for dependency in METRIC_DEPENDENCIES.get(metric, ()):
            if dependency not in selected:
                selected.append(dependency)
    load_dotenv()
    sec_identity = identity or os.getenv('EDGAR_IDENTITY') or os.getenv('IDENTITY')
    if sec_identity:
        set_identity(sec_identity)

    company = Company(ticker.upper())
    filing_lookback = start - pd.DateOffset(
        months=3 * (history_quarters + 2)
    )
    filings = company.get_filings(
        form=['10-Q', '10-K'], amendments=False,
        filing_date=(str(filing_lookback.date()), str((end + pd.Timedelta(days=370)).date())),
        sort_by=[('filing_date', 'ascending')],
    )
    filing_records = list(filings)
    acceptance_by_date = {
        pd.Timestamp(filing.filing_date).normalize(): _filing_timestamp(filing)
        for filing in filing_records
    }
    observations: list[dict[str, object]] = []
    company_facts: pd.DataFrame | None = None
    for filing in filing_records:
        report = filing.obj()
        period_end = _resolve_period_end(report, filing, ticker.upper())
        filing_date = pd.Timestamp(report.filing_date).normalize()
        row: dict[str, object] = {
            'ticker': ticker.upper(), 'period_end': period_end,
            'filing_date': filing_date,
            'filing_timestamp': _filing_timestamp(filing),
            'form': str(getattr(filing, 'form', '')),
        }
        cache: dict[str, pd.DataFrame] = {}
        for metric in selected:
            statement_name, concepts = METRICS[metric]
            if statement_name not in cache:
                statement = getattr(report, statement_name, None)
                cache[statement_name] = statement.to_dataframe(view='summary', standard=True, presentation=True) if statement is not None else pd.DataFrame()
            value, source, matched_concept = _resolve_metric_value(
                metric, cache[statement_name], concepts, period_end
            )
            # A parsed statement can be non-empty but incomplete.  This occurs in
            # older filings whose presentation linkbase exposes only one section
            # of a statement (for example, assets without liabilities).  Fall
            # back at the metric level rather than only when the whole statement
            # failed to parse.
            if pd.isna(value):
                if company_facts is None:
                    company_facts = _load_company_facts(company)
                value, matched_concept, fact_filing_date = _resolve_company_fact(
                    metric=metric,
                    facts=company_facts,
                    concepts=concepts,
                    period_end=period_end,
                    filing_date=filing_date,
                    form=str(getattr(filing, 'form', '')),
                )
                if pd.notna(value):
                    source = 'company_facts'
                    if fact_filing_date is not None:
                        row['filing_date'] = max(pd.Timestamp(row['filing_date']), fact_filing_date)
                        row['filing_timestamp'] = acceptance_by_date.get(
                            pd.Timestamp(row['filing_date']).normalize(),
                            pd.NaT,
                        )
            value = _normalize_metric_sign(metric, value)
            row[metric] = value
            row[f'_{metric}_source'] = source
            row[f'_{metric}_concept'] = matched_concept
        _derive_missing_metrics(row)
        observations.append(row)

    result = pd.DataFrame(observations)
    columns = [
        'ticker', 'period_end', 'filing_date', 'filing_timestamp', *requested
    ]
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values(['period_end', 'filing_date']).drop_duplicates('period_end', keep='last')
    result = _to_discrete_quarters(result, selected)
    information_dates = _filing_information_dates(result)
    result = result.loc[
        result['period_end'].le(end) & information_dates.le(end)
    ]
    information_dates = information_dates.loc[result.index]
    history = (
        result.loc[information_dates.le(start)]
        .sort_values(['period_end', 'filing_date'])
        .tail(history_quarters)
    )
    analysis_filings = result.loc[information_dates.gt(start)]
    result = (
        pd.concat([history, analysis_filings], ignore_index=True)
        .drop_duplicates('period_end', keep='last')
    )
    return result.reindex(columns=columns).sort_values('period_end').reset_index(drop=True)

def _filing_information_dates(frame: pd.DataFrame) -> pd.Series:
    """Estimate when filings become usable before mapping them to trading dates.

    Args:
        frame (pd.DataFrame): Filing observations containing filing dates and UTC
            acceptance timestamps.

    Returns:
        pd.Series: Calendar dates adjusted forward one day for post-4 p.m. filings.
    """
    filing_dates = pd.to_datetime(frame['filing_date'], errors='coerce').dt.normalize()
    timestamps = pd.to_datetime(
        frame['filing_timestamp'], errors='coerce', utc=True
    ).dt.tz_convert('America/New_York')
    local_dates = timestamps.dt.tz_localize(None).dt.normalize()
    information_dates = filing_dates.where(timestamps.isna(), local_dates)
    after_close = timestamps.notna() & (
        timestamps.dt.hour.mul(60).add(timestamps.dt.minute) > 16 * 60
    )
    return information_dates + pd.to_timedelta(
        after_close.astype('int64'), unit='D'
    )

def _filing_timestamp(filing: object) -> pd.Timestamp | None:
    """Normalize a filing's exact SEC acceptance timestamp to UTC.

    Args:
        filing (object): Edgartools filing that may expose ``acceptance_datetime``.

    Returns:
        pd.Timestamp | None: Timezone-aware UTC acceptance timestamp when available.
    """
    raw_timestamp = getattr(filing, 'acceptance_datetime', None)
    timestamp = pd.to_datetime(raw_timestamp, errors='coerce', utc=True)
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)

def _normalize_metric_sign(metric: Metric, value: float) -> float:
    """Normalize statement-presentation signs while preserving cash-flow directions.

    Args:
        metric (Metric): Canonical metric associated with the value.
        value (float): Numeric value extracted from a financial statement.

    Returns:
        float: Sign-normalized value for the metric.
    """
    if metric == 'cost_of_goods_sold' and pd.notna(value):
        return abs(value)
    return value

def _derive_missing_metrics(row: dict[str, object]) -> None:
    """Fill missing statement totals from their standard accounting identities.

    Args:
        row (dict[str, object]): Filing observation containing extracted metric values.

    Returns:
        None.
    """
    if 'gross_profit' in row and pd.isna(row['gross_profit']):
        revenue = pd.to_numeric(row.get('revenue'), errors='coerce')
        cost = pd.to_numeric(row.get('cost_of_goods_sold'), errors='coerce')
        if pd.notna(revenue) and pd.notna(cost):
            row['gross_profit'] = float(revenue - cost)

    if 'total_liabilities' in row and pd.isna(row['total_liabilities']):
        assets = pd.to_numeric(row.get('total_assets'), errors='coerce')
        equity = pd.to_numeric(row.get('shareholders_equity'), errors='coerce')
        if pd.notna(assets) and pd.notna(equity):
            row['total_liabilities'] = float(assets - equity)

def _resolve_period_end(report: object, filing: object, ticker: str) -> pd.Timestamp:
    """Resolve a filing's fiscal period end from report metadata, SEC XBRL facts, or statement columns.

    Args:
        report (object): Parsed edgartools filing report.
        filing (object): Source edgartools filing containing SEC metadata.
        ticker (str): Ticker used to identify failures in error messages.

    Returns:
        pd.Timestamp: Normalized fiscal period-end date for the filing.
    """
    metadata_dates: list[pd.Timestamp] = []
    for owner in (report, filing):
        try:
            source = getattr(owner, 'period_of_report', None)
        except Exception:
            source = None
        timestamp = _valid_timestamp(source)
        if timestamp is not None:
            metadata_dates.append(timestamp)

    fact_dates: list[pd.Timestamp] = []
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
                            fact_dates.append(timestamp)
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
    if fact_dates:
        return fact_dates[0]
    if metadata_dates:
        return metadata_dates[0]

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
    cash_metrics = [metric for metric in metrics if METRICS[metric][0] == 'cash_flow_statement']
    income_metrics = [metric for metric in metrics if METRICS[metric][0] == 'income_statement']
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

def _load_company_facts(company: Company) -> pd.DataFrame:
    """Load SEC Company Facts once for filings whose parsed statements are unavailable.

    Args:
        company (Company): Edgartools company whose XBRL facts should be loaded.

    Returns:
        pd.DataFrame: Point-in-time company facts, or an empty frame when unavailable.
    """
    try:
        entity_facts = company.get_facts()
        return entity_facts.to_dataframe(pit_mode=True) if entity_facts is not None else pd.DataFrame()
    except (AttributeError, KeyError, TypeError, ValueError):
        return pd.DataFrame()

def _resolve_company_fact(
    metric: Metric,
    facts: pd.DataFrame,
    concepts: Iterable[str],
    period_end: pd.Timestamp,
    filing_date: pd.Timestamp,
    form: str,
) -> tuple[float, str | None, pd.Timestamp | None]:
    """Resolve an exact canonical metric from the matching SEC Company Facts filing.

    Args:
        metric (Metric): Canonical metric being resolved.
        facts (pd.DataFrame): SEC Company Facts observations for the company.
        concepts (Iterable[str]): Exact XBRL concept names accepted for the metric.
        period_end (pd.Timestamp): Fiscal period end of the failed parsed statement.
        filing_date (pd.Timestamp): Filing date used to prevent look-ahead matches.
        form (str): SEC form type, such as ``'10-Q'`` or ``'10-K'``.

    Returns:
        tuple[float, str | None, pd.Timestamp | None]: Numeric fact, matched concept,
            and its filing date, or ``NaN``, ``None``, and ``None``.
    """
    required = {'concept', 'period_end', 'filing_date', 'numeric_value'}
    if facts.empty or not required.issubset(facts.columns):
        return float('nan'), None, None
    eligible = facts.copy()
    eligible['_period_end'] = pd.to_datetime(eligible['period_end'], errors='coerce').dt.normalize()
    eligible['_filing_date'] = pd.to_datetime(eligible['filing_date'], errors='coerce').dt.normalize()
    eligible = eligible.loc[
        eligible['_period_end'].eq(period_end)
        & eligible['_filing_date'].ge(filing_date)
    ]
    form_column = 'form_type' if 'form_type' in eligible else 'form'
    if form_column in eligible:
        eligible = eligible.loc[eligible[form_column].astype(str).str.startswith(form)]
    if eligible.empty:
        return float('nan'), None, None

    names = eligible['concept'].astype(str).str.replace('us-gaap_', '', regex=False).str.split(':').str[-1]
    for concept in concepts:
        matches = eligible.loc[names.eq(concept)].copy()
        matches['numeric_value'] = pd.to_numeric(matches['numeric_value'], errors='coerce')
        matches = matches.loc[matches['numeric_value'].notna()]
        if matches.empty:
            continue
        statement_name = METRICS[metric][0]
        if statement_name != 'balance_sheet' and 'period_start' in matches:
            starts = pd.to_datetime(matches['period_start'], errors='coerce').dt.normalize()
            matches['_duration'] = (period_end - starts).dt.days
            matches = matches.loc[matches['_duration'].ge(0)]
            if matches.empty:
                continue
            wants_annual_or_ytd = form == '10-K' or statement_name == 'cash_flow_statement'
            duration = matches['_duration'].max() if wants_annual_or_ytd else matches['_duration'].min()
            matches = matches.loc[matches['_duration'].eq(duration)]
        earliest_filing = matches['_filing_date'].min()
        matches = matches.loc[matches['_filing_date'].eq(earliest_filing)]
        return (
            float(matches.iloc[-1]['numeric_value']),
            str(matches.iloc[-1]['concept']),
            pd.Timestamp(earliest_filing),
        )
    return float('nan'), None, None

def _resolve_metric_value(
    metric: Metric,
    frame: pd.DataFrame,
    concepts: Iterable[str],
    period_end: pd.Timestamp,
) -> tuple[float, str, str | None]:
    """Resolve a metric through exact, standardized, and constrained-label matches.

    Args:
        metric (Metric): Canonical metric being resolved.
        frame (pd.DataFrame): Financial statement rows returned by edgartools.
        concepts (Iterable[str]): Exact XBRL concept names accepted for the metric.
        period_end (pd.Timestamp): Filing period end used to choose the value column.

    Returns:
        tuple[float, str, str | None]: Value, resolution source, and matched concept name.
    """
    if frame.empty or 'concept' not in frame:
        return float('nan'), 'missing', None
    prefer_ytd = METRICS[metric][0] == 'cash_flow_statement'
    eligible = frame
    if 'abstract' in eligible:
        eligible = eligible.loc[~eligible['abstract'].fillna(False).astype(bool)]
    names = frame['concept'].astype(str).str.replace('us-gaap_', '', regex=False).str.split(':').str[-1]
    for concept in concepts:
        matches = eligible.loc[names.eq(concept)]
        if not matches.empty:
            value = _extract_period_value(
                matches, frame, period_end, prefer_ytd=prefer_ytd
            )
            if pd.notna(value):
                return value, 'reported', str(matches.iloc[0]['concept'])

    component_values: dict[str, float] = {}
    if 'standard_concept' in eligible:
        standardized = eligible['standard_concept'].fillna('').map(str).str.split(':').str[-1]
        for concept in STANDARD_CONCEPTS.get(metric, ()):
            matches = eligible.loc[standardized.eq(concept)]
            if not matches.empty:
                value = _extract_period_value(
                    matches, frame, period_end, prefer_ytd=prefer_ytd
                )
                if pd.notna(value):
                    return value, 'standardized', str(matches.iloc[0]['concept'])

        for concept in COMPONENT_STANDARD_CONCEPTS.get(metric, ()):
            matches = eligible.loc[standardized.eq(concept)]
            if not matches.empty:
                value = _extract_period_value(
                    matches, frame, period_end, prefer_ytd=prefer_ytd
                )
                if pd.notna(value):
                    component_values[str(matches.iloc[0]['concept'])] = value

    labels = eligible.get('label', pd.Series('', index=eligible.index)).astype(str).str.strip()
    excluded = LABEL_EXCLUSIONS.get(metric, ())
    for pattern in LABEL_PATTERNS.get(metric, ()):
        mask = labels.str.match(pattern, case=False, na=False)
        for term in excluded:
            mask &= ~labels.str.contains(re.escape(term), case=False, na=False)
        matches = eligible.loc[mask]
        if not matches.empty:
            value = _extract_period_value(
                matches, frame, period_end, prefer_ytd=prefer_ytd
            )
            if pd.notna(value):
                return value, 'regex', str(matches.iloc[0]['concept'])

    for pattern in COMPONENT_LABEL_PATTERNS.get(metric, ()):
        matches = eligible.loc[labels.str.match(pattern, case=False, na=False)]
        if not matches.empty:
            value = _extract_period_value(
                matches, frame, period_end, prefer_ytd=prefer_ytd
            )
            if pd.notna(value):
                component_values[str(matches.iloc[0]['concept'])] = value
    raw_concepts = eligible['concept'].fillna('').map(str)
    for pattern in COMPONENT_CONCEPT_PATTERNS.get(metric, ()):
        matches = eligible.loc[raw_concepts.str.match(pattern, case=False, na=False)]
        if not matches.empty:
            value = _extract_period_value(
                matches, frame, period_end, prefer_ytd=prefer_ytd
            )
            if pd.notna(value):
                component_values[str(matches.iloc[0]['concept'])] = value
    if component_values:
        return float(sum(component_values.values())), 'components', '+'.join(component_values)
    return float('nan'), 'missing', None

def _extract_period_value(
    rows: pd.DataFrame,
    frame: pd.DataFrame,
    period_end: pd.Timestamp,
    prefer_ytd: bool = False,
) -> float:
    """Extract the best period value from one or more matched statement rows.

    Args:
        rows (pd.DataFrame): Candidate statement rows for one metric.
        frame (pd.DataFrame): Full statement used to identify value columns.
        period_end (pd.Timestamp): Filing period end used to rank value columns.
        prefer_ytd (bool): Whether year-to-date columns should precede discrete-quarter columns.

    Returns:
        float: Best numeric period value, or ``NaN`` when none exists.
    """
    candidates = [column for column in frame.columns if str(column) not in META_COLUMNS]
    dated = [column for column in candidates if period_end.strftime('%Y-%m-%d') in str(column)]
    candidates = dated or candidates
    quarter = [column for column in candidates if re.search(r'\bQ[1-4]\b', str(column), re.I)]
    year_to_date = [
        column for column in candidates
        if re.search(r'\bYTD\b', str(column), re.I)
    ]
    preferred = year_to_date if prefer_ytd else quarter
    ordered = preferred + [column for column in candidates if column not in preferred]
    for _, row in rows.iterrows():
        for column in ordered:
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

for _metric in METRICS:
    globals()[_metric] = _make_metric_function(_metric)

__all__ = ['Metric', 'load_fundamentals', *METRICS]

"""
data_computation.py

Purpose:
--------
Computation layer of the analytics module.

The computation module is responsible for taking ingested/cleaned data
and calculating analytical results.

It answers business questions by:
- aggregating data
- calculating metrics
- finding trends
- measuring relationships
- ranking entities
- generating statistical outputs

Inputs:
- Raw or processed data tables
- Columns, Series, or DataFrames

Outputs:
- Calculated metrics
- Aggregated DataFrames
- Statistical results
- Values ready for visualization

The computation layer DOES NOT create charts or visual outputs.

Rules:
------
- No charts.
- No visualization logic.
- No dataset-specific assumptions.
- Accept pandas Series, arrays, or DataFrames.
- Return clean analytical outputs.

Question Coverage:
------------------
Q1   total, count, unique_count, average, calculate_ratio
Q2   aggregate_time_period, growth_rate, rolling_average, trend_slope
Q3   aggregate_time_period, percentage_share, rank_values
Q4   aggregate_metric, concentration_analysis, pareto_table
Q5   aggregate_metric, concentration_analysis, pareto_table, rank_values
Q6   aggregate_metric, concentration_analysis, pareto_table, rank_values
Q7   aggregate_metric, average, median, rank_values, calculate_ratio
Q8   average, median, standard_deviation, summary_statistics
Q9   aggregate_multiple_metrics
Q10  aggregate_metric, rank_values, percentage_share
Q11  aggregate_multiple_metrics, percentage_share, rank_values
Q12  aggregate_multiple_metrics, calculate_ratio
Q13  aggregate_multiple_metrics, rank_values
Q14  aggregate_multiple_metrics, rank_values, percentage_share
Q15  aggregate_multiple_metrics, calculate_ratio
Q16  aggregate_multiple_metrics, rank_values
Q17  aggregate_multiple_metrics
Q18  aggregate_multiple_metrics
Q19  calculate_ratio, aggregate_multiple_metrics
Q20  delivery_duration, average, median, aggregate_time_period, trend_slope
Q21  delivery_duration, delivery_delay, aggregate_multiple_metrics, rank_values
Q22  delivery_duration, delivery_delay, aggregate_multiple_metrics, rank_values
Q23  delivery_delay, delivery_status_rates
Q24  correlation
Q25  aggregate_multiple_metrics, rank_values
Q26  aggregate_multiple_metrics, rank_values
Q27  aggregate_time_period, rolling_average, trend_slope
Q28  aggregate_multiple_metrics, percentage_share
Q29  aggregate_multiple_metrics, summary_statistics
Q30  aggregate_multiple_metrics
Q31  basket_size, correlation
Q32  calculate_ratio, detect_outliers
Q33  total, calculate_ratio, percentage_share
Q34  aggregate_multiple_metrics, weighted_score
Q35  aggregate_multiple_metrics, weighted_score
Q36  aggregate_multiple_metrics, calculate_ratio, weighted_score
"""

from typing import Any, cast, Dict, Optional, Union, List

import numpy as np
import pandas as pd
from scipy import stats


Numeric = Union[pd.Series, np.ndarray, List[Any]]


# ============================================================
# BASIC METRICS
# ============================================================

def total(values: Numeric) -> float:
    """Calculate total value."""
    return pd.Series(values).sum()


def count(values: Numeric) -> int:
    """Count non-null records."""
    return pd.Series(values).count()


def unique_count(values: Numeric) -> int:
    """Count unique entities."""
    return pd.Series(values).nunique()


def average(values: Numeric) -> float:
    """Calculate mean value."""
    return pd.Series(values).mean()


def median(values: Numeric) -> float:
    """Calculate median value."""
    return pd.Series(values).median()


def standard_deviation(values: Numeric) -> float:
    """Measure value variation."""
    return pd.Series(values).std()


def summary_statistics(values: Numeric) -> Dict[str, Any]:
    """
    Generate common descriptive statistics.

    Useful for:
    - spending distribution
    - delivery variation
    - ratings
    """

    s = pd.Series(values).dropna()

    return {
        "count": s.count(),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
        "25%": s.quantile(.25),
        "75%": s.quantile(.75)
    }


# ============================================================
# RATIOS AND CONTRIBUTIONS
# ============================================================

def calculate_ratio(
    numerator: Numeric,
    denominator: Numeric
) -> pd.Series:
    """
    Calculate ratio between two metrics.

    Examples:
    - revenue per customer
    - freight / revenue
    - orders per customer
    """

    return (
        pd.Series(numerator)
        /
        pd.Series(denominator).replace(0, np.nan)
    )


def percentage_share(values: Numeric) -> pd.Series:
    """
    Calculate percentage contribution.

    Examples:
    - seller revenue share
    - product revenue share
    - category share
    """

    s = pd.Series(values)

    return s / s.sum() * 100


# ============================================================
# RANKING
# ============================================================

def rank_values(
    values: Numeric,
    ascending: bool = False
) -> pd.Series:
    """
    Rank values.

    ascending=False:
        Highest value receives rank 1

    ascending=True:
        Lowest value receives rank 1
    """

    return (
        pd.Series(values)
        .rank(
            method="dense",
            ascending=ascending
        )
    )


# ============================================================
# GROUP ANALYSIS
# ============================================================

def aggregate_metric(
    df: pd.DataFrame,
    group_column: str,
    metric_column: str,
    aggregation: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate a metric by category.

    Examples:
    - revenue by seller
    - orders by customer
    - ratings by category
    """

    return (
        df.groupby(group_column)[metric_column]
        .agg(aggregation)
        .reset_index()
    )


def aggregate_multiple_metrics(
    df: pd.DataFrame,
    group_column: str,
    metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Aggregate multiple metrics together.

    Example:
    Seller:
        revenue=sum
        orders=count
        rating=mean
    """

    return (
        df.groupby(group_column)
        .agg(metrics)
        .reset_index()
    )


def basket_size(order_ids: Numeric) -> pd.Series:
    """
    Count items per order.

    Input: order_id column from the order items table
           (one row per item purchased)

    Returns a Series indexed by order_id.

    Useful for:
    - basket size vs revenue correlation (Q31)
    """

    s = pd.Series(order_ids)

    return (
        s.value_counts()
        .rename("basket_size")
    )


# ============================================================
# TIME SERIES ANALYSIS
# ============================================================

def aggregate_time_period(
    dates: Numeric,
    values: Numeric,
    frequency: str = "ME",
    aggregation: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate values over time.

    Frequency examples:
    ME = month
    QE = quarter
    YE = year
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "value": values
        }
    )

    return (
        df
        .set_index("date")
        .resample(frequency)
        ["value"]
        .agg(aggregation)
        .reset_index()
    )


def growth_rate(values: Numeric) -> pd.Series:
    """
    Calculate percentage change between periods.
    """

    return (
        pd.Series(values)
        .pct_change()
        * 100
    )


def rolling_average(
    values: Numeric,
    window: int = 3
) -> pd.Series:
    """
    Smooth time-series fluctuations.
    """

    return (
        pd.Series(values)
        .rolling(window)
        .mean()
    )


def trend_slope(values: Numeric) -> float:
    """
    Calculate overall upward/downward trend.
    """

    y = pd.Series(values).dropna()

    if len(y) < 2:
        return np.nan

    slope, *_ = stats.linregress(
        range(len(y)),
        y
    )

    return cast(float, slope)


# ============================================================
# CONCENTRATION AND DISTRIBUTION
# ============================================================

def cumulative_percentage(values: Numeric) -> pd.Series:
    """
    Calculate cumulative contribution.

    Used for:
    - Pareto analysis
    - concentration risk
    """

    s = (
        pd.Series(values)
        .sort_values(
            ascending=False
        )
    )

    return (
        s.cumsum()
        /
        s.sum()
        *
        100
    )


def concentration_analysis(values: Numeric) -> Dict[str, Any]:
    """
    Measure how concentrated a metric is across entities.

    Returns the share held by the top 1%, 5%, and 10% of entities,
    and the percentage of entities responsible for 80% of the total.

    Useful for:
    - customer concentration risk (Q4)
    - seller concentration risk (Q5)
    - product concentration risk (Q6)
    """

    s = (
        pd.Series(values)
        .dropna()
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )

    total_val = s.sum()
    n = len(s)
    result: Dict[str, Any] = {}

    for pct in [0.01, 0.05, 0.10]:
        top_n = max(1, int(np.ceil(n * pct)))
        share = s.head(top_n).sum() / total_val * 100
        result[f"top_{int(pct * 100)}_pct_share"] = round(share, 2)

    cumsum = s.cumsum()
    pareto_n = int((cumsum >= total_val * 0.80).idxmax()) + 1
    result["entities_for_80pct_of_value"] = round(pareto_n / n * 100, 2)

    return result


def pareto_table(
    df: pd.DataFrame,
    group_column: str,
    value_column: str
) -> pd.DataFrame:
    """
    Build a sorted table with a cumulative percentage column.

    Output is ready to pass directly into pareto_chart().

    Useful for:
    - customer Pareto (Q4)
    - seller Pareto (Q5)
    - product Pareto (Q6)
    """

    result = (
        df
        .groupby(group_column)[value_column]
        .sum()
        .reset_index()
        .sort_values(value_column, ascending=False)
        .reset_index(drop=True)
    )

    result["cumulative_pct"] = cumulative_percentage(
        result[value_column]
    )

    return result


def correlation(
    x: Numeric,
    y: Numeric
) -> Dict[str, Any]:
    """
    Calculate relationship between variables.
    """

    df = pd.DataFrame(
        {
            "x": x,
            "y": y
        }
    ).dropna()

    return {
        "pearson": stats.pearsonr(df.x, df.y)[0],
        "spearman": stats.spearmanr(df.x, df.y)[0],
    }


def detect_outliers(
    values: Numeric
) -> pd.Series:
    """
    Detect extreme values using IQR.
    """

    s = pd.Series(values)

    q1 = s.quantile(.25)
    q3 = s.quantile(.75)

    iqr = q3 - q1

    return (
        (s < q1 - 1.5 * iqr)
        |
        (s > q3 + 1.5 * iqr)
    )


# ============================================================
# DECISION SCORING
# ============================================================

def weighted_score(
    metrics: List[Numeric],
    weights: Optional[List[float]] = None
) -> pd.Series:
    """
    Combine normalized metrics into one score.

    Each metric is min-max normalized then multiplied by its weight.
    Higher score = better overall performance.

    Useful for:
    - category investment ranking (Q34)
    - strategic seller selection (Q35)
    - market opportunity scoring (Q36)
    """

    frame = pd.concat(
        [pd.Series(m) for m in metrics],
        axis=1
    )

    if weights is None:
        weights = [
            1 / len(metrics)
            for _ in metrics
        ]

    normalized = (
        (frame - frame.min())
        / (frame.max() - frame.min())
    )

    return (
        normalized
        .multiply(weights, axis=1)
        .sum(axis=1)
    )


# ============================================================
# DELIVERY ANALYSIS
# ============================================================

def delivery_duration(
    purchase_dates: Numeric,
    delivered_dates: Numeric
) -> pd.Series:
    """
    Calculate days from purchase to delivery.

    Useful for:
    - average delivery time (Q20)
    - seller delivery performance (Q21)
    - state delivery performance (Q22)
    """

    return (
        pd.Series(pd.to_datetime(delivered_dates))
        - pd.Series(pd.to_datetime(purchase_dates))
    ).dt.days


def delivery_delay(
    actual_dates: Numeric,
    estimated_dates: Numeric
) -> pd.Series:
    """
    Calculate days between actual and promised delivery.

    Negative = delivered early
    Zero     = delivered on time
    Positive = delivered late

    Useful for:
    - late delivery identification (Q21, Q22)
    - service level measurement (Q23)
    """

    return (
        pd.Series(pd.to_datetime(actual_dates))
        - pd.Series(pd.to_datetime(estimated_dates))
    ).dt.days


def delivery_status_rates(delay_days: Numeric) -> Dict[str, Any]:
    """
    Calculate early, on-time, and late delivery rates.

    Input: result of delivery_delay()

    Useful for:
    - service-level failure measurement (Q23)
    """

    s = pd.Series(delay_days).dropna()
    n = len(s)

    return {
        "late_rate_pct":    round(float((s > 0).sum() / n * 100), 2),
        "on_time_rate_pct": round(float((s == 0).sum() / n * 100), 2),
        "early_rate_pct":   round(float((s < 0).sum() / n * 100), 2),
        "average_delay_days": round(float(s.mean()), 2)
    }
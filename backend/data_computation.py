"""
data_computation.py
===================
Pure-data computation building blocks for retail star-schema analytics.

Design principle
----------------
Every arithmetic function receives the actual data it needs — pandas Series,
numpy arrays, or scalars — not a DataFrame plus column-name strings.
Feed the columns in directly, like variables in a formula:

    ratio  = compute_ratio(df['freight_value'], df['order_value'])
    slope  = compute_trend_slope(monthly['revenue'])
    corr   = compute_correlation(df['delivery_days'], df['review_score'])

Grouping helpers accept a value Series and a group Series so the caller
decides what data flows in.  Multi-metric helpers that need several columns
together still accept a DataFrame but keep the API surface small.

Return types: pd.Series, scalar, or dict — ready to chain or pass to
data_visualization.py.

Dependencies: pandas, numpy, scipy
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# convenience alias used throughout
Numeric = Union[pd.Series, np.ndarray, list]


# ── 1. Scalar aggregation ─────────────────────────────────────────────────────
#
# Feed any numeric Series (or array) directly — no DataFrame, no column names.
# These are the terminal "answer" values for KPI cards and summary metrics.

def total(values: Numeric) -> float:
    """Sum of all values.

    Q1  total revenue, total freight cost
    Q33 total freight spend
    """
    return float(pd.Series(values).sum())


def count_unique(values: Numeric) -> int:
    """Number of distinct values.

    Q1  total customers (unique customer_id), total sellers (unique seller_id)
    """
    return int(pd.Series(values).nunique())


def count_records(values: Numeric) -> int:
    """Count of non-null entries.

    Q1  total orders
    """
    return int(pd.Series(values).count())


def average(values: Numeric) -> float:
    """Arithmetic mean.

    Q1  average order value
    Q20 average delivery time
    Q26 average seller rating
    """
    return round(float(pd.Series(values).mean()), 4)


def median(values: Numeric) -> float:
    """Median value.

    Q8  median order value
    Q20 median delivery time
    """
    return round(float(pd.Series(values).median()), 4)


def std_dev(values: Numeric) -> float:
    """Standard deviation.

    Q8  spending variability across customers
    Q20 delivery time variance
    """
    return round(float(pd.Series(values).std()), 4)


# ── 2. Ratio & share ──────────────────────────────────────────────────────────
#
# Pass the numerator Series and the denominator Series — the function does
# the division and handles zeros safely.

def compute_ratio(numerator: Numeric, denominator: Numeric) -> pd.Series:
    """Element-wise ratio; zero denominators become NaN.

    Q1  revenue_per_customer  = compute_ratio(df['revenue'],       df['customer_count'])
    Q19 freight_ratio         = compute_ratio(df['freight_value'], df['revenue'])
    Q33 freight_share         = compute_ratio(df['freight_value'], df['order_value'])
    Q32 freight_to_value flag = compute_ratio(df['freight'],       df['price'])
    """
    num = pd.Series(numerator).reset_index(drop=True)
    den = pd.Series(denominator).replace(0, np.nan).reset_index(drop=True)
    return num / den


def compute_share(values: Numeric) -> pd.Series:
    """Each value expressed as a percentage of the column total (sums to 100 %).

    Q3  monthly revenue contribution %
    Q5  seller revenue share %
    Q6  product revenue share %
    Q11 category revenue share %
    Q28 payment-type revenue share %
    """
    s = pd.Series(values)
    return (s / s.sum() * 100).round(2)


# ── 3. Ranking ────────────────────────────────────────────────────────────────

def rank_series(values: Numeric, ascending: bool = False) -> pd.Series:
    """Integer rank aligned to values (1 = best).

    ascending=False → rank 1 has the highest value  (revenue, rating, CLV).
    ascending=True  → rank 1 has the lowest value   (delivery time, late rate).

    Q7  CLV rank        : rank_series(df['lifetime_value'])
    Q10 product revenue : rank_series(df['product_revenue'])
    Q21 seller delivery : rank_series(df['avg_delivery_days'], ascending=True)
    Q22 state delivery  : rank_series(df['avg_delivery_days'], ascending=True)
    """
    return pd.Series(values).rank(method="min", ascending=ascending).astype(int)


# ── 4. Pareto & concentration ─────────────────────────────────────────────────

def cumulative_pct(values: Numeric) -> pd.Series:
    """Cumulative percentage along a Series that is already sorted descending.

    Sort values largest-first, then call this to get the Pareto curve.

    Q4  customer revenue Pareto
    Q5  seller revenue Pareto
    Q6  product revenue Pareto
    """
    s = pd.Series(values)
    return (s.cumsum() / s.sum() * 100).round(2)


def concentration_stats(
    values: Numeric,
    percentiles: Optional[List[float]] = None,
) -> Dict:
    """Top-N % revenue share and Pareto ratio for a per-entity metric.

    Pass one value per entity (already aggregated — one row per customer,
    seller, or product).

    Parameters
    ----------
    values      : per-entity aggregated metric Series.
    percentiles : fraction cut-offs. Defaults to [0.01, 0.05, 0.10].

    Returns
    -------
    dict with total_entities, total_value, top_Xpct_share keys, and
    pareto_ratio_pct_entities_for_80pct_value.

    Q4  customer revenue concentration
    Q5  seller revenue concentration
    Q6  product revenue concentration
    """
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.10]

    s = pd.Series(values).dropna().sort_values(ascending=False).reset_index(drop=True)
    total = s.sum()
    n = len(s)

    result: Dict = {"total_entities": n, "total_value": round(float(total), 2)}
    for p in percentiles:
        top_n = max(1, int(np.ceil(n * p)))
        share = s.head(top_n).sum() / total * 100
        result[f"top_{int(p * 100)}pct_share"] = round(float(share), 2)

    cumsum = s.cumsum()
    pareto_idx = int((cumsum >= total * 0.80).idxmax()) + 1
    result["pareto_ratio_pct_entities_for_80pct_value"] = round(pareto_idx / n * 100, 2)
    return result


# ── 5. Grouping ───────────────────────────────────────────────────────────────
#
# Pass the value Series and the group Series — the function groups and
# aggregates without needing the full DataFrame or column-name strings.

def group_aggregate(
    values: Numeric,
    groups: Numeric,
    func: str = "sum",
) -> pd.Series:
    """Group values by groups and apply an aggregation.

    Returns a named Series indexed by the unique group labels.

    Parameters
    ----------
    values : numeric column (e.g. df['revenue']).
    groups : category column that defines the grouping (e.g. df['seller_id']).
    func   : 'sum', 'mean', 'median', 'count', 'nunique', 'std', 'min', 'max'.

    Q14 seller revenue  : group_aggregate(df['revenue'],        df['seller_id'])
    Q11 category revenue: group_aggregate(df['revenue'],        df['category'])
    Q17 state revenue   : group_aggregate(df['revenue'],        df['state'])
    Q28 payment revenue : group_aggregate(df['payment_value'],  df['payment_type'])
    Q26 seller rating   : group_aggregate(df['review_score'],   df['seller_id'], 'mean')
    Q21 seller delivery : group_aggregate(df['delivery_days'],  df['seller_id'], 'mean')
    """
    return pd.Series(values).groupby(pd.Series(groups)).agg(func)


def multi_group_aggregate(
    df: pd.DataFrame,
    groups: Union[str, List[str]],
    agg_map: Dict[str, Union[str, List[str]]],
) -> pd.DataFrame:
    """Multiple aggregations in one groupby pass.

    Use when you need several metrics for the same grouping key to avoid
    repeated groupby calls.

    Parameters
    ----------
    df      : source DataFrame containing all needed columns.
    groups  : column name(s) to group by (str or list of str).
    agg_map : {column_name: agg_func_or_list}
              e.g. {'revenue': 'sum', 'order_id': 'count', 'rating': 'mean'}

    Returns pd.DataFrame with flattened column names.

    Q10 product metrics : multi_group_aggregate(df, 'product_id',
                              {'revenue': 'sum', 'order_id': 'count', 'price': 'mean'})
    Q35 seller metrics  : multi_group_aggregate(df, 'seller_id',
                              {'revenue': 'sum', 'review_score': 'mean',
                               'delivery_days': 'mean', 'order_id': 'count'})
    """
    result = df.groupby(groups, as_index=False).agg(agg_map)
    # Flatten MultiIndex columns produced when agg_map values are lists
    result.columns = [
        "_".join(str(p) for p in col if p).strip("_")
        if isinstance(col, tuple) else col
        for col in result.columns
    ]
    return result


# ── 6. Time-series transformations ───────────────────────────────────────────
#
# All functions accept already-extracted Series. For resampling, pass the
# timestamp Series and the value Series together.

def resample_to_period(
    timestamps: Numeric,
    values: Numeric,
    freq: str = "ME",
    func: str = "sum",
) -> pd.DataFrame:
    """Aggregate a value Series to a fixed time frequency.

    Parameters
    ----------
    timestamps : datetime Series (e.g. df['purchase_timestamp']).
    values     : numeric Series to aggregate (e.g. df['payment_value']).
    freq       : 'ME' month-end (default), 'QE' quarter, 'YE' year, 'W' week.
    func       : 'sum', 'mean', 'count', 'median'.

    Returns pd.DataFrame with columns ['period', <value_name_or_'value'>].

    Q2  monthly revenue : resample_to_period(df['purchase_date'], df['revenue'])
    Q27 monthly rating  : resample_to_period(df['review_date'], df['score'], func='mean')
    """
    vs = pd.Series(values)
    col_name = vs.name if vs.name else "value"
    combined = pd.DataFrame({
        "period": pd.to_datetime(pd.Series(timestamps)),
        "value":  vs.values,
    }).dropna(subset=["period"])
    result = combined.resample(freq, on="period")["value"].agg(func).reset_index()
    result.rename(columns={"value": col_name}, inplace=True)
    return result


def compute_growth_rate(
    values: Numeric,
    as_pct: bool = True,
) -> pd.Series:
    """Period-over-period change of a time-ordered Series.

    values must be in chronological order (call resample_to_period first).

    as_pct=True  → percentage change (Q2 MoM revenue growth %)
    as_pct=False → absolute change   (Q2 MoM absolute revenue change)
    """
    s = pd.Series(values)
    return (s.pct_change() * 100).round(2) if as_pct else s.diff().round(4)


def compute_rolling_average(values: Numeric, window: int = 3) -> pd.Series:
    """Rolling mean over window periods, aligned to the same index as values.

    Q2  3-month rolling revenue average
    Q27 rolling satisfaction score trend
    """
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().round(4)


def compute_trend_slope(values: Numeric) -> float:
    """OLS slope of a time-ordered numeric Series.

    Positive → upward trend.  Negative → downward trend.
    Returns NaN when fewer than 2 non-null data points exist.

    Q2  is revenue trending up or down?
    Q20 is delivery time improving over time?
    Q27 is satisfaction score rising or falling?
    """
    s = pd.Series(values).dropna()
    if len(s) < 2:
        return float("nan")
    slope, *_ = stats.linregress(np.arange(len(s)), s.to_numpy())
    return round(float(slope), 6)


# ── 7. Delivery ───────────────────────────────────────────────────────────────
#
# Pass the date columns directly; functions return Series aligned to the
# original DataFrame index.

def delivery_delay_days(actual: Numeric, estimated: Numeric) -> pd.Series:
    """Days between actual delivery and the promised delivery date.

    Negative → delivered early.
    Zero     → delivered exactly on time.
    Positive → delivered late.

    Q20 average delivery delay
    Q23 late-delivery identification
    """
    return (pd.to_datetime(actual) - pd.to_datetime(estimated)).dt.days


def delivery_duration_days(purchase: Numeric, delivered: Numeric) -> pd.Series:
    """Calendar days from purchase timestamp to actual delivery.

    Q20 average end-to-end delivery time
    Q21 seller delivery performance
    Q22 state delivery performance
    """
    return (pd.to_datetime(delivered) - pd.to_datetime(purchase)).dt.days


def classify_delivery(delay_days: Numeric) -> pd.Series:
    """Label each row as 'early', 'on_time', or 'late' from delay_days.

    Feed the output of delivery_delay_days() directly.

    Q23 on-time delivery rate
    Q24 satisfaction vs delivery status
    """
    s = pd.Series(delay_days)
    return pd.cut(s, bins=[-np.inf, -1, 0, np.inf], labels=["early", "on_time", "late"])


def delivery_rate_stats(delivery_status: Numeric) -> Dict:
    """On-time / late / early rates (%) from a classify_delivery() result.

    Q23 late-delivery rate, early rate, on-time rate
    """
    s = pd.Series(delivery_status)
    counts = s.value_counts()
    n = len(s)
    return {
        status: round(float(counts.get(status, 0) / n * 100), 2)
        for status in ("early", "on_time", "late")
    }


# ── 8. Statistical summary ────────────────────────────────────────────────────

def summary_stats(values: Numeric) -> Dict:
    """Descriptive statistics for any numeric Series.

    Returns mean, median, std, min, max, count, q25, q75.

    Q8  order value distribution
    Q20 delivery time spread
    Q24 review score distribution
    Q29 payment value by installment type
    """
    s = pd.Series(values).dropna()
    return {
        "mean":   round(float(s.mean()),           4),
        "median": round(float(s.median()),         4),
        "std":    round(float(s.std()),            4),
        "min":    round(float(s.min()),            4),
        "max":    round(float(s.max()),            4),
        "count":  int(s.count()),
        "q25":    round(float(s.quantile(0.25)),   4),
        "q75":    round(float(s.quantile(0.75)),   4),
    }


def flag_outliers(values: Numeric, method: str = "iqr") -> pd.Series:
    """Boolean Series — True marks statistical outliers.

    method='iqr'    → outside 1.5 × IQR fence  (Q32 high-freight orders)
    method='zscore' → |z| > 3                  (any anomaly detection)
    """
    s = pd.Series(values)
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    if method == "zscore":
        z = np.abs(stats.zscore(s.fillna(float(s.mean()))))
        return pd.Series(z > 3, index=s.index)
    raise ValueError(f"method must be 'iqr' or 'zscore'. Got: {method!r}")


def compute_correlation(x: Numeric, y: Numeric) -> Dict:
    """Pearson and Spearman correlations between two numeric Series.

    Returns pearson_r, pearson_p, spearman_r, spearman_p, n.

    Q24 delivery_days ↔ review_score
    Q31 basket_size   ↔ order_revenue
    """
    pair = pd.DataFrame({"x": pd.Series(x), "y": pd.Series(y)}).dropna()
    p_r, p_p = stats.pearsonr(pair["x"], pair["y"])
    s_r, s_p = stats.spearmanr(pair["x"], pair["y"])
    return {
        "pearson_r":  round(float(p_r), 4),
        "pearson_p":  round(float(p_p), 4),
        "spearman_r": round(float(s_r), 4),
        "spearman_p": round(float(s_p), 4),
        "n":          int(len(pair)),
    }


# ── 9. Composite scoring ──────────────────────────────────────────────────────

def composite_score(
    metric_series: List[Numeric],
    weights: Optional[List[float]] = None,
    ascending_flags: Optional[List[bool]] = None,
) -> pd.Series:
    """Min-max normalize multiple metrics, weight them, and sum to a [0, 1] score.

    Higher composite score always means "better" regardless of metric direction.

    Parameters
    ----------
    metric_series   : list of numeric Series, all the same length and aligned.
    weights         : per-metric weights summing to 1. Equal weights if None.
    ascending_flags : True  → higher raw value is better (revenue, rating).
                      False → lower raw value is better  (delivery time, late rate).
                      All True if None.

    Returns pd.Series of composite scores (higher = better).

    Q34 best investment category:
        composite_score([revenue_s, units_s, rating_s], weights=[0.5, 0.25, 0.25])

    Q35 strategic seller partners:
        composite_score([revenue_s, rating_s, delivery_s],
                        weights=[0.4, 0.4, 0.2],
                        ascending_flags=[True, True, False])

    Q36 underpenetrated growth markets:
        composite_score([rev_per_customer_s, orders_per_customer_s])
    """
    n = len(metric_series)
    if weights is None:
        weights = [1 / n] * n
    if ascending_flags is None:
        ascending_flags = [True] * n

    frame = pd.DataFrame({i: pd.Series(s) for i, s in enumerate(metric_series)})
    for i, (asc, w) in enumerate(zip(ascending_flags, weights)):
        col_min, col_max = frame[i].min(), frame[i].max()
        if col_max == col_min:
            frame[i] = 0.5 * w
        else:
            norm = (frame[i] - col_min) / (col_max - col_min)
            frame[i] = (1 - norm if not asc else norm) * w

    return frame.sum(axis=1).round(4)

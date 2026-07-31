"""
analytics_controller.py

Purpose:
--------
Simple task-based controller for the analytics module.

Takes a task description string and returns Plotly figures
by orchestrating the computation and visualization layers.

Usage:
------
from analytics_controller import analyze

# Examples:
fig = analyze("total revenue")
fig = analyze("revenue by seller as bar chart")
fig = analyze("delivery time trend over months")
fig = analyze("top 10 customers pareto")
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from date_analysis_functions.data_computation import *
from date_analysis_functions.data_visualization import *



# ============================================================
# TASK PATTERNS
# ============================================================

TASK_PATTERNS = {
    # KPI / Summary metrics
    "total": {"compute": "total", "visual": "kpi"},
    "sum": {"compute": "total", "visual": "kpi"},
    "count": {"compute": "count", "visual": "kpi"},
    "average": {"compute": "average", "visual": "kpi"},
    "mean": {"compute": "average", "visual": "kpi"},
    "median": {"compute": "median", "visual": "kpi"},
    
    # Distribution
    "distribution": {"compute": "summary_statistics", "visual": "distribution"},
    "histogram": {"compute": None, "visual": "distribution"},
    "box plot": {"compute": None, "visual": "box"},
    
    # Trend over time
    "trend": {"compute": "time_series", "visual": "line"},
    "over time": {"compute": "time_series", "visual": "line"},
    "monthly": {"compute": "time_series", "visual": "line"},
    
    # Comparison by category
    "by ": {"compute": "aggregate", "visual": "bar"},
    "per ": {"compute": "aggregate", "visual": "bar"},
    
    # Pareto / concentration
    "pareto": {"compute": "pareto", "visual": "pareto"},
    "top": {"compute": "rank", "visual": "bar"},
    "concentration": {"compute": "concentration", "visual": "kpi"},
    
    # Relationships
    "vs": {"compute": "correlation", "visual": "scatter"},
    "relationship": {"compute": "correlation", "visual": "scatter"},
    "correlation": {"compute": "correlation", "visual": "scatter"},
    
    # Delivery specific
    "delivery time": {"compute": "delivery_duration", "visual": "distribution"},
    "delivery delay": {"compute": "delivery_delay", "visual": "distribution"},
    "delivery rate": {"compute": "delivery_status", "visual": "kpi"},
    
    # Growth
    "growth": {"compute": "growth_rate", "visual": "dual_axis"},
    "change": {"compute": "growth_rate", "visual": "line"},
}


# ============================================================
# CONTROLLER
# ============================================================

def analyze(
    task: str,
    df: pd.DataFrame,
    column: Optional[str] = None,
    group_column: Optional[str] = None,
    date_column: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> Union[go.Figure, List[go.Figure], Dict[str, Any]]:
    """
    Execute analytics task from a simple string description.
    
    Parameters
    ----------
    task : str
        Task description (e.g., "total revenue", "revenue by seller",
        "delivery time trend", "top 10 customers pareto")
    df : pd.DataFrame
        Input data
    column : str, optional
        Primary metric column (e.g., "revenue", "rating")
    group_column : str, optional
        Grouping column (e.g., "seller", "category", "customer")
    date_column : str, optional
        Date column for time series
    title : str, optional
        Chart title (auto-generated if not provided)
    **kwargs
        Additional parameters passed to compute/visual functions
    
    Returns
    -------
    Plotly figure, list of figures, or dict of metrics
    
    Examples
    --------
    >>> analyze("total revenue", df, column="revenue")
    >>> analyze("revenue by seller", df, column="revenue", group_column="seller")
    >>> analyze("monthly revenue trend", df, column="revenue", date_column="date")
    >>> analyze("top 10 sellers pareto", df, column="revenue", group_column="seller")
    >>> analyze("revenue vs rating", df, column="revenue", group_column="rating")
    """
    
    task_lower = task.lower()
    
    # Detect pattern
    pattern = None
    matched_key = None
    
    for key, value in TASK_PATTERNS.items():
        if key in task_lower:
            pattern = value
            matched_key = key
            break
    
    if pattern is None:
        raise ValueError(f"Unknown task pattern: {task}. Available patterns: {list(TASK_PATTERNS.keys())}")
    
    # Auto-generate title
    if title is None:
        title = f"{task.capitalize()}"
        if column:
            title += f" - {column}"
        if group_column:
            title += f" by {group_column}"
    
    # Execute computation
    compute_result = _execute_computation(
        pattern["compute"],
        df,
        column,
        group_column,
        date_column,
        **kwargs
    )
    
    # Execute visualization
    fig = _execute_visualization(
        pattern["visual"],
        compute_result,
        df,
        column,
        group_column,
        title,
        **kwargs
    )
    
    return fig


def _execute_computation(
    compute_type: Optional[str],
    df: pd.DataFrame,
    column: Optional[str],
    group_column: Optional[str],
    date_column: Optional[str],
    **kwargs
) -> Any:
    """Execute computation layer based on pattern type."""
    
    if compute_type is None:
        return None
    
    if compute_type == "total":
        return total(df[column])
    
    if compute_type == "count":
        return count(df[column])
    
    if compute_type == "average":
        return average(df[column])
    
    if compute_type == "median":
        return median(df[column])
    
    if compute_type == "summary_statistics":
        return summary_statistics(df[column])
    
    if compute_type == "aggregate":
        agg_func = kwargs.get("agg", "sum")
        return aggregate_metric(df, group_column, column, agg_func)
    
    if compute_type == "time_series":
        freq = kwargs.get("freq", "ME")
        agg = kwargs.get("agg", "sum")
        return aggregate_time_period(df[date_column], df[column], freq, agg)
    
    if compute_type == "pareto":
        return pareto_table(df, group_column, column)
    
    if compute_type == "rank":
        top_n = kwargs.get("top_n", 10)
        agg = aggregate_metric(df, group_column, column, "sum")
        agg["rank"] = rank_values(agg[column], ascending=False)
        return agg[agg["rank"] <= top_n]
    
    if compute_type == "concentration":
        return concentration_analysis(df[column])
    
    if compute_type == "correlation":
        if group_column is None:
            raise ValueError("correlation requires two columns")
        return correlation(df[column], df[group_column])
    
    if compute_type == "delivery_duration":
        return delivery_duration(df["purchase_date"], df["delivered_date"])
    
    if compute_type == "delivery_delay":
        return delivery_delay(df["actual_date"], df["estimated_date"])
    
    if compute_type == "delivery_status":
        delays = delivery_delay(df["actual_date"], df["estimated_date"])
        return delivery_status_rates(delays)
    
    if compute_type == "growth_rate":
        return growth_rate(df[column])
    
    raise ValueError(f"Unknown compute type: {compute_type}")


def _execute_visualization(
    visual_type: str,
    compute_result: Any,
    df: pd.DataFrame,
    column: Optional[str],
    group_column: Optional[str],
    title: str,
    **kwargs
) -> Union[go.Figure, List[go.Figure], Dict[str, Any]]:
    """Execute visualization layer based on pattern type."""
    
    if visual_type == "kpi":
        if isinstance(compute_result, dict):
            return kpi_cards(compute_result, title)
        else:
            return kpi_cards({column or "Metric": compute_result}, title)
    
    if visual_type == "bar":
        return bar_chart(compute_result, x=group_column, y=column, title=title)
    
    if visual_type == "line":
        x_col = kwargs.get("x_col", "date")
        return line_chart(compute_result, x=x_col, y="value", title=title)
    
    if visual_type == "dual_axis":
        x_col = kwargs.get("x_col", "date")
        y_bar = kwargs.get("y_bar", "growth")
        return dual_axis_chart(
            compute_result,
            x=x_col,
            y_line="value",
            y_bar=y_bar,
            title=title
        )
    
    if visual_type == "distribution":
        chart_type = kwargs.get("chart_type", "histogram")
        if compute_result is None:
            return distribution_chart(df, column, chart_type, title)
        else:
            # compute_result is a Series of values
            temp_df = pd.DataFrame({"values": compute_result})
            return distribution_chart(temp_df, "values", chart_type, title)
    
    if visual_type == "pareto":
        return pareto_chart(
            compute_result,
            category=group_column,
            value=column,
            cumulative="cumulative_pct",
            title=title
        )
    
    if visual_type == "scatter":
        return scatter_chart(
            df,
            x=column,
            y=group_column,
            title=title
        )
    
    if visual_type == "box":
        return distribution_chart(df, column, "box", title)
    
    raise ValueError(f"Unknown visual type: {visual_type}")


# ============================================================
# QUICK HELPERS
# ============================================================

def kpi(task: str, df: pd.DataFrame, column: str, **kwargs) -> go.Figure:
    """Quick KPI display."""
    return analyze(task, df, column=column, **kwargs)


def compare(task: str, df: pd.DataFrame, column: str, group: str, **kwargs) -> go.Figure:
    """Quick comparison chart."""
    return analyze(task, df, column=column, group_column=group, **kwargs)


def trend(task: str, df: pd.DataFrame, column: str, date: str, **kwargs) -> go.Figure:
    """Quick trend chart."""
    return analyze(task, df, column=column, date_column=date, **kwargs)


def top_n(task: str, df: pd.DataFrame, column: str, group: str, n: int = 10, **kwargs) -> go.Figure:
    """Quick top N ranking."""
    return analyze(task, df, column=column, group_column=group, top_n=n, **kwargs)



"""
data_visualization.py

Purpose:
--------
Visualization layer of the analytics module.

The visualization module is responsible for displaying the results
produced by the computation layer.

It takes computed outputs and transforms them into:
- charts
- graphs
- dashboards
- KPI displays
- geographic visualizations

Inputs:
- Computed metrics
- Aggregated DataFrames
- Statistical outputs

Outputs:
- Visual representations of analytical results

The visualization layer DOES NOT perform calculations or modify
analytical results.

Rules:
------
- No calculations.
- No groupby operations.
- No metric creation.
- Receives prepared data.
- Returns Plotly figures.

Question Coverage:
------------------
Q1   kpi_cards
Q2   dual_axis_chart, line_chart
Q3   bar_chart
Q4   pareto_chart
Q5   pareto_chart
Q6   pareto_chart
Q7   bar_chart
Q8   distribution_chart
Q9   map_chart, bar_chart
Q10  bar_chart
Q11  bar_chart
Q12  scatter_chart
Q13  scatter_chart
Q14  bar_chart
Q15  bar_chart, scatter_chart
Q16  map_chart, bar_chart
Q17  map_chart
Q18  bar_chart, map_chart
Q19  map_chart, bar_chart
Q20  line_chart
Q21  bar_chart
Q22  map_chart
Q23  kpi_cards
Q24  scatter_chart
Q25  bar_chart
Q26  bar_chart
Q27  line_chart
Q28  bar_chart
Q29  distribution_chart, bar_chart
Q30  bar_chart
Q31  scatter_chart
Q32  scatter_chart
Q33  kpi_cards, bar_chart
Q34  bar_chart
Q35  bar_chart
Q36  scatter_chart
"""

from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# BAR CHART
# ============================================================

def bar_chart(
    df,
    x,
    y,
    title=""
):
    """
    Compare categories.

    Examples:
    - revenue by seller
    - sales by category
    """

    return px.bar(
        df,
        x=x,
        y=y,
        title=title
    )


# ============================================================
# LINE CHART
# ============================================================

def line_chart(
    df,
    x,
    y,
    title=""
):
    """
    Show trends over time.
    """

    return px.line(
        df,
        x=x,
        y=y,
        title=title
    )


# ============================================================
# DUAL AXIS CHART
# ============================================================

def dual_axis_chart(
    df,
    x,
    y_line,
    y_bar,
    title=""
):
    """
    Show a metric trend alongside period-over-period change.

    Left axis:  line  (e.g. monthly revenue)
    Right axis: bars  (e.g. month-over-month growth %)

    Useful for:
    - revenue trend with growth rate (Q2)
    - satisfaction trend with review volume (Q27)
    """

    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y_line],
            name=y_line,
            mode="lines+markers"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(
            x=df[x],
            y=df[y_bar],
            name=y_bar,
            opacity=0.4
        ),
        secondary_y=True
    )

    fig.update_layout(title=title)

    return fig


# ============================================================
# SCATTER CHART
# ============================================================

def scatter_chart(
    df,
    x,
    y,
    size=None,
    color=None,
    title=""
):
    """
    Show relationships.

    Examples:
    - delivery vs rating
    - revenue vs customers
    """

    return px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        title=title
    )


# ============================================================
# DISTRIBUTION CHART
# ============================================================

def distribution_chart(
    df,
    column,
    chart_type="histogram",
    title=""
):
    """
    Show value distribution.
    """

    if chart_type == "histogram":

        return px.histogram(
            df,
            x=column,
            title=title
        )

    if chart_type == "box":

        return px.box(
            df,
            y=column,
            title=title
        )


# ============================================================
# PARETO CHART
# ============================================================

def pareto_chart(
    df,
    category,
    value,
    cumulative,
    title=""
):
    """
    Display concentration.

    Requires:
    - category column
    - value column
    - cumulative percentage column
    """

    fig = go.Figure()

    fig.add_bar(
        x=df[category],
        y=df[value],
        name=value
    )

    fig.add_scatter(
        x=df[category],
        y=df[cumulative],
        mode="lines",
        name="Cumulative %"
    )

    return fig


# ============================================================
# HEATMAP
# ============================================================

def heatmap(
    df,
    x,
    y,
    value,
    title=""
):
    """
    Show two-dimensional patterns.
    """

    table = df.pivot_table(
        index=y,
        columns=x,
        values=value
    )

    return px.imshow(
        table,
        title=title
    )


# ============================================================
# GEOGRAPHIC MAP
# ============================================================

def map_chart(
    df,
    location,
    value,
    title=""
):
    """
    Geographic metric visualization.
    """

    return px.choropleth(
        df,
        locations=location,
        color=value,
        title=title
    )


# ============================================================
# KPI DISPLAY
# ============================================================

def kpi_cards(
    metrics: dict,
    title=""
):
    """
    Display important summary numbers.

    Useful for:
    - executive summary (Q1)
    - delivery rate summary (Q23)
    - freight cost summary (Q33)
    """

    n = len(metrics)

    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "indicator"}] * n]
    )

    for i, (name, value) in enumerate(metrics.items(), start=1):

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=float(value),
                title={"text": name}
            ),
            row=1,
            col=i
        )

    fig.update_layout(
        title=title,
        height=200
    )

    return fig
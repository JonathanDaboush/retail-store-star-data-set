"""
data_visualization.py
=====================
General-purpose visualization building blocks for retail star-schema analytics.

Every function accepts a pandas DataFrame and column-name parameters, returns a
plotly Figure, and can be further customised by the caller via .update_layout()
or .update_traces() after the call.

Dependencies: pandas, numpy, plotly
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design system ──────────────────────────────────────────────────────────────
_PALETTE = [
    "#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#4F46E5",
]
_PRIMARY  = "#2563EB"
_POSITIVE = "#16A34A"
_NEGATIVE = "#DC2626"
_NEUTRAL  = "#9CA3AF"

_BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Arial, sans-serif", size=13, color="#111827"),
    title_font=dict(size=17, color="#111827"),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(l=64, r=40, t=72, b=60),
    hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#E5E7EB"),
)


def _base(fig: go.Figure, title: str = "", height: int = 460) -> go.Figure:
    """Apply the shared design system to a figure and return it."""
    fig.update_layout(title=dict(text=title, x=0.04), height=height, **_BASE_LAYOUT)
    return fig


# ── 1. Bar chart ───────────────────────────────────────────────────────────────

def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    orientation: str = "v",
    top_n: Optional[int] = None,
    color_col: Optional[str] = None,
    text_col: Optional[str] = None,
    bar_color: str = _PRIMARY,
) -> go.Figure:
    """
    Vertical or horizontal bar chart, optionally limited to the top N rows.

    Parameters
    ----------
    x : str
        Category axis column (label when orientation='v', value when 'h').
    y : str
        Value axis column.
    orientation : str
        'v' (default) – vertical bars.  'h' – horizontal bars.
    top_n : int, optional
        Keep only the top N rows sorted by value before plotting.
    color_col : str, optional
        Column that maps bar colours to a discrete palette.
    text_col : str, optional
        Column whose values are displayed on each bar.
    bar_color : str
        Single colour used when color_col is not provided.

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q11 – revenue by product category (vertical)
    plot_bar(category_revenue, x='category', y='revenue', title='Revenue by Category')

    # Q14 – top 20 sellers by revenue (horizontal)
    plot_bar(sellers, x='revenue', y='seller_id', orientation='h', top_n=20,
             title='Top 20 Sellers by Revenue')
    """
    df = df.copy()
    sort_col = y if orientation == "v" else x
    df = df.sort_values(sort_col, ascending=(orientation == "h"))
    if top_n:
        df = df.tail(top_n) if orientation == "h" else df.head(top_n)

    kwargs: Dict = dict(
        data_frame=df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        orientation=orientation,
    )
    if color_col:
        kwargs["color"] = color_col
        kwargs["color_discrete_sequence"] = _PALETTE
    else:
        kwargs["color_discrete_sequence"] = [bar_color]

    if text_col:
        kwargs["text"] = text_col

    fig = px.bar(**kwargs)
    fig.update_traces(marker_line_width=0)
    return _base(fig, title)


# ── 2. Line chart ──────────────────────────────────────────────────────────────

def plot_line(
    df: pd.DataFrame,
    x: str,
    y: Union[str, List[str]],
    title: str = "",
    markers: bool = True,
    color_col: Optional[str] = None,
) -> go.Figure:
    """
    Single or multi-line chart.

    Parameters
    ----------
    x : str
        Horizontal axis column (typically a date or period column).
    y : str or list of str
        One column → single line.  List → one line per column.
    markers : bool
        Show point markers on the line.
    color_col : str, optional
        Column that splits the data into one line per category (long-format data).
        Only used when y is a single string.

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q2  – monthly revenue trend
    plot_line(monthly, x='month', y='revenue', title='Monthly Revenue')

    # Q2  – revenue + 3-month rolling average together
    plot_line(monthly, x='month', y=['revenue', 'rolling_3_revenue'],
              title='Revenue with Rolling Average')
    """
    mode = "lines+markers" if markers else "lines"

    if isinstance(y, list):
        fig = go.Figure()
        for i, col in enumerate(y):
            fig.add_trace(go.Scatter(
                x=df[x], y=df[col], name=col, mode=mode,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            ))
    else:
        kwargs: Dict = dict(
            data_frame=df, x=x, y=y, markers=markers,
            color_discrete_sequence=_PALETTE,
        )
        if color_col:
            kwargs["color"] = color_col
        fig = px.line(**kwargs)
        fig.update_traces(line_width=2)

    return _base(fig, title)


# ── 3. Time-series with trend overlay ─────────────────────────────────────────

def plot_time_series_with_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "",
    rolling_col: Optional[str] = None,
    growth_col: Optional[str] = None,
) -> go.Figure:
    """
    Time-series line with optional rolling-average overlay and growth-rate bars
    on a secondary y-axis.

    Parameters
    ----------
    date_col : str
        Datetime column (horizontal axis).
    value_col : str
        Primary numeric metric to plot as a line.
    rolling_col : str, optional
        Pre-computed rolling average column to overlay (dotted green line).
        Produce it with compute_rolling_average() before calling this function.
    growth_col : str, optional
        Period-over-period growth rate column to display as bars on a secondary
        axis (green = positive, red = negative).
        Produce it with compute_growth_rate() before calling this function.

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q2  – monthly revenue with rolling average and MoM growth
    monthly = compute_time_series(orders, 'purchase_date', 'revenue')
    monthly = compute_rolling_average(monthly, 'revenue', window=3)
    monthly = compute_growth_rate(monthly, 'revenue', growth_pct_col='growth_pct')
    plot_time_series_with_trend(monthly, 'purchase_date', 'revenue',
                                rolling_col='rolling_3_revenue',
                                growth_col='growth_pct',
                                title='Monthly Revenue Trend')
    """
    has_secondary = bool(growth_col and growth_col in df.columns)
    fig = make_subplots(specs=[[{"secondary_y": has_secondary}]])

    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[value_col], name=value_col,
        mode="lines+markers", line=dict(color=_PRIMARY, width=2),
    ), secondary_y=False)

    if rolling_col and rolling_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[rolling_col], name=rolling_col,
            mode="lines", line=dict(color=_POSITIVE, dash="dot", width=2),
        ), secondary_y=False)

    if has_secondary:
        bar_colors = df[growth_col].apply(
            lambda v: _POSITIVE if (pd.notna(v) and v >= 0) else _NEGATIVE
        )
        fig.add_trace(go.Bar(
            x=df[date_col], y=df[growth_col], name=growth_col,
            marker_color=bar_colors.tolist(), opacity=0.35,
        ), secondary_y=True)
        fig.update_yaxes(title_text="Growth Rate %", secondary_y=True)

    fig.update_yaxes(title_text=value_col, secondary_y=False)
    return _base(fig, title)


# ── 4. Pareto chart ────────────────────────────────────────────────────────────

def plot_pareto(
    df: pd.DataFrame,
    entity_col: str,
    value_col: str,
    cumulative_col: str,
    title: str = "",
    top_n: Optional[int] = None,
) -> go.Figure:
    """
    Pareto chart: descending bars for each entity + cumulative % line on a
    secondary axis, with a dashed 80 % reference line.

    Parameters
    ----------
    entity_col : str
        Category label column (x-axis).
    value_col : str
        Numeric column driving bar heights.
    cumulative_col : str
        Pre-computed cumulative % column (from compute_pareto()).
    top_n : int, optional
        Limit to the top N entities.

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q5  – seller revenue Pareto
    seller_rev = aggregate_metric(orders, 'seller_id', 'revenue', 'sum')
    seller_rev = compute_pareto(seller_rev, 'revenue')
    plot_pareto(seller_rev, 'seller_id', 'revenue', 'cumulative_pct',
                title='Seller Revenue Pareto', top_n=30)
    """
    df = df.sort_values(value_col, ascending=False)
    if top_n:
        df = df.head(top_n)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df[entity_col], y=df[value_col], name=value_col,
        marker_color=_PRIMARY, opacity=0.82, marker_line_width=0,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df[entity_col], y=df[cumulative_col], name="Cumulative %",
        mode="lines+markers", line=dict(color=_NEGATIVE, width=2),
    ), secondary_y=True)

    fig.add_hline(y=80, line_dash="dash", line_color=_NEUTRAL,
                  annotation_text="80 %", annotation_position="top right",
                  secondary_y=True)

    fig.update_yaxes(title_text=value_col, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    return _base(fig, title)


# ── 5. Scatter / Bubble chart ─────────────────────────────────────────────────

def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    trendline: bool = True,
    hover_cols: Optional[List[str]] = None,
) -> go.Figure:
    """
    Scatter or bubble chart with an optional OLS trendline.

    Parameters
    ----------
    x : str
        Horizontal axis column.
    y : str
        Vertical axis column.
    color_col : str, optional
        Column that maps points to discrete colours.
    size_col : str, optional
        Column that controls bubble radius (bubble chart mode).
    trendline : bool
        Draw an OLS linear trendline (default True).
    hover_cols : list of str, optional
        Additional columns shown in the hover tooltip.

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q24 – delivery time vs review score correlation
    plot_scatter(orders, x='delivery_days', y='review_score',
                 title='Delivery Time vs Customer Rating')

    # Q36 – underpenetrated markets: revenue vs customer count (bubbles by state)
    plot_scatter(geo, x='customer_count', y='revenue_per_customer',
                 size_col='revenue', color_col='state',
                 title='Market Penetration Map', trendline=False)
    """
    kwargs: Dict = dict(
        data_frame=df, x=x, y=y,
        trendline="ols" if trendline else None,
        color_discrete_sequence=_PALETTE,
        opacity=0.75,
    )
    if color_col:
        kwargs["color"] = color_col
    if size_col:
        kwargs["size"] = size_col
    if hover_cols:
        kwargs["hover_data"] = hover_cols

    fig = px.scatter(**{k: v for k, v in kwargs.items() if v is not None})
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _base(fig, title)


# ── 6. Choropleth map ──────────────────────────────────────────────────────────

def plot_choropleth(
    df: pd.DataFrame,
    location_col: str,
    value_col: str,
    title: str = "",
    geojson=None,
    featureidkey: Optional[str] = None,
    locationmode: str = "geojson-id",
    scope: Optional[str] = None,
    colorscale: str = "Blues",
) -> go.Figure:
    """
    Choropleth map for geographic metric display.

    For Brazilian state-level maps pass a Brazil GeoJSON via the geojson
    parameter and set featureidkey to match the property that contains the
    state code in that GeoJSON (e.g. 'properties.sigla').

    Parameters
    ----------
    location_col : str
        Column whose values match the geographic identifiers in geojson or
        the built-in Plotly location set.
    value_col : str
        Numeric metric shown as colour intensity.
    geojson : dict, optional
        GeoJSON FeatureCollection for custom boundaries (e.g. Brazilian states).
    featureidkey : str, optional
        Path inside each GeoJSON feature used to match location_col values.
    locationmode : str
        Used only when geojson is None (e.g. 'USA-states', 'country names').
    scope : str, optional
        Map scope when geojson is None (e.g. 'south america', 'world').
    colorscale : str
        Plotly colorscale name.

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q17 – revenue by Brazilian state (requires Brazil GeoJSON)
    plot_choropleth(state_rev, location_col='customer_state',
                    value_col='revenue', geojson=brazil_geojson,
                    featureidkey='properties.sigla',
                    title='Revenue by State')
    """
    kwargs: Dict = dict(
        data_frame=df,
        locations=location_col,
        color=value_col,
        color_continuous_scale=colorscale,
    )
    if geojson is not None:
        kwargs["geojson"] = geojson
        kwargs["featureidkey"] = featureidkey
    else:
        kwargs["locationmode"] = locationmode
        if scope:
            kwargs["scope"] = scope

    fig = px.choropleth(**kwargs)
    fig.update_geos(showframe=False, showcoastlines=True, coastlinecolor="#E5E7EB")
    return _base(fig, title, height=500)


# ── 7. Distribution plot ───────────────────────────────────────────────────────

def plot_distribution(
    df: pd.DataFrame,
    col: str,
    title: str = "",
    kind: str = "histogram",
    nbins: int = 40,
    group_col: Optional[str] = None,
) -> go.Figure:
    """
    Plot the distribution of a numeric column.

    Parameters
    ----------
    col : str
        Numeric column to visualise.
    kind : str
        'histogram' (default), 'box', or 'violin'.
    nbins : int
        Number of histogram bins (ignored for box / violin).
    group_col : str, optional
        Column used to split distributions by category (colour-coded).

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q8  – order value distribution
    plot_distribution(orders, 'order_value', title='Order Value Distribution')

    # Q29 – payment value by installment count
    plot_distribution(payments, 'payment_value', kind='box',
                      group_col='payment_installments',
                      title='Payment Value by Instalment Count')
    """
    kwargs: Dict = dict(
        data_frame=df, title=title, color_discrete_sequence=_PALETTE
    )
    if group_col:
        kwargs["color"] = group_col

    if kind == "histogram":
        fig = px.histogram(df, x=col, nbins=nbins, **{
            k: v for k, v in kwargs.items()
            if k not in ("data_frame", "title")
        })
        fig.update_traces(marker_line_width=0.4, marker_line_color="white")
    elif kind == "box":
        fig = px.box(df, y=col, x=group_col, **{
            k: v for k, v in kwargs.items()
            if k not in ("data_frame", "title")
        })
    elif kind == "violin":
        fig = px.violin(df, y=col, x=group_col, box=True, **{
            k: v for k, v in kwargs.items()
            if k not in ("data_frame", "title")
        })
    else:
        raise ValueError(f"kind must be 'histogram', 'box', or 'violin'. Got: {kind!r}")

    return _base(fig, title)


# ── 8. Heatmap ────────────────────────────────────────────────────────────────

def plot_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: str = "",
    colorscale: str = "Blues",
    aggfunc: str = "mean",
) -> go.Figure:
    """
    Heatmap from long-form data, pivoted to a x × y grid.

    Parameters
    ----------
    x : str
        Column mapped to the horizontal axis (e.g. month, payment_type).
    y : str
        Column mapped to the vertical axis (e.g. state, category).
    value : str
        Numeric column aggregated into each cell.
    colorscale : str
        Plotly colorscale name.
    aggfunc : str
        Pivot aggregation function: 'mean', 'sum', 'count'.

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q24 – average review score by month × delivery status
    plot_heatmap(orders, x='month', y='delivery_status',
                 value='review_score', title='Rating Heatmap')

    # Q19 – freight ratio by state × month
    plot_heatmap(orders, x='month', y='customer_state',
                 value='freight_ratio', aggfunc='mean',
                 title='Freight Ratio by State and Month')
    """
    pivot = df.pivot_table(index=y, columns=x, values=value, aggfunc=aggfunc)
    fig = px.imshow(
        pivot,
        color_continuous_scale=colorscale,
        aspect="auto",
        text_auto=".2f",
    )
    fig.update_xaxes(side="bottom")
    return _base(fig, title, height=520)


# ── 9. KPI cards ──────────────────────────────────────────────────────────────

def plot_kpi_cards(
    metrics: Dict[str, float],
    title: str = "",
    number_format: Optional[Dict[str, Dict]] = None,
) -> go.Figure:
    """
    Render a row of numeric KPI indicator cards.

    Parameters
    ----------
    metrics : dict
        {label: numeric_value} mapping.  One card is rendered per entry.
    number_format : dict, optional
        Per-label format overrides passed to go.Indicator's 'number' parameter.
        e.g. {'Total Revenue': {'prefix': 'R$ ', 'valueformat': ',.0f'}}

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q1  – executive summary metrics
    plot_kpi_cards({
        'Total Revenue':    12_500_000,
        'Total Orders':     99_441,
        'Total Customers':  96_096,
        'Total Sellers':    3_095,
        'Avg Order Value':  154.1,
    }, title='Executive Revenue Summary')
    """
    n = len(metrics)
    if n == 0:
        return go.Figure()

    if number_format is None:
        number_format = {}

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "indicator"}] * n],
    )
    for i, (label, value) in enumerate(metrics.items(), start=1):
        num_kwargs = number_format.get(label, {"font": {"size": 34, "color": _PRIMARY}})
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=float(value),
                number=num_kwargs,
                title={"text": label, "font": {"size": 13, "color": "#374151"}},
            ),
            row=1, col=i,
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, -apple-system, Arial, sans-serif"),
        paper_bgcolor="#FFFFFF",
        title=dict(text=title, font=dict(size=17, color="#111827"), x=0.04),
        height=180,
        margin=dict(l=24, r=24, t=56, b=16),
    )
    return fig


# ── 10. Grouped / stacked bar chart ───────────────────────────────────────────

def plot_grouped_bar(
    df: pd.DataFrame,
    x: str,
    y_cols: List[str],
    title: str = "",
    barmode: str = "group",
    top_n: Optional[int] = None,
    orientation: str = "v",
) -> go.Figure:
    """
    Grouped or stacked bar chart for comparing multiple metrics per category.

    Parameters
    ----------
    x : str
        Category column (horizontal axis when orientation='v').
    y_cols : list of str
        Numeric columns, each rendered as a separate bar series.
    barmode : str
        'group' (default) – bars side by side.  'stack' – bars stacked.
    top_n : int, optional
        Limit to the first N rows of the (pre-sorted) DataFrame.
    orientation : str
        'v' (default) – vertical.  'h' – horizontal (x becomes the value axis).

    Returns
    -------
    go.Figure

    Examples
    --------
    # Q12 – units sold vs revenue per product (side-by-side)
    plot_grouped_bar(products, x='product_name',
                     y_cols=['units_sold', 'revenue'],
                     title='Volume vs Revenue per Product')

    # Q11 – category revenue + units stacked
    plot_grouped_bar(categories, x='category',
                     y_cols=['revenue', 'units_sold'],
                     barmode='stack', title='Category Performance')
    """
    if top_n:
        df = df.head(top_n)

    fig = go.Figure()
    for i, col in enumerate(y_cols):
        bar_args: Dict = dict(
            name=col,
            marker_color=_PALETTE[i % len(_PALETTE)],
            marker_line_width=0,
            opacity=0.85,
        )
        if orientation == "v":
            bar_args["x"] = df[x]
            bar_args["y"] = df[col]
        else:
            bar_args["y"] = df[x]
            bar_args["x"] = df[col]
            bar_args["orientation"] = "h"

        fig.add_trace(go.Bar(**bar_args))

    fig.update_layout(barmode=barmode)
    return _base(fig, title)

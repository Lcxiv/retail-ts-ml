"""
Visualization Module.

All charts are built with Plotly for interactivity.
Designed for both notebook and standalone HTML output.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_sales_trend(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "units_sold",
    color_col: str = "retailer_id",
    title: str = "Sales Trend Over Time",
) -> go.Figure:
    """Line chart of sales over time, colored by retailer or object."""
    agg = df.groupby([date_col, color_col])[target_col].sum().reset_index()
    fig = px.line(
        agg, x=date_col, y=target_col, color=color_col,
        title=title, template="plotly_dark",
        labels={target_col: target_col.replace("_", " ").title(), date_col: "Date"},
    )
    fig.update_layout(hovermode="x unified", legend_title_text=color_col.replace("_", " ").title())
    return fig


def plot_seasonality_heatmap(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "units_sold",
    retailer_col: str = "retailer_id",
    title: str = "Seasonality Heatmap (Week vs Year)",
) -> go.Figure:
    """Heatmap of average weekly sales by year and week number."""
    df = df.copy()
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    df["week"] = pd.to_datetime(df[date_col]).dt.isocalendar().week.astype(int)
    pivot = df.groupby(["year", "week"])[target_col].mean().unstack("week")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Viridis",
        colorbar_title=target_col,
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Week of Year", yaxis_title="Year",
    )
    return fig


def plot_forecast_vs_actual(
    test_df: pd.DataFrame,
    date_col: str = "date",
    actual_col: str = "units_sold",
    pred_col: str = "prediction",
    group_col: str | None = "retailer_id",
    title: str = "Forecast vs Actual",
) -> go.Figure:
    """Overlay actual vs predicted values."""
    if group_col and group_col in test_df.columns:
        agg = test_df.groupby([date_col, group_col])[[actual_col, pred_col]].mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        for g, grp in agg.groupby(group_col):
            fig.add_trace(go.Scatter(
                x=grp[date_col], y=grp[actual_col], name=f"{g} actual",
                mode="lines", line=dict(dash="solid")
            ))
            fig.add_trace(go.Scatter(
                x=grp[date_col], y=grp[pred_col], name=f"{g} forecast",
                mode="lines", line=dict(dash="dash")
            ))
    else:
        agg = test_df.groupby(date_col)[[actual_col, pred_col]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg[date_col], y=agg[actual_col], name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=agg[date_col], y=agg[pred_col], name="Forecast", mode="lines", line=dict(dash="dash")))

    fig.update_layout(title=title, template="plotly_dark", hovermode="x unified")
    return fig


def plot_error_by_retailer(
    metrics_df: pd.DataFrame,
    retailer_col: str = "retailer_id",
    metric: str = "MAE",
    title: str | None = None,
) -> go.Figure:
    """Bar chart of a given error metric by retailer."""
    title = title or f"{metric} by Retailer"
    fig = px.bar(
        metrics_df.sort_values(metric, ascending=True),
        x=metric, y=retailer_col, orientation="h",
        title=title, template="plotly_dark", color=metric,
        color_continuous_scale="RdYlGn_r",
    )
    return fig


def plot_rolling_average_comparison(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "units_sold",
    retailer_col: str = "retailer_id",
    window: int = 8,
    title: str = "Rolling Average Sales Comparison",
) -> go.Figure:
    """Compare retailers using rolling average smoothing."""
    agg = df.groupby([date_col, retailer_col])[target_col].sum().reset_index()
    agg = agg.sort_values([retailer_col, date_col])
    agg[f"rolling_{window}w"] = agg.groupby(retailer_col)[target_col].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    fig = px.line(
        agg, x=date_col, y=f"rolling_{window}w", color=retailer_col,
        title=title, template="plotly_dark",
        labels={f"rolling_{window}w": f"{window}-week Rolling Avg"},
    )
    return fig

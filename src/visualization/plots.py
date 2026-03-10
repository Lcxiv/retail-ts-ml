"""Visualization Module.

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
    target_col: str = "qty_sold",
    color_col: str = "retailer",
    title: str = "Sales Trend Over Time",
) -> go.Figure:
    """Line chart of sales over time, colored by retailer or store."""
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
    target_col: str = "qty_sold",
    retailer_col: str = "retailer",
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
    actual_col: str = "qty_sold",
    pred_col: str = "prediction",
    group_col: str | None = "retailer",
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
    retailer_col: str = "retailer",
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
    target_col: str = "qty_sold",
    retailer_col: str = "retailer",
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


def plot_distribution(
    df: pd.DataFrame,
    target_col: str = "qty_sold",
    group_col: str = "retailer",
    title: str = "Target Distribution by Group",
) -> go.Figure:
    """Histogram + KDE of target variable by group."""
    fig = px.histogram(
        df, x=target_col, color=group_col, marginal="box",
        title=title, template="plotly_dark",
        barmode="overlay", opacity=0.7,
    )
    return fig


def plot_feature_importance(
    fi_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Feature Importances",
) -> go.Figure:
    """Horizontal bar chart of top N feature importances."""
    top = fi_df.head(top_n).sort_values("importance", ascending=True)
    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        title=title, template="plotly_dark",
        color="importance", color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Importance")
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "MAE",
    title: str = "Model Comparison",
) -> go.Figure:
    """Bar chart comparing models on a given metric."""
    fig = px.bar(
        results_df.sort_values(metric, ascending=True),
        x="model", y=metric,
        title=title, template="plotly_dark",
        color=metric, color_continuous_scale="RdYlGn_r",
    )
    return fig

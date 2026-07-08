"""
charts.py
---------
Centralised Plotly chart builders used throughout the dashboard
(analytics page, prediction result page, sustainability gauge, etc).
Keeping chart construction here keeps app.py focused on layout/flow.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import CATEGORY_COLORS

THEME_GREEN = "#1DB954"
THEME_DARKGRAY = "#1E1E1E"
THEME_WHITE = "#FFFFFF"

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#2E2E2E"),
    margin=dict(l=30, r=30, t=50, b=30),
)


def gauge_chart(score: int, title="Sustainability Score"):
    """Build a gauge/progress-ring chart for the sustainability score."""
    color = THEME_GREEN if score >= 75 else ("#F5A623" if score >= 45 else "#E5484D")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 18}},
            number={"suffix": " / 100", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E5E5E5",
                "steps": [
                    {"range": [0, 45], "color": "#FDEDEE"},
                    {"range": [45, 75], "color": "#FEF6E7"},
                    {"range": [75, 100], "color": "#E9F9EF"},
                ],
            },
        )
    )
    fig.update_layout(height=280, **PLOTLY_LAYOUT_DEFAULTS)
    return fig


def category_pie_chart(df: pd.DataFrame):
    """Pie chart of predicted waste category distribution."""
    counts = df["predicted_category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.pie(
        counts,
        names="category",
        values="count",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        hole=0.55,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=340, showlegend=True, **PLOTLY_LAYOUT_DEFAULTS)
    return fig


def category_bar_chart(df: pd.DataFrame):
    """Bar chart comparing average waste % per category."""
    grouped = df.groupby("predicted_category")["waste_percentage"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="predicted_category",
        y="waste_percentage",
        color="predicted_category",
        color_discrete_map=CATEGORY_COLORS,
        text_auto=".1f",
    )
    fig.update_layout(
        height=340,
        showlegend=False,
        xaxis_title="Category",
        yaxis_title="Avg Waste %",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def trend_line_chart(df: pd.DataFrame):
    """Line chart of waste % over time (chronological trend)."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    fig = px.line(
        data,
        x="timestamp",
        y="waste_percentage",
        markers=True,
    )
    fig.update_traces(line_color=THEME_GREEN)
    fig.update_layout(
        height=340,
        xaxis_title="Time",
        yaxis_title="Waste %",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def daily_stats_bar(df: pd.DataFrame):
    """Bar chart of average waste % grouped by day."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["day"] = data["timestamp"].dt.date
    grouped = data.groupby("day")["waste_percentage"].mean().reset_index()
    fig = px.bar(grouped, x="day", y="waste_percentage")
    fig.update_traces(marker_color=THEME_GREEN)
    fig.update_layout(
        height=320,
        xaxis_title="Day",
        yaxis_title="Avg Waste %",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def monthly_stats_bar(df: pd.DataFrame):
    """Bar chart of average waste % grouped by month."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["month"] = data["timestamp"].dt.to_period("M").astype(str)
    grouped = data.groupby("month")["waste_percentage"].mean().reset_index()
    fig = px.bar(grouped, x="month", y="waste_percentage")
    fig.update_traces(marker_color="#0F9D58")
    fig.update_layout(
        height=320,
        xaxis_title="Month",
        yaxis_title="Avg Waste %",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def confidence_bar_chart(probabilities: dict):
    """Horizontal bar chart of model class probabilities for a prediction."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[CATEGORY_COLORS.get(l, THEME_GREEN) for l in labels],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=220,
        xaxis_title="Confidence (%)",
        xaxis_range=[0, 100],
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig

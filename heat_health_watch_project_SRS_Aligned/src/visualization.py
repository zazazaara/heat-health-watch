"""Plotly chart builders (pure functions returning figures; no Streamlit imports)."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C

_LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), height=380,
               font=dict(family="Inter, Segoe UI, sans-serif", size=13),
               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))


def _base(fig: go.Figure, title: str, height: int = None) -> go.Figure:
    fig.update_layout(title=title, **_LAYOUT)
    if height:
        fig.update_layout(height=height)
    return fig


def line_chart(df: pd.DataFrame, x: str, ys: List[str], title: str, ytitle: str = "") -> go.Figure:
    fig = go.Figure()
    for y in ys:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y.replace("_", " ")))
    fig.update_yaxes(title_text=ytitle)
    return _base(fig, title)


def distribution_bar(counts: Dict[str, int], title: str, order: List[str] = None) -> go.Figure:
    order = order or list(counts)
    vals = [counts.get(k, 0) for k in order]
    fig = go.Figure(go.Bar(x=order, y=vals, marker_color=[C.LEVEL_COLORS.get(k, "#607d8b") for k in order],
                           text=vals, textposition="outside"))
    fig.update_yaxes(title_text="Days")
    return _base(fig, title, 340)


def score_with_levels(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines", name="Risk score", line=dict(color="#37474f")))
    for lo, name in C.RISK_LEVELS[1:]:
        fig.add_hline(y=lo, line_dash="dot", line_color=C.LEVEL_COLORS[name],
                      annotation_text=f"{name} ≥ {lo:g}", annotation_position="top left")
    fig.update_yaxes(title_text="Score (0-100)")
    return _base(fig, title)


def confusion_heatmap(cm: List[List[int]], labels: List[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred {l}" for l in labels], y=[f"Actual {l}" for l in labels],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, "Confusion matrix (held-out test block)", 340)


def importance_bar(importances: Dict[str, float]) -> go.Figure:
    items = sorted(importances.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color="#1565c0"))
    fig.update_xaxes(title_text="Importance")
    return _base(fig, "Random Forest feature importance", 420)


def forecast_context_chart(observed: pd.DataFrame, synthetic: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Show observed history, generated continuation, and the forecast as distinct sources."""
    fig = go.Figure()
    if not observed.empty:
        fig.add_trace(go.Scatter(
            x=observed[C.DATE_COL], y=observed["Max_Temperature_C"], mode="lines",
            name="Original observed Tmax", line=dict(color="#1565c0", width=1.8),
        ))
    if not synthetic.empty:
        fig.add_trace(go.Scatter(
            x=synthetic[C.DATE_COL], y=synthetic["Max_Temperature_C"], mode="lines",
            name="Synthetic continuation (not observed)",
            line=dict(color="#ef6c00", width=1.8, dash="dash"),
        ))
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["Date"], y=forecast["Max_Temperature_C_high80"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["Date"], y=forecast["Max_Temperature_C_low80"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(46,158,91,0.16)",
            name="Forecast 80% interval", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["Date"], y=forecast["Max_Temperature_C"], mode="lines+markers",
            name="Experimental forecast", line=dict(color="#2e9e5b", width=2.5),
        ))
    fig.update_yaxes(title_text="Maximum temperature (°C)")
    fig.update_xaxes(title_text="Date")
    return _base(fig, "Temperature timeline: observed → synthetic continuation → forecast", 430)


def forecast_temperature_chart(f: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Max_Temperature_C_high80"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Max_Temperature_C_low80"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(239,108,0,0.18)", name="80% band (Tmax)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Max_Temperature_C"], mode="lines+markers", name="Max temp (est.)",
                             line=dict(color="#ef6c00")))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Mean_Temperature_C"], mode="lines", name="Mean temp (est.)",
                             line=dict(color="#1565c0", dash="dash")))
    fig.update_yaxes(title_text="°C")
    return _base(fig, "Estimated temperature")


def forecast_score_chart(f: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Risk_Score_high80"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Risk_Score_low80"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(55,71,79,0.15)", name="80% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Risk_Score"], mode="lines+markers", name="Estimated score",
                             line=dict(color="#37474f"),
                             marker=dict(color=[C.LEVEL_COLORS[c] for c in f["Score_Category_5L"]], size=7)))
    for lo, name in C.RISK_LEVELS[1:]:
        fig.add_hline(y=lo, line_dash="dot", line_color=C.LEVEL_COLORS[name])
    fig.update_yaxes(title_text="Score (0-100)", range=[0, 100])
    return _base(fig, "Estimated Heat-Health Risk Score (project-defined levels)")


def hourly_chart(h: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=h["Time"], y=h["Temperature_C"], name="Temperature (°C)",
                             line=dict(color="#ef6c00")), secondary_y=False)
    fig.add_trace(go.Scatter(x=h["Time"], y=h["Risk_Score"], name="Risk score",
                             line=dict(color="#37474f"), mode="lines+markers",
                             marker=dict(color=[C.LEVEL_COLORS[c] for c in h["Score_Category_5L"]], size=8)),
                  secondary_y=True)
    fig.update_yaxes(title_text="°C", secondary_y=False)
    fig.update_yaxes(title_text="Score (0-100)", secondary_y=True)
    fig.update_xaxes(tickangle=-45)
    return _base(fig, "SIMULATED hourly temperature and risk score")


def synthetic_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for lvl in C.LEVEL_ORDER:
        d = df[df["Score_Category_5L"] == lvl]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["Max_Temperature_C"], y=d["AQI"], mode="markers", name=lvl,
                                 marker=dict(color=C.LEVEL_COLORS[lvl], size=10, line=dict(width=1, color="white")),
                                 text=d["Risk_Score"], hovertemplate="Tmax %{x}°C<br>AQI %{y}<br>score %{text}<extra>" + lvl + "</extra>"))
    fig.update_xaxes(title_text="Max temperature (°C)")
    fig.update_yaxes(title_text="AQI")
    return _base(fig, "Synthetic scenarios by project-defined level")

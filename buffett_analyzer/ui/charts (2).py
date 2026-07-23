"""Plotly 차트 (화면 3·4·7·9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#123B2E"
ACCENT = "#B8860B"
RED = "#B03A2E"
GRID = "#E4E4DA"


def _base(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title, template="plotly_white", height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1C1C1A"), legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def roic_chart(table: pd.DataFrame, wacc: float | None, rf: float | None):
    d = table["roic"].dropna() * 100
    fig = go.Figure(go.Bar(x=d.index.astype(int), y=d.values,
                           marker_color=INK, name="ROIC"))
    if wacc:
        fig.add_hline(y=wacc * 100, line_dash="dash", line_color=RED,
                      annotation_text=f"WACC {wacc:.1%}")
    if rf:
        fig.add_hline(y=rf * 100, line_dash="dot", line_color=ACCENT,
                      annotation_text=f"무위험 {rf:.1%}")
    fig.update_yaxes(title="ROIC (%)")
    return _base(fig, "연도별 ROIC vs 자본비용")


def cash_chart(table: pd.DataFrame):
    fig = go.Figure()
    colors = {"revenue": "#8A8A7A", "net_income": ACCENT,
              "ocf": "#3E6B54", "fcf": INK}
    labels = {"revenue": "매출", "net_income": "순이익",
              "ocf": "영업현금흐름", "fcf": "FCF"}
    for col, c in colors.items():
        s = table[col].dropna()
        fig.add_trace(go.Bar(x=s.index.astype(int), y=s.values,
                             name=labels[col], marker_color=c))
    fig.update_layout(barmode="group")
    return _base(fig, "매출·이익·현금흐름")


def margin_chart(table: pd.DataFrame):
    fig = go.Figure()
    for col, name, c in (("fcf_margin", "FCF 마진", INK),
                         ("conversion", "현금전환율", ACCENT)):
        s = table[col].dropna() * 100
        fig.add_trace(go.Scatter(x=s.index.astype(int), y=s.values,
                                 name=name, mode="lines+markers",
                                 line=dict(color=c)))
    fig.update_yaxes(title="%")
    return _base(fig, "FCF 마진 · 현금전환율")


def debt_chart(net_debt: pd.Series):
    s = net_debt.dropna()
    colors = [RED if v > 0 else INK for v in s.values]
    fig = go.Figure(go.Bar(x=s.index.astype(int), y=s.values,
                           marker_color=colors, name="순부채(+)/순현금(−)"))
    return _base(fig, "순부채 추이")


def dcf_chart(scen: dict, price: float | None, currency: str | None):
    names = list(scen.keys())
    vals = [scen[n]["fair"] or np.nan for n in names]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[ACCENT, INK, "#3E6B54"],
                           name="적정가치"))
    if price:
        fig.add_hline(y=price, line_dash="dash", line_color=RED,
                      annotation_text="현재주가")
    return _base(fig, "DCF 시나리오별 주당 적정가치")


def sens_heatmap(df: pd.DataFrame):
    fig = go.Figure(go.Heatmap(
        z=df.values, x=list(df.columns), y=list(df.index),
        colorscale=[[0, "#F2E8D5"], [1, INK]],
        text=np.round(df.values, 1), texttemplate="%{text}",
        hovertemplate="%{y} · %{x}<br>적정가 %{z:,.1f}<extra></extra>"))
    return _base(fig, "민감도: WACC × 영구성장률 → 주당 적정가치")


def price_chart(hist: pd.Series):
    fig = go.Figure(go.Scatter(x=hist.index, y=hist.values, mode="lines",
                               line=dict(color=INK), name="종가"))
    return _base(fig, "주가 (5년)")

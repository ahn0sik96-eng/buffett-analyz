import pandas as pd

from analysis.roic import compute_roic
from analysis.cashflow import compute_cashflow
from analysis.cyclicality import compute_cyclicality
from tests.test_roic import sample


def _build(rev_pattern):
    df = sample()
    yrs = list(range(2024 - len(rev_pattern) + 1, 2025))
    df = pd.DataFrame(index=yrs)
    df["revenue"] = rev_pattern
    df["ebit"] = [r * 0.28 for r in rev_pattern]
    df["pretax_income"] = [r * 0.27 for r in rev_pattern]
    df["tax_provision"] = [r * 0.27 * 0.21 for r in rev_pattern]
    df["net_income"] = [r * 0.21 for r in rev_pattern]
    df["ocf"] = [r * 0.30 for r in rev_pattern]
    df["capex_out"] = [r * 0.06 for r in rev_pattern]
    df["fcf"] = [r * 0.24 for r in rev_pattern]
    df["total_assets"] = [r * 1.2 for r in rev_pattern]
    df["cash"] = [r * 0.15 for r in rev_pattern]
    df["current_liabilities"] = [r * 0.3 for r in rev_pattern]
    df["current_debt"] = [r * 0.03 for r in rev_pattern]
    df["equity"] = [r * 0.55 for r in rev_pattern]
    df["total_debt"] = [r * 0.20 for r in rev_pattern]
    df.index.name = "fy"
    return df


def test_stable_company_scores_defensive():
    stable = [1000 * 1.05 ** i for i in range(6)]   # 매끄러운 5% 성장
    df = _build(stable)
    rr = compute_roic(df, wacc=0.09)
    cc = compute_cashflow(df)
    res = compute_cyclicality(df, rr, cc, 0.09)
    assert res["score"] >= 7.5
    assert res["summary"]["level"] == "경기 방어적"
    assert res["summary"]["is_proxy"] is True


def test_volatile_company_scores_cyclical():
    volatile = [1000, 1300, 800, 1200, 700, 1250]   # 급등락
    df = _build(volatile)
    rr = compute_roic(df, wacc=0.09)
    cc = compute_cashflow(df)
    res = compute_cyclicality(df, rr, cc, 0.09)
    assert res["score"] < res.get("_max", 10)
    assert res["summary"]["rev_neg"] >= 2


def test_thin_data_not_scored():
    df = _build([1000, 1100])   # 2년 → 미채점
    rr = compute_roic(df, wacc=0.09)
    cc = compute_cashflow(df)
    res = compute_cyclicality(df, rr, cc, 0.09)
    assert res.get("score") is None
    assert res["summary"]["level"] is None

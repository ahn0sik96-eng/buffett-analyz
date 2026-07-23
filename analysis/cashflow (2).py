"""잉여현금흐름 분석 (명세 5.2).

FCF = 영업활동현금흐름 − 자본적지출(절대값), 현금전환율 = FCF / 순이익.
순이익 증가 vs FCF 정체·감소 괴리, 매출채권·재고의 매출 대비 과속 증가를 감지한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.roic import _f, _g


def cagr(s: pd.Series, n: int | None = None):
    """양(+) 양끝값 기준 연복리성장률. 계산 불가 시 None."""
    s = s.dropna()
    if n is not None:
        s = s.iloc[-(n + 1):]
    if len(s) >= 2 and s.iloc[0] > 0 and s.iloc[-1] > 0:
        yrs = len(s) - 1
        return _f((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)
    return None


def _slope(s: pd.Series):
    s = s.dropna()
    if len(s) < 3:
        return None
    scale = max(abs(s.mean()), 1e-9)
    return _f(np.polyfit(np.arange(len(s)), s.values.astype(float), 1)[0] / scale)


def compute_cashflow(annual: pd.DataFrame, ttm: dict | None = None,
                     shares_now: float | None = None) -> dict:
    df = annual
    flags: list[str] = []
    risk_codes: set[str] = set()

    rev, ni = _g(df, "revenue"), _g(df, "net_income")
    ocf, capex = _g(df, "ocf"), _g(df, "capex_out")
    fcf = _g(df, "fcf")
    sbc = _g(df, "sbc_out").fillna(0)

    margin = fcf / rev.where(rev > 0)
    conv = fcf / ni.where(ni > 0)
    sh = _g(df, "diluted_shares").fillna(_g(df, "shares_out"))
    if sh.isna().all() and shares_now:
        sh = pd.Series(shares_now, index=df.index)
        flags.append("연도별 주식수 부재 → 현재 발행주식수로 대체(주당 지표 참고용)")
    fcf_ps = fcf / sh.where(sh > 0)
    fcf_adj = fcf - sbc

    table = pd.DataFrame({
        "revenue": rev, "net_income": ni, "ocf": ocf, "capex_out": capex,
        "fcf": fcf, "fcf_margin": margin, "conversion": conv,
        "sbc_out": sbc, "fcf_adj": fcf_adj, "shares": sh, "fcf_ps": fcf_ps,
    })

    f = fcf.dropna()
    neg = int((f < 0).sum())
    if neg:
        run, best = 0, 0
        for v in f.values:
            run = run + 1 if v < 0 else 0
            best = max(best, run)
        if best >= 2:
            risk_codes.add("FCF_NEG_2Y")

    if len(f) >= 4:
        ni_sl, fcf_sl = _slope(ni), _slope(fcf)
        if ni_sl is not None and fcf_sl is not None and ni_sl > 0.02 and fcf_sl < 0:
            risk_codes.add("NI_FCF_DIVERGE")
            flags.append("순이익은 증가 추세이나 FCF는 감소 추세 — 이익의 질 점검 필요")

    g_rev3 = cagr(rev, 3)
    for col, code, label in (("receivables", "AR_FASTER", "매출채권"),
                             ("inventory", "INV_FASTER", "재고자산")):
        g_x = cagr(_g(df, col), 3)
        if g_rev3 is not None and g_x is not None and g_x > g_rev3 + 0.05:
            risk_codes.add(code)
            flags.append(f"{label} 3년 성장률({g_x:.1%})이 매출({g_rev3:.1%})을 크게 상회 — "
                         f"운전자본 악화 원인 확인 필요")

    shv = sh.dropna()
    share_chg = _f(shv.iloc[-1] / shv.iloc[0] - 1) if len(shv) >= 2 else None
    if share_chg is not None and share_chg > 0.02:
        risk_codes.add("SHARES_RISING")

    sbc_ratio = _f((sbc / fcf.where(fcf > 0)).dropna().mean())
    if sbc_ratio is not None and sbc_ratio > 0.20:
        risk_codes.add("SBC_HEAVY")
        flags.append(f"주식보상이 FCF의 평균 {sbc_ratio:.0%} — 희석 부담 큼(조정 FCF 참조)")

    growth = f.pct_change().dropna() if len(f) >= 3 else pd.Series(dtype=float)

    summary = {
        "years": int(len(f)),
        "fcf_latest": _f(f.iloc[-1]) if len(f) else None,
        "fcf_ttm": _f(ttm.get("fcf")) if ttm else None,
        "margin_latest": _f(margin.dropna().iloc[-1]) if margin.notna().any() else None,
        "margin_avg": _f(margin.dropna().mean()) if margin.notna().any() else None,
        "cagr3": cagr(fcf, 3), "cagr5": cagr(fcf, 5), "cagr_max": cagr(fcf),
        "growth_std": _f(growth.std()) if len(growth) >= 2 else None,
        "neg_count": neg,
        "conv_avg": _f(conv.dropna().mean()) if conv.notna().any() else None,
        "fcf_ps_cagr": cagr(fcf_ps),
        "share_change": share_chg,
        "sbc_ratio": sbc_ratio,
        "rev_cagr3": g_rev3,
    }
    return {"table": table, "summary": summary, "flags": flags,
            "risk_codes": risk_codes}

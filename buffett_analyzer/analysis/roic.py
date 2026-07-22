"""ROIC 분석 (명세 5.1).

ROIC = NOPAT / Invested Capital,  NOPAT = EBIT × (1 − 유효세율)
IC(A) = 총자산 − 현금성자산 − 무이자유동부채
IC(B) = 자기자본 + 이자부부채 − 현금성자산
두 방식 괴리가 크면 경고. 분모 ≤ 0 연도는 N/A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _g(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, dtype=float)


def _f(x):
    """finite float 또는 None."""
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def compute_roic(annual: pd.DataFrame, tax_fallback: float = 0.24,
                 wacc: float | None = None, rf: float | None = None,
                 ttm: dict | None = None, ic_method: str = "auto") -> dict:
    df = annual
    flags: list[str] = []
    risk_codes: set[str] = set()

    ebit = _g(df, "ebit")
    pretax, tax = _g(df, "pretax_income"), _g(df, "tax_provision")
    eff = (tax / pretax).where((pretax > 0) & tax.notna())
    eff = eff.clip(0.0, 0.45)
    if eff.isna().any():
        flags.append(f"일부 연도 유효세율 산출 불가 → 대체 세율 {tax_fallback:.0%} 적용")
    eff = eff.fillna(tax_fallback)
    nopat = ebit * (1 - eff)

    ta, cash = _g(df, "total_assets"), _g(df, "cash").fillna(0)
    cl, cd = _g(df, "current_liabilities"), _g(df, "current_debt")
    if cd.isna().all() and cl.notna().any():
        flags.append("단기차입금 항목 부재 → 무이자유동부채 = 유동부채 전체로 가정(IC(A) 과소 추정 가능)")
    nib_cl = cl - cd.fillna(0)
    ic_a = ta - cash - nib_cl
    ic_b = _g(df, "equity") + _g(df, "total_debt").fillna(0) - cash

    both = pd.concat([ic_a, ic_b], axis=1).dropna()
    if len(both):
        denom = both.mean(axis=1).abs().replace(0, np.nan)
        rel = ((both.iloc[:, 0] - both.iloc[:, 1]).abs() / denom).median()
        if np.isfinite(rel) and rel > 0.25:
            flags.append(f"투하자본 A/B 방식 괴리 중앙값 {rel:.0%} — 리스·소수지분 등 자본구조 확인 필요")

    if ic_method == "B":
        ic, used = ic_b, "B(자기자본+이자부부채−현금)"
    elif ic_method == "A":
        ic, used = ic_a, "A(총자산−현금−무이자유동부채)"
    else:
        a_ok, b_ok = int(ic_a.notna().sum()), int(ic_b.notna().sum())
        ic, used = (ic_a, "A(총자산−현금−무이자유동부채)") if a_ok >= b_ok \
            else (ic_b, "B(자기자본+이자부부채−현금)")

    bad = ic.dropna() <= 0
    if bad.any():
        flags.append("투하자본 ≤ 0 연도 존재 — 해당 연도 ROIC N/A")
    roic = nopat / ic.where(ic > 0)

    table = pd.DataFrame({
        "revenue": _g(df, "revenue"), "ebit": ebit, "eff_tax": eff,
        "nopat": nopat, "ic_a": ic_a, "ic_b": ic_b, "ic": ic, "roic": roic,
    })

    r = roic.dropna()
    n = len(r)

    def mean_last(k):
        return _f(r.iloc[-k:].mean()) if n >= 1 else None

    trend = None
    if n >= 3:
        trend = _f(np.polyfit(np.arange(n), r.values.astype(float), 1)[0])
        last3 = r.iloc[-3:]
        if len(last3) == 3 and last3.iloc[0] > last3.iloc[1] > last3.iloc[2]:
            risk_codes.add("ROIC_3Y_DOWN")

    roic_ttm = None
    if ttm and _f(ttm.get("ebit")) is not None and n:
        ic_l = _f(ic.dropna().iloc[-1])
        eff_l = _f(eff.dropna().iloc[-1]) or tax_fallback
        if ic_l and ic_l > 0:
            roic_ttm = ttm["ebit"] * (1 - eff_l) / ic_l

    mean_all = _f(r.mean()) if n else None
    summary = {
        "years": n,
        "period": f"{int(r.index[0])}–{int(r.index[-1])}" if n else "N/A",
        "latest": _f(r.iloc[-1]) if n else None,
        "ttm": _f(roic_ttm),
        "mean3": mean_last(3) if n >= 3 else None,
        "mean5": mean_last(5) if n >= 5 else None,
        "mean_all": mean_all,
        "median": _f(r.median()) if n else None,
        "std": _f(r.std()) if n >= 2 else None,
        "min": _f(r.min()) if n else None,
        "pct_ge_15": _f((r >= 0.15).mean()) if n else None,
        "pct_ge_20": _f((r >= 0.20).mean()) if n else None,
        "pct_gt_wacc": _f((r > wacc).mean()) if (n and wacc) else None,
        "spread_wacc": (mean_all - wacc) if (mean_all is not None and wacc) else None,
        "spread_rf": (mean_all - rf) if (mean_all is not None and rf) else None,
        "trend": trend,                    # 연간 변화폭(소수). 0.01 = +1%p/년
        "ic_method": used,
    }

    # 분해: ROIC = NOPAT마진 × 투하자본회전율 (최근 유효 연도)
    decomp = None
    valid = table.dropna(subset=["roic", "revenue"])
    if len(valid):
        row = valid.iloc[-1]
        if row["revenue"] > 0 and row["ic"] > 0:
            decomp = {"fy": int(valid.index[-1]),
                      "nopat_margin": _f(row["nopat"] / row["revenue"]),
                      "ic_turnover": _f(row["revenue"] / row["ic"])}

    return {"table": table, "summary": summary, "decomposition": decomp,
            "flags": flags, "risk_codes": risk_codes}

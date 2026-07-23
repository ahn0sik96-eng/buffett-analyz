"""재투자 능력 분석 (명세 5.3).

재투자율(1) = (순설비투자 + 운전자본증가 + 인수합병) / NOPAT
재투자율(2) = 1 − (배당 + 자사주매입) / 순이익            ← 보조식
증분 ROIC(n년) = ΔNOPAT / ΔIC, 분모가 IC의 5% 미만이거나 음수면 '신뢰불가'.
지속가능성장률 = ROIC × 재투자율.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.roic import _f, _g

QUADRANTS = {
    (True, True):   ("① 높은 ROIC + 높은 재투자", "최상급 장기 복리 후보"),
    (True, False):  ("② 높은 ROIC + 낮은 재투자", "우량 현금창출형(성숙기)"),
    (False, True):  ("③ 낮은 ROIC + 높은 재투자", "가치 파괴 가능성 — 경계"),
    (False, False): ("④ 낮은 ROIC + 낮은 재투자", "성장성·자본효율 모두 낮음"),
}


def compute_reinvestment(annual: pd.DataFrame, roic_res: dict,
                         wacc: float | None = None) -> dict:
    df = annual
    flags: list[str] = []
    risk_codes: set[str] = set()

    nopat = roic_res["table"]["nopat"]
    ic = roic_res["table"]["ic"]
    roic_avg = roic_res["summary"].get("mean_all")

    capex = _g(df, "capex_out")
    dep = _g(df, "depreciation").fillna(0)
    acq = _g(df, "acquisitions_out").fillna(0)
    wc = _g(df, "working_capital")
    if wc.isna().all():
        wc = _g(df, "current_assets") - _g(df, "current_liabilities")
    dwc = wc.diff()

    rr1 = (capex - dep + dwc.fillna(0) + acq) / nopat.where(nopat > 0)
    payout = _g(df, "dividends_out").fillna(0) + _g(df, "buybacks_out").fillna(0)
    ni = _g(df, "net_income")
    rr2 = 1 - payout / ni.where(ni > 0)

    if rr1.notna().sum() >= 2:
        rr_series, method = rr1, "① (순설비투자+ΔWC+M&A)/NOPAT"
    elif rr2.notna().sum() >= 2:
        rr_series, method = rr2, "② 1 − 배당·자사주매입/순이익 (보조식)"
        flags.append("재투자율 기본식 산출 불가 → 배당성향 기반 보조식 사용")
    else:
        rr_series, method = pd.Series(dtype=float), None

    rr_avg = _f(rr_series.clip(-1, 2).dropna().mean()) if method else None

    # 증분 ROIC
    inc: dict[int, float | None] = {}
    icd = ic.dropna()
    nd = nopat.reindex(icd.index)
    for n in (1, 3, 5):
        if len(icd) > n and pd.notna(nd.iloc[-1]) and pd.notna(nd.iloc[-1 - n]):
            d_ic = icd.iloc[-1] - icd.iloc[-1 - n]
            d_np = nd.iloc[-1] - nd.iloc[-1 - n]
            if d_ic > 0.05 * abs(icd.iloc[-1]):
                inc[n] = _f(d_np / d_ic)
            else:
                inc[n] = None
                flags.append(f"{n}년 증분 ROIC — 투하자본 변화가 미미/음수여서 신뢰불가")
        else:
            inc[n] = None

    inc_best = next((inc[k] for k in (5, 3, 1) if inc.get(k) is not None), None)
    if inc_best is not None and wacc is not None and inc_best < wacc:
        risk_codes.add("INC_ROIC_LT_WACC")
        flags.append(f"증분 ROIC({inc_best:.1%}) < WACC({wacc:.1%}) — 추가 투자분의 가치 창출 의문")

    sgr = _f(roic_avg * rr_avg) if (roic_avg is not None and rr_avg is not None) else None

    quadrant = None
    if roic_avg is not None and rr_avg is not None:
        quadrant = QUADRANTS[(roic_avg >= 0.15, rr_avg >= 0.30)]

    table = pd.DataFrame({"rr_capex": rr1, "rr_payout": rr2,
                          "capex_out": capex, "depreciation": dep,
                          "d_wc": dwc, "acquisitions_out": acq,
                          "dividends_out": _g(df, "dividends_out"),
                          "buybacks_out": _g(df, "buybacks_out")})

    summary = {"rr_avg": rr_avg, "rr_method": method,
               "inc_roic": inc, "inc_best": inc_best,
               "sgr": sgr, "quadrant": quadrant, "roic_avg": roic_avg}
    return {"table": table, "summary": summary, "flags": flags,
            "risk_codes": risk_codes}

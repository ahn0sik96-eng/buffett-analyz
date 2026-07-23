"""경기 방어력 간이 추정 (배점 10).

정식 4단계는 거시지표(GDP·금리·업황) 상관분석이 필요하지만, 그 전까지도
보유 데이터만으로 '실적 변동성'이라는 대리지표를 계산할 수 있다. 이는 정식
경기민감도 분석이 아니라 **과거 변동성 기반 추정치**이며, 화면에도 그렇게 표기한다.

근거 지표(모두 보유 데이터):
- 매출 성장률의 표준편차(변동성) 및 역성장 연도 수
- FCF 성장률 표준편차 및 FCF 적자 연도 수
- ROIC 최저치가 자본비용을 지켜냈는지(불황 저점 방어력의 대리지표)
"""
from __future__ import annotations

import numpy as np

from analysis.roic import _g, _f


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def compute_cyclicality(annual, roic_res, cf_res, wacc) -> dict:
    rev = _g(annual, "revenue").dropna()
    rev_growth = rev.pct_change().dropna() if len(rev) >= 2 else None
    rev_std = float(rev_growth.std()) if (rev_growth is not None and
                                          len(rev_growth) >= 2) else None
    rev_neg = int((rev_growth < 0).sum()) if rev_growth is not None else None
    rev_worst = float(rev_growth.min()) if (rev_growth is not None and
                                            len(rev_growth)) else None

    cs = cf_res["summary"]
    fcf_std = cs.get("growth_std")
    fcf_neg = cs.get("neg_count")

    rs = roic_res["summary"]
    roic_min = rs.get("min")
    years = rs.get("years", 0)

    # 데이터가 너무 얇으면 미채점(허위 정확성 방지)
    if years < 3 or rev_std is None:
        return {"summary": {"years": years, "rev_std": rev_std,
                            "rev_neg": rev_neg, "fcf_neg": fcf_neg,
                            "rev_worst": rev_worst, "roic_min": roic_min,
                            "level": None, "is_proxy": True},
                "flags": ["경기 방어력: 데이터 부족(3개년 미만)으로 추정 불가"],
                "risk_codes": set()}

    flags, risk = [], set()

    # 매출 변동성 (낮을수록 경기방어적)
    if rev_std < 0.05:
        rev_pts, rev_lv = 4.0, "매우 안정"
    elif rev_std < 0.10:
        rev_pts, rev_lv = 3.0, "안정"
    elif rev_std < 0.20:
        rev_pts, rev_lv = 2.0, "보통"
    else:
        rev_pts, rev_lv = 0.5, "변동 큼"
        flags.append(f"매출 성장률 변동성 높음(σ {rev_std*100:.0f}%p) — 경기민감 가능성")

    # 역성장·FCF 적자 빈도
    down_pts = 3.0
    if rev_neg:
        down_pts -= min(rev_neg, 2) * 0.75
        flags.append(f"매출 역성장 {rev_neg}회 관측")
    if fcf_neg:
        down_pts -= min(fcf_neg, 2) * 0.75
    down_pts = max(down_pts, 0.0)

    # 불황 저점 방어(ROIC 최저치가 WACC 위였나)
    if roic_min is not None and wacc is not None:
        if roic_min >= wacc:
            floor_pts, floor_lv = 3.0, "저점에도 자본비용 상회"
        elif roic_min >= wacc * 0.6:
            floor_pts, floor_lv = 1.5, "저점에서 자본비용 근접"
        else:
            floor_pts, floor_lv = 0.5, "저점에서 자본비용 크게 하회"
            risk.add("CYCLICAL_TROUGH")
    else:
        floor_pts, floor_lv = 1.5, "판정 불가(중립)"

    pts = _clip(rev_pts + down_pts + floor_pts, 0, 10)
    level = ("경기 방어적" if pts >= 7.5 else
             "중립" if pts >= 5 else
             "경기 민감")

    return {
        "score": round(pts, 1),
        "summary": {
            "years": years, "rev_std": _f(rev_std), "rev_neg": rev_neg,
            "rev_worst": _f(rev_worst), "fcf_neg": fcf_neg,
            "roic_min": _f(roic_min), "rev_level": rev_lv,
            "floor_level": floor_lv, "level": level, "is_proxy": True,
        },
        "flags": flags,
        "risk_codes": risk,
    }

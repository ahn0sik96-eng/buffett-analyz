"""WACC 산정.

Ke = rf + β·ERP (β는 0.4~2.5로 제한, 부재 시 1.0)
Kd = 이자비용/이자부부채 (rf ~ rf+6%p 제한, 산출 불가 시 rf+1.5%p 추정)
WACC = E/(E+D)·Ke + D/(E+D)·Kd·(1−t),  하한 5%.
"""
from __future__ import annotations

from config import settings
from analysis.roic import _f


def compute_wacc(market_cap, total_debt, equity_book, beta,
                 interest_expense, rf: float, erp: float, tax: float) -> dict:
    notes: list[str] = []

    b = _f(beta)
    if b is None:
        b, msg = settings.BETA_DEFAULT, "β 부재 → 1.0 가정"
        notes.append(msg)
    elif b < settings.BETA_MIN or b > settings.BETA_MAX:
        b = min(max(b, settings.BETA_MIN), settings.BETA_MAX)
        notes.append(f"β를 {settings.BETA_MIN}~{settings.BETA_MAX} 범위로 제한")
    ke = rf + b * erp

    td = _f(total_debt) or 0.0
    ie = _f(interest_expense)
    if td > 0 and ie:
        kd_raw = abs(ie) / td
        kd = min(max(kd_raw, rf), rf + settings.KD_SPREAD_MAX)
        if kd != kd_raw:
            notes.append(f"내재 부채비용({kd_raw:.1%})을 합리 범위로 제한 → {kd:.1%}")
    else:
        kd = rf + settings.KD_FALLBACK_SPREAD
        if td > 0:
            notes.append("이자비용 부재 → 부채비용 = rf+1.5%p 추정")

    e = _f(market_cap)
    if not e or e <= 0:
        e = max(_f(equity_book) or 1.0, 1.0)
        notes.append("시가총액 부재 → 장부 자기자본으로 가중치 산정")
    we = e / (e + td)
    wd = 1 - we

    wacc = we * ke + wd * kd * (1 - tax)
    if wacc < settings.WACC_FLOOR:
        wacc = settings.WACC_FLOOR
        notes.append(f"WACC 하한 {settings.WACC_FLOOR:.0%} 적용")

    return {"wacc": _f(wacc), "ke": _f(ke), "kd_pre": _f(kd),
            "kd_after": _f(kd * (1 - tax)), "beta_used": _f(b),
            "we": _f(we), "wd": _f(wd), "notes": notes}

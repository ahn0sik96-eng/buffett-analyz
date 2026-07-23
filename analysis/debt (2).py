"""부채 및 재무 안전성 (명세 5.5 + 경고 시스템 10장 일부).

핵심: 순부채/EBITDA, 이자보상배율, FCF 대비 순부채, 유동성 3종,
Altman Z-score, Piotroski F-score, 적색 경고 목록.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.roic import _f, _g


def _ratio(n, d, allow_neg_d: bool = False):
    n, d = _f(n), _f(d)
    if n is None or d is None:
        return None
    if d == 0 or (not allow_neg_d and d < 0):
        return None
    return n / d


def _altman(latest: pd.Series, market_cap: float | None):
    ta = _f(latest.get("total_assets"))
    if not ta or ta <= 0:
        return None
    tl = _f(latest.get("total_liabilities"))
    if tl is None:
        eq = _f(latest.get("equity"))
        tl = ta - eq if eq is not None else None
    ca, cl = _f(latest.get("current_assets")), _f(latest.get("current_liabilities"))
    wc = (ca - cl) if (ca is not None and cl is not None) else None
    re = _f(latest.get("retained_earnings"))
    ebit, rev = _f(latest.get("ebit")), _f(latest.get("revenue"))
    if None in (tl, wc, re, ebit, rev) or not market_cap or tl <= 0:
        return None
    z = (1.2 * wc / ta + 1.4 * re / ta + 3.3 * ebit / ta
         + 0.6 * market_cap / tl + 1.0 * rev / ta)
    zone = "안전" if z > 2.99 else ("회색지대" if z >= 1.81 else "부실 위험")
    return {"z": _f(z), "zone": zone}


def _piotroski(df: pd.DataFrame):
    if len(df) < 2:
        return None
    t, p = df.iloc[-1], df.iloc[-2]

    def r(a, b):
        a, b = _f(a), _f(b)
        return a / b if (a is not None and b not in (None, 0)) else None

    checks: list[tuple[str, bool | None]] = []
    ni, ocf = _f(t.get("net_income")), _f(t.get("ocf"))
    checks.append(("순이익 > 0", ni > 0 if ni is not None else None))
    checks.append(("영업현금흐름 > 0", ocf > 0 if ocf is not None else None))
    roa_t, roa_p = r(t.get("net_income"), t.get("total_assets")), r(p.get("net_income"), p.get("total_assets"))
    checks.append(("ROA 개선", roa_t > roa_p if None not in (roa_t, roa_p) else None))
    checks.append(("OCF > 순이익(발생액 건전)", ocf > ni if None not in (ocf, ni) else None))
    lev_t, lev_p = r(t.get("long_term_debt"), t.get("total_assets")), r(p.get("long_term_debt"), p.get("total_assets"))
    checks.append(("장기부채비율 하락", lev_t < lev_p if None not in (lev_t, lev_p) else None))
    cr_t, cr_p = r(t.get("current_assets"), t.get("current_liabilities")), r(p.get("current_assets"), p.get("current_liabilities"))
    checks.append(("유동비율 개선", cr_t > cr_p if None not in (cr_t, cr_p) else None))
    sh_t = _f(t.get("shares_out")) or _f(t.get("diluted_shares"))
    sh_p = _f(p.get("shares_out")) or _f(p.get("diluted_shares"))
    checks.append(("신주 미발행", sh_t <= sh_p * 1.02 if None not in (sh_t, sh_p) else None))
    gm_t, gm_p = r(t.get("gross_profit"), t.get("revenue")), r(p.get("gross_profit"), p.get("revenue"))
    checks.append(("매출총이익률 개선", gm_t > gm_p if None not in (gm_t, gm_p) else None))
    at_t, at_p = r(t.get("revenue"), t.get("total_assets")), r(p.get("revenue"), p.get("total_assets"))
    checks.append(("자산회전율 개선", at_t > at_p if None not in (at_t, at_p) else None))

    valid = [c for c in checks if c[1] is not None]
    return {"score": sum(1 for _, ok in valid if ok), "valid": len(valid),
            "detail": checks}


def compute_debt(annual: pd.DataFrame, market_cap: float | None = None) -> dict:
    df = annual
    warnings: list[str] = []
    risk_codes: set[str] = set()
    latest = df.iloc[-1] if len(df) else pd.Series(dtype=float)

    td = _f(latest.get("total_debt")) or 0.0
    cash = _f(latest.get("cash")) or 0.0
    net_debt = td - cash
    ta = _f(latest.get("total_assets"))
    debt_free = bool(ta and td < 0.02 * ta)

    ebitda = _f(latest.get("ebitda"))
    nd_ebitda = _ratio(net_debt, ebitda) if (ebitda and ebitda > 0) else None
    de = _ratio(td, latest.get("equity"))
    interest = _f(latest.get("interest_expense"))
    icov = None
    if interest and abs(interest) > 1e-9:
        icov = _ratio(latest.get("ebit"), abs(interest))
    fcf_l = _f(latest.get("fcf"))
    nd_fcf = _ratio(net_debt, fcf_l) if (fcf_l and fcf_l > 0 and net_debt > 0) else None
    ocf_l = _f(latest.get("ocf"))
    td_ocf = _ratio(td, ocf_l) if (ocf_l and ocf_l > 0) else None

    ca, cl = _f(latest.get("current_assets")), _f(latest.get("current_liabilities"))
    inv = _f(latest.get("inventory")) or 0.0
    cur_ratio = _ratio(ca, cl)
    quick = _ratio((ca - inv) if ca is not None else None, cl)
    cash_ratio = _ratio(cash, cl)
    cd = _f(latest.get("current_debt")) or 0.0
    short_share = _ratio(cd, td) if td > 0 else None

    # ── 적색 경고 (명세 10장 계산 가능 항목) ──
    if nd_ebitda is not None and nd_ebitda > 3:
        warnings.append(f"순부채/EBITDA {nd_ebitda:.1f}배 > 3배")
    if icov is not None and icov < 3:
        warnings.append(f"이자보상배율 {icov:.1f}배 < 3배")
        risk_codes.add("ICOV_LOW")
    td_series = _g(df, "total_debt").dropna()
    debt_rising = len(td_series) >= 2 and td_series.iloc[-1] > td_series.iloc[-2] * 1.02
    if fcf_l is not None and fcf_l < 0 and debt_rising:
        warnings.append("FCF 적자 상태에서 부채 증가")
    if cd > cash:
        warnings.append("단기부채가 현금성자산 초과 — 차환 위험 점검")
    div_bb = (_f(latest.get("dividends_out")) or 0) + (_f(latest.get("buybacks_out")) or 0)
    if ocf_l is not None and div_bb > ocf_l > 0:
        warnings.append("배당+자사주매입이 영업현금흐름 초과")
        risk_codes.add("PAYOUT_GT_OCF")
    bb = _f(latest.get("buybacks_out")) or 0
    if bb > 0 and debt_rising and fcf_l is not None and fcf_l < bb:
        warnings.append("자사주매입 규모가 FCF를 초과하며 부채 증가 — 부채 조달 매입 의심")
        risk_codes.add("BUYBACK_DEBT")
    gw = _f(latest.get("goodwill_intangibles"))
    eq_l = _f(latest.get("equity"))
    if gw and eq_l and eq_l > 0 and gw / eq_l > 0.5:
        warnings.append(f"영업권·무형자산이 자기자본의 {gw/eq_l:.0%} — 손상 위험 주시")
        risk_codes.add("GOODWILL_HEAVY")

    return {
        "latest": {
            "total_debt": td, "cash": cash, "net_debt": _f(net_debt),
            "nd_ebitda": nd_ebitda, "de": de, "icov": icov,
            "nd_fcf": nd_fcf, "td_ocf": td_ocf,
            "cur_ratio": cur_ratio, "quick": quick, "cash_ratio": cash_ratio,
            "short_share": short_share, "debt_free": debt_free,
        },
        "net_debt_series": (_g(df, "total_debt").fillna(0) - _g(df, "cash").fillna(0)),
        "altman": _altman(latest, market_cap),
        "piotroski": _piotroski(df),
        "warnings": warnings,
        "risk_codes": risk_codes,
    }

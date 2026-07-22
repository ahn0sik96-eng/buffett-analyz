"""상대가치 지표 (명세 5.7 중 MVP 범위: 현재 배수 + 무위험수익률 스프레드).

역사적 밴드·경쟁사 비교는 4단계(경쟁사 데이터), 10개년 배수는 5단계에서 확장.
"""
from __future__ import annotations

import numpy as np

from analysis.roic import _f


def _pos(x):
    x = _f(x)
    return x if (x is not None and x > 0) else None


def compute_multiples(fd, annual, ttm: dict | None, net_debt: float | None,
                      rf: float) -> dict:
    latest = annual.iloc[-1] if len(annual) else {}

    def pick(key):
        v = _f(ttm.get(key)) if ttm else None
        if v is None:
            v = _f(latest.get(key) if hasattr(latest, "get") else None)
            return v, "연간"
        return v, "TTM"

    ni, ni_src = pick("net_income")
    rev, _ = pick("revenue")
    ebitda, _ = pick("ebitda")
    ebit, _ = pick("ebit")
    fcf, fcf_src = pick("fcf")

    price, mcap, shares = _f(fd.price), _f(fd.market_cap), _f(fd.shares)
    eq = _f(latest.get("equity") if hasattr(latest, "get") else None)
    ev = (mcap + net_debt) if (mcap is not None and net_debt is not None) else None

    eps = ni / shares if (ni is not None and _pos(shares)) else None
    per = price / eps if (price and _pos(eps)) else None
    if per is None:
        per = _pos(fd.trailing_pe)
    fwd_pe = _pos(fd.forward_pe)
    pbr = mcap / eq if (mcap and _pos(eq)) else None
    psr = mcap / rev if (mcap and _pos(rev)) else None
    ev_ebitda = ev / ebitda if (ev is not None and _pos(ebitda)) else None
    ev_ebit = ev / ebit if (ev is not None and _pos(ebit)) else None
    p_fcf = mcap / fcf if (mcap and _pos(fcf)) else None

    fcf_yield = fcf / mcap if (_pos(mcap) and fcf is not None) else None
    earn_yield = (1 / per) if _pos(per) else (
        ni / mcap if (ni is not None and _pos(mcap)) else None)

    div_bb = None
    if hasattr(latest, "get"):
        d = _f(latest.get("dividends_out")) or 0
        b = _f(latest.get("buybacks_out")) or 0
        div_bb = d + b
    sh_yield = div_bb / mcap if (div_bb is not None and _pos(mcap)) else None

    return {
        "basis": {"수익성 지표 기준": ni_src, "FCF 기준": fcf_src},
        "per": _f(per), "forward_pe": fwd_pe, "pbr": _f(pbr), "psr": _f(psr),
        "ev_ebitda": _f(ev_ebitda), "ev_ebit": _f(ev_ebit), "p_fcf": _f(p_fcf),
        "fcf_yield": _f(fcf_yield), "earn_yield": _f(earn_yield),
        "shareholder_yield": _f(sh_yield),
        "peg": None,  # 성장률 추정(4단계) 전까지 N/A — 허위 정확성 방지
        "spread_fcf_rf": _f(fcf_yield - rf) if fcf_yield is not None else None,
        "spread_earn_rf": _f(earn_yield - rf) if earn_yield is not None else None,
        "ev": _f(ev),
    }

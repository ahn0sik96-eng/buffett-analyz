"""DCF·역산 DCF·민감도 (명세 6장) 및 매수가격 구간 (12장).

단순 2단계 모형: 기준 FCF를 g1로 5년 성장 → 영구성장률 gT로 종가치.
WACC − gT 최소 간격 미달 시 gT를 자동 하향하고 표기한다.
기준 FCF ≤ 0이면 DCF는 N/A(임의값 대입 금지 — 명세 18·21).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from config.scoring_rules import PRICE_ZONES
from analysis.roic import _f


def dcf_value(fcf0: float, wacc: float, g1: float, gT: float,
              years: int = settings.DCF_YEARS) -> dict | None:
    fcf0 = _f(fcf0)
    if fcf0 is None or fcf0 <= 0:
        return None
    note = None
    if wacc - gT < settings.TERMINAL_GAP_MIN:
        gT = wacc - settings.TERMINAL_GAP_MIN
        note = f"영구성장률을 WACC−{settings.TERMINAL_GAP_MIN:.1%}로 조정({gT:.2%})"
    pv, f = 0.0, fcf0
    for t in range(1, years + 1):
        f *= (1 + g1)
        pv += f / (1 + wacc) ** t
    tv = f * (1 + gT) / (wacc - gT)
    pv_tv = tv / (1 + wacc) ** years
    return {"ev": pv + pv_tv, "pv_stage": pv, "pv_tv": pv_tv, "tv": tv,
            "gT_used": gT, "note": note}


def per_share(ev: float | None, net_debt: float | None,
              shares: float | None) -> tuple[float | None, str | None]:
    if ev is None:
        return None, "기준 FCF ≤ 0 — DCF 산출 불가"
    sh = _f(shares)
    if not sh or sh <= 0:
        return None, "발행주식수 부재"
    eq = ev - (_f(net_debt) or 0.0)
    if eq <= 0:
        return None, "순부채가 기업가치 초과 — 주당가치 산출 불가"
    return eq / sh, None


def run_scenarios(fcf0, wacc, g_base, gT_base, net_debt, shares,
                  price) -> dict:
    dg, dw = settings.SCENARIO_G_DELTA, settings.SCENARIO_WACC_DELTA
    defs = {
        "보수적": (max(g_base - dg, -0.02), wacc + dw, max(gT_base - 0.005, 0.005)),
        "기준":   (g_base, wacc, gT_base),
        "낙관적": (min(g_base + dg, 0.20), max(wacc - dw / 2, settings.WACC_FLOOR),
                   gT_base + 0.005),
    }
    out = {}
    for name, (g1, w, gT) in defs.items():
        res = dcf_value(fcf0, w, g1, gT)
        ps, err = per_share(res["ev"] if res else None, net_debt, shares)
        out[name] = {
            "g1": g1, "wacc": w,
            "gT": res["gT_used"] if res else gT, "note": res["note"] if res else err,
            "tv_share": _f(res["pv_tv"] / res["ev"]) if (res and res["ev"]) else None,
            "fair": _f(ps),
            "upside": _f(ps / price - 1) if (ps and price) else None,
            "mos": _f((ps - price) / ps) if (ps and price) else None,
        }
    return out


def sensitivity(fcf0, wacc, g1, net_debt, shares) -> pd.DataFrame | None:
    if _f(fcf0) is None or fcf0 <= 0 or not shares:
        return None
    w_offsets = [-0.015, -0.0075, 0.0, 0.0075, 0.015]
    gts = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
    rows = {}
    for dw in w_offsets:
        w = max(wacc + dw, settings.WACC_FLOOR)
        row = {}
        for gt in gts:
            if w - gt < 0.010:
                row[f"g={gt:.1%}"] = np.nan
                continue
            res = dcf_value(fcf0, w, g1, gt)
            ps, _ = per_share(res["ev"] if res else None, net_debt, shares)
            row[f"g={gt:.1%}"] = ps if ps else np.nan
        rows[f"WACC {w:.2%}"] = row
    return pd.DataFrame(rows).T


def sanity_filter(scen: dict, price: float | None, fx_adjusted: bool) -> tuple[dict, str | None]:
    """ADR 등 환산비율 불일치로 인한 허위 정확성 방지.

    재무제표 통화를 환산한 종목(fx_adjusted)에서 기준 시나리오 적정가치가
    현재가 대비 4배 이상 벗어나면 — 야후 발행주식수가 ADR 환산 기준인지
    확인이 불가하므로 — 임의로 결과를 표시하지 않고 N/A 처리한다.
    """
    if not fx_adjusted or not price:
        return scen, None
    base = scen.get("기준", {}).get("fair")
    if base is None or base <= 0:
        return scen, None
    ratio = base / price
    if 0.25 <= ratio <= 4.0:
        return scen, None
    for sc in scen.values():
        sc["fair"], sc["upside"], sc["mos"] = None, None, None
        sc["note"] = "ADR 환산비율 불일치 의심으로 N/A 처리"
    msg = (f"기준 시나리오 적정가치가 현재가의 {ratio:.1f}배로 산출되어 신뢰할 수 없습니다. "
          f"ADR 종목은 발행주식수 환산비율(예: TSMC 1 ADR=보통주 5주)을 확인할 수 없어 "
          f"DCF 주당 적정가치를 표시하지 않습니다. 상대가치 배수(PER·EV/EBITDA 등)로 "
          f"판단하거나 원주 기준 데이터를 별도 확인하세요.")
    return scen, msg


def reverse_dcf(price, wacc, gT, net_debt, shares, fcf0) -> dict:
    """현재 주가를 정당화하는 1단계 성장률(g1)을 이분법으로 역산."""
    if _f(fcf0) is None or fcf0 <= 0 or not price or not shares:
        return {"implied_g": None, "msg": "기준 FCF 또는 주가 정보 부족"}

    def val(g):
        res = dcf_value(fcf0, wacc, g, gT)
        ps, _ = per_share(res["ev"] if res else None, net_debt, shares)
        return ps if ps is not None else -1e18   # 순부채 초과 구간 = 하방 취급

    lo, hi = -0.50, 0.80
    if val(hi) < price:
        return {"implied_g": None,
                "msg": "성장률 80%로도 현 주가 정당화 불가 — 시장 기대가 모형 범위를 초과"}
    if val(lo) > price:
        return {"implied_g": lo,
                "msg": "연 −50% 역성장 가정에서도 현 주가가 정당화됨(극단적 저평가 신호)"}
    for _ in range(80):
        mid = (lo + hi) / 2
        if val(mid) < price:
            lo = mid
        else:
            hi = mid
    return {"implied_g": _f((lo + hi) / 2), "msg": None}


def price_zones(fair: float, price: float | None) -> pd.DataFrame:
    rows = []
    for name, lo, hi in PRICE_ZONES:
        lo_p, hi_p = fair * lo, fair * hi
        cur = ""
        if price is not None and lo_p <= price < hi_p:
            cur = "◀ 현재"
        rows.append({"구간": name,
                     "가격대": f"{lo_p:,.0f} ~ {hi_p:,.0f}" if hi < 90 else f"{lo_p:,.0f} ~",
                     "현재": cur})
    return pd.DataFrame(rows)

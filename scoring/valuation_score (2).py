"""밸류에이션 채점 (배점 15) — 기준 시나리오 적정가치 대비 주가 우선,
DCF 불가 시 FCF 수익률 스프레드 → PER 순으로 대체."""
from __future__ import annotations


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def score_valuation(price, fair_base, fcf_yield, rf, per):
    d = []
    pts = None
    if price and fair_base:
        ratio = price / fair_base
        pts = 15 if ratio <= .7 else 13 if ratio <= .8 else 11 if ratio <= .9 \
            else 9 if ratio <= 1.0 else 7 if ratio <= 1.1 else 4 if ratio <= 1.3 else 2
        d.append(f"주가/적정가치(기준 시나리오) {ratio:.2f} → {pts}점")
    elif fcf_yield is not None:
        sp = fcf_yield - rf
        pts = 12 if sp >= .02 else 9 if sp >= 0 else 6 if sp >= -.02 else 3
        d.append(f"DCF 불가 → FCF수익률−rf 스프레드 {sp*100:+.1f}%p 기준 {pts}점")
    elif per:
        pts = 8 if per <= 15 else 5 if per <= 25 else 3
        d.append(f"DCF·FCF수익률 불가 → PER {per:.1f}배 기준 {pts}점")
    if pts is None:
        return None, 15, ["밸류에이션 산출 불가 — 미채점"]
    if fcf_yield is not None and price and fair_base:
        sp = fcf_yield - rf
        adj = 1 if sp >= .01 else (-1 if sp <= -.02 else 0)
        if adj:
            pts += adj
            d.append(f"FCF수익률−rf {sp*100:+.1f}%p 보정 → {adj:+d}")
    return _clip(pts, 0, 15), 15, d

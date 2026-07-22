"""기업의 질 채점 (배점: ROIC 20 · FCF 15 · 재투자 15 · 부채 10).

각 함수는 (점수|None, 만점, 근거목록)을 반환. 데이터 부족 시 None(미채점) —
임의 점수 부여 금지(명세 21).
"""
from __future__ import annotations


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def score_roic(roic_res: dict, wacc: float | None):
    s = roic_res["summary"]
    avg = s.get("mean_all")
    if avg is None:
        return None, 20, ["ROIC 산출 불가 — 미채점"]
    d = []
    base = 15 if avg >= .20 else 12 if avg >= .15 else 8 if avg >= .10 else 4 if avg >= .05 else 1
    pts = base
    d.append(f"평균 ROIC {avg:.1%} → 기본 {base}점")
    p15 = s.get("pct_ge_15") or 0
    add = 2 if p15 >= .8 else 1 if p15 >= .5 else 0
    if add:
        pts += add
        d.append(f"15% 상회 연도 비율 {p15:.0%} → +{add}")
    sp = s.get("spread_wacc")
    if sp is not None:
        a2 = 2 if sp >= .10 else 1 if sp >= .05 else 0 if sp > 0 else -2
        pts += a2
        d.append(f"ROIC−WACC 스프레드 {sp*100:+.1f}%p → {a2:+d}")
    tr = s.get("trend")
    if tr is not None:
        a3 = 1 if tr >= 0 else (-1 if tr <= -0.015 else 0)
        if a3:
            pts += a3
            d.append(f"추세 연 {tr*100:+.1f}%p → {a3:+d}")
    return _clip(pts, 0, 20), 20, d


def score_fcf(cf_res: dict):
    s = cf_res["summary"]
    if s.get("fcf_latest") is None and s.get("margin_avg") is None:
        return None, 15, ["FCF 산출 불가 — 미채점"]
    d, pts = [], 0
    m = s.get("margin_avg")
    if m is not None:
        a = 5 if m >= .20 else 4 if m >= .10 else 2.5 if m >= .05 else 1 if m > 0 else 0
        pts += a
        d.append(f"평균 FCF 마진 {m:.1%} → +{a}")
    g = s.get("cagr_max")
    if g is not None:
        a = 4 if g >= .12 else 3 if g >= .07 else 2 if g >= .02 else 1 if g >= 0 else 0
        pts += a
        d.append(f"FCF CAGR {g:.1%} → +{a}")
    else:
        d.append("FCF CAGR 산출 불가(음수·기간 부족) → 성장 가점 0")
    c = s.get("conv_avg")
    if c is not None:
        a = 3 if c >= 1.0 else 2.5 if c >= .8 else 1.5 if c >= .6 else 0.5 if c > 0 else 0
        pts += a
        d.append(f"현금전환율 평균 {c:.0%} → +{a}")
    neg = s.get("neg_count", 0)
    a = 2 if neg == 0 else 1 if neg == 1 else 0
    pts += a
    d.append(f"FCF 적자 {neg}회 → +{a}")
    ni_div = "NI_FCF_DIVERGE" in cf_res.get("risk_codes", set())
    if ni_div:
        pts -= 1
        d.append("순이익·FCF 괴리 → −1")
    return _clip(pts, 0, 15), 15, d


def score_reinvestment(re_res: dict, wacc: float | None):
    s = re_res["summary"]
    roic_avg, rr = s.get("roic_avg"), s.get("rr_avg")
    if roic_avg is None or rr is None:
        return None, 15, ["재투자율 산출 불가 — 미채점"]
    d = []
    hi_r, hi_rr = roic_avg >= .15, rr >= .30
    base = 13 if (hi_r and hi_rr) else 10 if (hi_r and not hi_rr) \
        else 3 if (not hi_r and hi_rr) else 5
    pts = base
    d.append(f"{s['quadrant'][0]} → 기본 {base}점")
    inc = s.get("inc_best")
    if inc is not None and wacc is not None:
        a = 2 if inc >= wacc + .05 else 1 if inc >= wacc else -3
        pts += a
        d.append(f"증분 ROIC {inc:.1%} vs WACC {wacc:.1%} → {a:+d}")
    else:
        d.append("증분 ROIC 신뢰불가 — 4사분면 판정만 반영")
    sgr = s.get("sgr")
    if sgr is not None and sgr >= .08:
        pts += 1
        d.append(f"지속가능성장률 {sgr:.1%} → +1")
    return _clip(pts, 0, 15), 15, d


def score_debt(debt_res: dict):
    L = debt_res["latest"]
    d, pts = [], 0.0
    nd_e = L.get("nd_ebitda")
    nd = L.get("net_debt")
    if nd is not None and nd < 0:
        pts += 3.5
        d.append("순현금 상태 → +3.5")
    elif nd_e is not None:
        a = 3 if nd_e < 1 else 2.5 if nd_e < 2 else 1.5 if nd_e < 3 else 0
        pts += a
        d.append(f"순부채/EBITDA {nd_e:.1f}배 → +{a}")
    else:
        d.append("순부채/EBITDA 산출 불가(EBITDA≤0 등) → 0")
    icov = L.get("icov")
    if L.get("debt_free"):
        pts += 3
        d.append("사실상 무차입(부채<자산의 2%) → +3")
    elif icov is not None:
        a = 3 if icov >= 10 else 2.5 if icov >= 5 else 1.5 if icov >= 3 else 0
        pts += a
        d.append(f"이자보상배율 {icov:.1f}배 → +{a}")
    else:
        pts += 1.5
        d.append("이자비용 미미/부재 → +1.5(중립)")
    cr = L.get("cur_ratio")
    if cr is not None:
        a = 1.5 if cr >= 1.5 else 1 if cr >= 1 else 0
        pts += a
        d.append(f"유동비율 {cr:.2f} → +{a}")
    alt = debt_res.get("altman")
    if alt:
        a = 2 if alt["zone"] == "안전" else 1 if alt["zone"] == "회색지대" else 0
        pts += a
        d.append(f"Altman Z {alt['z']:.2f}({alt['zone']}) → +{a}")
    else:
        pts += 1
        d.append("Altman Z 산출 불가 → +1(중립)")
    return _clip(pts, 0, 10), 10, d

"""서술형 해석 생성 (명세 20·22장 + 화면 10).

원칙(명세 21): 계산 결과와 프로그램의 추론을 구분해 표기하고,
데이터가 부족한 항목은 단정하지 않는다.
"""
from __future__ import annotations

from ui.formatting import pct, xs


def _lv(v, cuts, labels):
    for c, l in zip(cuts, labels):
        if v >= c:
            return l
    return labels[-1]


def roic_text(rs: dict, wacc: float | None) -> str:
    s = rs["summary"]
    if s.get("mean_all") is None:
        return "ROIC를 산출할 수 없습니다(핵심 항목 누락). — 데이터 부족"
    lvl = _lv(s["mean_all"], [.20, .15, .10, .05],
              ["매우 우수", "우수", "보통", "낮음", "매우 낮음"])
    t = [f"분석기간({s['period']}, {s['years']}개년) 평균 ROIC는 {pct(s['mean_all'])}로 "
         f"**{lvl}** 수준입니다."]
    if s.get("pct_ge_15") is not None:
        k = round(s["pct_ge_15"] * s["years"])
        t.append(f"{s['years']}개 연도 중 {k}개 연도에서 15%를 상회했습니다.")
    if s.get("trend") is not None:
        d = "상승" if s["trend"] > 0.002 else "하락" if s["trend"] < -0.002 else "횡보"
        t.append(f"추세는 연 {s['trend']*100:+.1f}%p로 {d}입니다.")
    if s.get("spread_wacc") is not None and wacc:
        ok = "초과해 가치를 창출" if s["spread_wacc"] > 0 else "하회해 가치 창출이 의문시"
        t.append(f"WACC({pct(wacc)}) 대비 스프레드는 {s['spread_wacc']*100:+.1f}%p로 "
                 f"자본비용을 {ok}됩니다.")
    if s["years"] < 10:
        t.append(f"*불확실성: {s['years']}개년 데이터로 10년 지속성 판단에는 한계가 있습니다.*")
    return " ".join(t)


def fcf_text(cs: dict) -> str:
    s = cs["summary"]
    if s.get("fcf_latest") is None:
        return "FCF를 산출할 수 없습니다. — 데이터 부족"
    t = []
    if s.get("margin_avg") is not None:
        t.append(f"평균 FCF 마진 {pct(s['margin_avg'])}")
    if s.get("cagr_max") is not None:
        t.append(f"FCF CAGR {pct(s['cagr_max'])}")
    if s.get("conv_avg") is not None:
        t.append(f"현금전환율 평균 {pct(s['conv_avg'], 0)}")
    head = " · ".join(t) + "." if t else ""
    tail = []
    if s.get("neg_count"):
        tail.append(f"FCF 적자 {s['neg_count']}회 발생.")
    if s.get("share_change") is not None:
        d = "감소(환원 우호적)" if s["share_change"] < 0 else "증가(희석 주의)"
        tail.append(f"기간 중 주식수 {pct(abs(s['share_change']))} {d}.")
    if s.get("sbc_ratio") is not None and s["sbc_ratio"] > 0.1:
        tail.append(f"주식보상은 FCF의 평균 {pct(s['sbc_ratio'],0)} — 조정 FCF 병행 확인 권장.")
    return head + (" " + " ".join(tail) if tail else "")


def reinvest_text(rs: dict) -> str:
    s = rs["summary"]
    if s.get("rr_avg") is None:
        return "재투자율을 산출할 수 없습니다. — 데이터 부족"
    q = s["quadrant"]
    t = [f"평균 재투자율 {pct(s['rr_avg'],0)}({s['rr_method']}), "
         f"판정: **{q[0]}** — {q[1]}."]
    if s.get("inc_best") is not None:
        t.append(f"최근 구간 증분 ROIC는 {pct(s['inc_best'])}입니다(계산 결과).")
    else:
        t.append("증분 ROIC는 분모 불안정으로 신뢰불가 처리했습니다.")
    if s.get("sgr") is not None:
        t.append(f"지속가능성장률(ROIC×재투자율)은 {pct(s['sgr'])}로 추정됩니다 — 프로그램의 추론.")
    return " ".join(t)


def debt_text(ds: dict) -> str:
    L = ds["latest"]
    t = []
    if L.get("net_debt") is not None:
        state = "순현금" if L["net_debt"] < 0 else "순부채"
        t.append(f"{state} 상태이며")
    if L.get("nd_ebitda") is not None:
        t.append(f"순부채/EBITDA {xs(L['nd_ebitda'])},")
    if L.get("debt_free"):
        t.append("사실상 무차입 구조입니다.")
    elif L.get("icov") is not None:
        t.append(f"이자보상배율 {xs(L['icov'])}입니다.")
    alt = ds.get("altman")
    if alt:
        t.append(f"Altman Z-score {alt['z']:.2f}({alt['zone']}).")
    pio = ds.get("piotroski")
    if pio:
        t.append(f"Piotroski F-score {pio['score']}/{pio['valid']}(유효 항목 기준).")
    if ds.get("warnings"):
        t.append("⚠ 경고 " + str(len(ds["warnings"])) + "건 — 재무 안전성 탭 참조.")
    return " ".join(t) if t else "부채 지표 산출 불가 — 데이터 부족"


def conclusion(fd, scores: dict, cls: tuple[str, str], dcf_scen: dict | None,
               reverse: dict | None, mult: dict, roic_res, cf_res, re_res,
               debt_res, mos_target: float) -> dict:
    """화면 10: 강점/약점/리스크/확인지표/훼손조건 + 핵심 판단문."""
    rs, cs, res, L = roic_res["summary"], cf_res["summary"], re_res["summary"], debt_res["latest"]

    strengths: list[str] = []
    if rs.get("mean_all") is not None and rs["mean_all"] >= .15:
        strengths.append(f"높은 자본수익률 — 평균 ROIC {pct(rs['mean_all'])}")
    if rs.get("spread_wacc") is not None and rs["spread_wacc"] >= .05:
        strengths.append(f"ROIC−WACC 스프레드 {rs['spread_wacc']*100:+.1f}%p")
    if cs.get("margin_avg") is not None and cs["margin_avg"] >= .15:
        strengths.append(f"강한 현금창출력 — FCF 마진 {pct(cs['margin_avg'])}")
    if cs.get("conv_avg") is not None and cs["conv_avg"] >= 1.0:
        strengths.append(f"이익의 질 양호 — 현금전환율 {pct(cs['conv_avg'],0)}")
    if L.get("net_debt") is not None and L["net_debt"] < 0:
        strengths.append("순현금 재무구조")
    if cs.get("share_change") is not None and cs["share_change"] < -0.02:
        strengths.append("주식수 감소(자사주 소각형 환원)")
    if res.get("quadrant") and res["quadrant"][0].startswith("①"):
        strengths.append("높은 ROIC를 유지하며 재투자 중(복리 구조)")

    weaknesses: list[str] = []
    if rs.get("trend") is not None and rs["trend"] < -0.005:
        weaknesses.append(f"ROIC 하락 추세(연 {rs['trend']*100:+.1f}%p)")
    if cs.get("cagr_max") is None:
        weaknesses.append("FCF 성장률 산출 불가(적자·기간 부족)")
    elif cs["cagr_max"] < 0.02:
        weaknesses.append(f"FCF 성장 정체(CAGR {pct(cs['cagr_max'])})")
    if cs.get("sbc_ratio") is not None and cs["sbc_ratio"] > .2:
        weaknesses.append("주식보상 비중 과다")
    if res.get("rr_avg") is not None and res["rr_avg"] < .1 and \
            (rs.get("mean_all") or 0) >= .15:
        weaknesses.append("재투자 기회 제한(성숙 국면 가능성)")
    if L.get("nd_ebitda") is not None and L["nd_ebitda"] > 2:
        weaknesses.append(f"레버리지 부담(순부채/EBITDA {xs(L['nd_ebitda'])})")
    if scores.get("val_norm") is not None and scores["val_norm"] < 50:
        weaknesses.append("현재 가격의 안전마진 부족")

    risks: list[str] = list(debt_res.get("warnings", []))[:3]
    risks += [f for f in cf_res.get("flags", []) if "괴리" in f or "상회" in f][:2]
    if reverse and reverse.get("implied_g") is not None:
        risks.append(f"시장 내재 성장 기대: 5년 FCF 연 {pct(reverse['implied_g'])} — "
                     f"미달 시 디레이팅 위험")

    checkpoints = ["분기 매출·영업이익률의 전년 대비 방향",
                   "영업현금흐름과 순이익의 괴리 여부",
                   "매출채권·재고 증가율 vs 매출 증가율"]
    if L.get("nd_ebitda") is not None and L["nd_ebitda"] > 1.5:
        checkpoints.append("차입금 만기 구조와 평균 조달금리")
    if res.get("inc_best") is None:
        checkpoints.append("CAPEX·M&A 집행분의 수익화 진척(증분 ROIC 재계산)")

    breakers = []
    if rs.get("mean_all") is not None:
        breakers.append(f"ROIC가 WACC 아래로 하락해 2개 연도 이상 지속")
    breakers.append("FCF가 2개 회계연도 연속 적자")
    breakers.append("순부채/EBITDA 3배 초과 진입")
    if reverse and reverse.get("implied_g") is not None:
        breakers.append(f"실적 성장률이 내재 기대치({pct(reverse['implied_g'])})를 "
                        f"연속 하회")

    # 핵심 판단문 (명세 화면 2 형식) — 프로그램의 추론
    parts = [f"{fd.name}은(는)"]
    parts.append(strengths[0] + ("과 " + strengths[1] if len(strengths) > 1 else "") +
                 "을(를) 보유"
                 if strengths else "정량 강점이 뚜렷하지 않음")
    if weaknesses:
        parts.append(f"하고 있으나, {weaknesses[0]}이(가) 관찰됩니다.")
    else:
        parts.append("하고 있습니다.")
    if scores.get("quality_norm") is not None and scores.get("val_norm") is not None:
        parts.append(f"기업의 질 환산 {scores['quality_norm']:.0f}점, "
                     f"밸류에이션 환산 {scores['val_norm']:.0f}점 — {cls[0]}.")
    thesis = " ".join(parts)

    return {"strengths": strengths[:5] or ["해당 없음(정량 기준 미충족)"],
            "weaknesses": weaknesses[:5] or ["뚜렷한 정량 약점 미검출"],
            "risks": risks[:5] or ["적색 경고 미검출"],
            "checkpoints": checkpoints[:5],
            "breakers": breakers[:5],
            "thesis": thesis}

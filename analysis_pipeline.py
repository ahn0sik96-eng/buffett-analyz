"""분석 파이프라인 — 단일 종목 뷰와 비교/워치리스트 뷰가 공유하는 계산 진입점.

app.py의 인라인 로직을 함수로 추출해, 여러 종목을 배치로 돌릴 수 있게 한다.
UI(st.*) 호출은 포함하지 않는다 — 순수 계산만.
"""
from __future__ import annotations

import numpy as np

from config import settings
from config.scoring_rules import WEIGHTS, NOT_IMPLEMENTED
from data.data_validator import validate
from analysis.roic import compute_roic, _g
from analysis.cashflow import compute_cashflow
from analysis.reinvestment import compute_reinvestment
from analysis.debt import compute_debt
from analysis.valuation import compute_multiples
from analysis.cyclicality import compute_cyclicality
from models.wacc import compute_wacc
from models import dcf as dcf_m
from models.financial_valuation import compute_financial_valuation
from scoring.quality_score import (score_roic, score_fcf,
                                   score_reinvestment, score_debt)
from scoring.valuation_score import score_valuation
from scoring.risk_penalties import collect as collect_penalties
from scoring.final_score import aggregate, classify
from reports import narrative_report as nr
from ui.formatting import pct


def pick_fcf0(annual, cf_summary, fcf_base_opt):
    """DCF 기준 FCF 선택.

    EDGAR/DART 연동으로 12~20년 이력이 확보되면서 '최근 3년'은 오히려 사이클
    위치에 결과가 좌우되는 선택지가 됐다. 시클리컬 종목은 사이클 정규화
    (FCF마진 중앙값 x 최근 매출)를 써야 정상이익 기준선이 나온다.
    """
    fcf_series = _g(annual, "fcf").dropna()
    rev = _g(annual, "revenue")

    if fcf_base_opt == "TTM" and cf_summary.get("fcf_ttm"):
        return cf_summary["fcf_ttm"], "TTM"
    if fcf_base_opt == "최근 연도" and len(fcf_series):
        return float(fcf_series.iloc[-1]), "최근 연도"

    if fcf_base_opt.startswith("전체기간") and len(fcf_series) >= 5:
        return float(fcf_series.median()), f"전체 {len(fcf_series)}년 중앙값"

    if fcf_base_opt.startswith("사이클 정규화"):
        margin = (fcf_series / rev.reindex(fcf_series.index)) \
            .replace([np.inf, -np.inf], np.nan).dropna()
        rev_recent = rev.dropna()
        if len(margin) >= 5 and len(rev_recent):
            m = float(margin.median())
            base_rev = float(rev_recent.iloc[-1])
            return m * base_rev, (
                f"사이클 정규화 — FCF마진 중앙값 {m:.1%}({len(margin)}년) "
                f"x 최근 매출")
        # 데이터 부족 → 아래 3년 중앙값으로 폴백

    if len(fcf_series):
        return float(fcf_series.iloc[-3:].median()), \
            f"최근 {min(3, len(fcf_series))}년 중앙값"
    return None, "산출 불가"


def analyze(fd, a: dict) -> dict:
    """fd: FinancialData, a: 가정 dict(rf, erp, tax_fb, ic_method, fcf_base_opt,
    g_mode, g_manual, gT, mos_target). 반환: 렌더링에 필요한 전체 결과 dict."""
    rf, erp, tax_fb = a["rf"], a["erp"], a["tax_fb"]
    msgs, data_shortage = validate(fd)
    annual = fd.annual
    cur = fd.fin_currency or fd.currency
    latest = annual.iloc[-1]

    wacc_res = compute_wacc(fd.market_cap, latest.get("total_debt"),
                            latest.get("equity"), fd.beta,
                            latest.get("interest_expense"), rf, erp, tax_fb)
    wacc = wacc_res["wacc"]

    roic_res = compute_roic(annual, tax_fallback=tax_fb, wacc=wacc, rf=rf,
                            ttm=fd.ttm, ic_method=a["ic_method"])
    cf_res = compute_cashflow(annual, ttm=fd.ttm, shares_now=fd.shares)
    re_res = compute_reinvestment(annual, roic_res, wacc=wacc)
    debt_res = compute_debt(annual, market_cap=fd.market_cap)
    net_debt = debt_res["latest"]["net_debt"]
    mult = compute_multiples(fd, annual, fd.ttm, net_debt, rf)
    cyc_res = compute_cyclicality(annual, roic_res, cf_res, wacc)

    fcf0, fcf0_label = pick_fcf0(annual, cf_res["summary"], a["fcf_base_opt"])

    if a["g_mode"].startswith("자동"):
        g_auto = cf_res["summary"].get("cagr5") or cf_res["summary"].get("cagr_max") \
            or cf_res["summary"].get("cagr3")
        g1 = float(np.clip(g_auto, -0.02, 0.15)) if g_auto is not None else 0.05
        g1_label = f"자동 {pct(g1)} (과거 FCF CAGR 기반, 상하한 −2%~15%)"
        g1_caution = (g_auto is not None and g_auto > 0.12)
    else:
        g1, g1_label = a["g_manual"], f"수동 {pct(a['g_manual'])}"
        g1_caution = a["g_manual"] > 0.12

    gT, mos_target = a["gT"], a["mos_target"]

    # ── 금융업: FCF·EV·WACC가 정의되지 않으므로 DCF를 아예 돌리지 않는다 ────
    #    (숫자가 나오되 전부 틀린 값이 되어, 결과를 신뢰하게 만드는 것이 더 위험)
    fin_val = None
    if fd.is_financial:
        fin_val = compute_financial_valuation(fd, annual, wacc_res.get("ke"),
                                              fd.price)
        scen = sens = reverse = None
        fair_base = fin_val["summary"].get("fair")
        fx_sanity_msg = None
        msgs.append({
            "level": "warn",
            "msg": f"금융회사({fin_val['subtype']})로 판별 — FCF 기반 DCF는 "
                   f"예금·대출 증감이 영업활동현금흐름에 그대로 섞여 무의미하므로 "
                   f"실행하지 않았습니다. 적정가치는 초과수익모형"
                   f"(적정 PBR = (ROE−g)/(Ke−g))으로 산출했습니다.",
        })
    else:
        scen = dcf_m.run_scenarios(fcf0, wacc, g1, gT, net_debt, fd.shares,
                                   fd.price) if fcf0 else None
        scen, fx_sanity_msg = dcf_m.sanity_filter(scen, fd.price, fd.fx_adjusted) \
            if scen else (scen, None)
        fair_base = scen["기준"]["fair"] if scen else None
        sens = dcf_m.sensitivity(fcf0, wacc, g1, net_debt, fd.shares) \
            if fcf0 else None
        reverse = dcf_m.reverse_dcf(fd.price, wacc, gT, net_debt, fd.shares,
                                    fcf0) if fcf0 else None
        if fx_sanity_msg:
            reverse, sens = None, None

    # ── 채점 ──
    if fd.is_financial:
        fin_reason = ["금융회사 — 일반 모델 부적합(5단계 전용 모델 예정)"]
        components = {k: (None, WEIGHTS[k], fin_reason)
                      for k in ("roic", "fcf", "reinvestment", "debt",
                                "cyclicality")}
        components["moat"] = (None, WEIGHTS["moat"],
                              [f"미구현({NOT_IMPLEMENTED['moat']})"])
    else:
        components = {
            "roic": score_roic(roic_res, wacc),
            "fcf": score_fcf(cf_res),
            "reinvestment": score_reinvestment(re_res, wacc),
            "debt": score_debt(debt_res),
        }
        # 경기 방어력: 간이 추정치가 있으면 반영, 없으면 미채점
        if cyc_res.get("score") is not None:
            components["cyclicality"] = (
                cyc_res["score"], WEIGHTS["cyclicality"],
                [f"{cyc_res['summary']['level']} (간이 추정, 정식 4단계 아님) — "
                 f"매출변동성 σ {pct(cyc_res['summary']['rev_std'],0)}"])
        else:
            components["cyclicality"] = (None, WEIGHTS["cyclicality"],
                                         cyc_res["flags"][:1] or ["미채점"])
        # 해자는 여전히 정성 데이터 필요 → 미구현
        components["moat"] = (None, WEIGHTS["moat"],
                              [f"미구현({NOT_IMPLEMENTED['moat']})"])

    if fd.is_financial:
        # FCF수익률은 금융업에서 의미가 없으므로 전달하지 않는다.
        # fair_base는 초과수익모형 결과(없으면 None → 밸류에이션 미채점).
        components["valuation"] = score_valuation(fd.price, fair_base, None,
                                                  rf, mult.get("per"))
    else:
        components["valuation"] = score_valuation(fd.price, fair_base,
                                                  mult.get("fcf_yield"), rf,
                                                  mult.get("per"))

    risk_codes = set()
    for r in (roic_res, cf_res, re_res, debt_res, cyc_res):
        risk_codes |= r.get("risk_codes", set())
    penalty_total, penalty_items = collect_penalties(risk_codes, data_shortage)

    scores = aggregate(components, penalty_total)
    z_zone = debt_res["altman"]["zone"] if debt_res.get("altman") else None
    cls = classify(scores["quality_norm"], scores["val_norm"],
                   components.get("debt"), z_zone, fd.is_financial,
                   roic_res["summary"]["years"])
    concl = nr.conclusion(fd, scores, cls, scen, reverse, mult,
                          roic_res, cf_res, re_res, debt_res, mos_target)

    return dict(
        fd=fd, cur=cur, msgs=msgs, data_shortage=data_shortage,
        wacc_res=wacc_res, wacc=wacc, roic_res=roic_res, cf_res=cf_res,
        re_res=re_res, debt_res=debt_res, mult=mult, cyc_res=cyc_res,
        net_debt=net_debt, fcf0=fcf0, fcf0_label=fcf0_label, g1=g1,
        g1_label=g1_label, g1_caution=g1_caution, gT=gT, mos_target=mos_target,
        scen=scen, fx_sanity_msg=fx_sanity_msg, fair_base=fair_base, sens=sens,
        reverse=reverse, components=components, penalty_items=penalty_items,
        scores=scores, cls=cls, concl=concl, fin_val=fin_val,
    )

"""버핏·멍거식 우량기업 분석기 — Streamlit 진입점 (MVP: 명세 23장 1~3단계 + α).

실행:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from config.scoring_rules import WEIGHTS, NOT_IMPLEMENTED
from data import data_fetcher
from data.data_validator import validate
from analysis.roic import compute_roic, _g
from analysis.cashflow import compute_cashflow
from analysis.reinvestment import compute_reinvestment
from analysis.debt import compute_debt
from analysis.valuation import compute_multiples
from models.wacc import compute_wacc
from models import dcf as dcf_m
from scoring.quality_score import (score_roic, score_fcf,
                                   score_reinvestment, score_debt)
from scoring.valuation_score import score_valuation
from scoring.risk_penalties import collect as collect_penalties
from scoring.final_score import aggregate, classify
from reports import narrative_report as nr
from reports.excel_report import build_excel
from ui import charts
from ui.formatting import pct, xs, money, price_fmt

COMP_LABEL = {"roic": "ROIC", "fcf": "FCF", "reinvestment": "재투자",
              "moat": "경제적 해자", "debt": "부채·안전성",
              "cyclicality": "경기 방어력", "valuation": "밸류에이션"}

st.set_page_config(page_title="버핏·멍거 우량기업 분석기", page_icon="📒",
                   layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker: str):
    return data_fetcher.fetch(ticker)


def fmt_table(df: pd.DataFrame, currency, money_cols=(), pct_cols=(),
              x_cols=()) -> pd.DataFrame:
    out = df.copy()
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: money(v, currency))
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: pct(v))
    for c in x_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: xs(v))
    return out


# ── 사이드바: 화면 1(종목 검색·가정) ────────────────────────────────────────
with st.sidebar:
    st.title("📒 종목 검색")
    ticker_in = st.text_input("티커 / 한국 6자리 코드",
                              placeholder="예: AAPL, V, 005930, 삼성전자")
    st.caption("미국: 티커 · 한국: 6자리 코드(.KS/.KQ 자동 판별) 또는 대표 종목명")
    st.divider()
    st.subheader("가정 (수정 가능)")
    country_hint = "KR" if ticker_in and any(
        c.isdigit() for c in ticker_in[:2]) else "US"
    rf = st.number_input("무위험수익률 (10년물, %)",
                         value=settings.DEFAULT_RF.get(country_hint, 0.04) * 100,
                         min_value=0.0, max_value=15.0, step=0.1) / 100
    erp = st.number_input("주식위험프리미엄 (%)", value=settings.DEFAULT_ERP * 100,
                          min_value=1.0, max_value=12.0, step=0.25) / 100
    tax_fb = st.number_input("대체 법인세율 (%)",
                             value=settings.TAX_FALLBACK.get(country_hint, .25) * 100,
                             min_value=0.0, max_value=45.0, step=0.5) / 100
    ic_method = st.selectbox("투하자본 산정", ["auto", "A", "B"],
                             help="A: 총자산−현금−무이자유동부채 / B: 자기자본+이자부부채−현금")
    fcf_base_opt = st.selectbox("DCF 기준 FCF", ["3년 중앙값", "TTM", "최근 연도"])
    g_mode = st.selectbox("1단계 성장률(5년)", ["자동(과거 FCF CAGR 기반)", "수동"])
    g_manual = st.slider("수동 성장률 (%)", -10.0, 25.0, 6.0, 0.5) / 100
    gT = st.number_input("영구성장률 (%)", value=settings.DEFAULT_TERMINAL_G * 100,
                         min_value=0.0, max_value=4.0, step=0.25) / 100
    mos_target = st.slider("목표 안전마진 (%)", 10, 50,
                           int(settings.DEFAULT_MOS_TARGET * 100), 5) / 100
    run = st.button("분석 실행", type="primary", use_container_width=True)

st.title("버핏·멍거식 우량기업 분석기")
st.caption("ROIC · FCF · 재투자 · 재무안전성 · DCF — 장기 복리 적합성 평가 (MVP 1~3단계)")

if not (run and ticker_in):
    st.info("좌측에서 티커를 입력하고 **분석 실행**을 누르세요. "
            "예: `AAPL`, `MSFT`, `V`, `005930`(삼성전자), `000660`(SK하이닉스)")
    st.stop()

try:
    with st.spinner("재무데이터 수집 중…"):
        fd = load(ticker_in.strip())
except Exception as e:
    st.error(str(e))
    st.stop()

msgs, data_shortage = validate(fd)
annual = fd.annual
cur = fd.fin_currency or fd.currency

# ── 파이프라인 ──────────────────────────────────────────────────────────────
latest = annual.iloc[-1]
wacc_res = compute_wacc(fd.market_cap, latest.get("total_debt"),
                        latest.get("equity"), fd.beta,
                        latest.get("interest_expense"), rf, erp, tax_fb)
wacc = wacc_res["wacc"]

roic_res = compute_roic(annual, tax_fallback=tax_fb, wacc=wacc, rf=rf,
                        ttm=fd.ttm, ic_method=ic_method)
cf_res = compute_cashflow(annual, ttm=fd.ttm, shares_now=fd.shares)
re_res = compute_reinvestment(annual, roic_res, wacc=wacc)
debt_res = compute_debt(annual, market_cap=fd.market_cap)
net_debt = debt_res["latest"]["net_debt"]
mult = compute_multiples(fd, annual, fd.ttm, net_debt, rf)

# DCF 기준 FCF
fcf_series = _g(annual, "fcf").dropna()
if fcf_base_opt == "TTM" and cf_res["summary"].get("fcf_ttm"):
    fcf0, fcf0_label = cf_res["summary"]["fcf_ttm"], "TTM"
elif fcf_base_opt == "최근 연도" and len(fcf_series):
    fcf0, fcf0_label = float(fcf_series.iloc[-1]), "최근 연도"
elif len(fcf_series):
    fcf0 = float(fcf_series.iloc[-3:].median())
    fcf0_label = f"최근 {min(3, len(fcf_series))}년 중앙값"
else:
    fcf0, fcf0_label = None, "산출 불가"

if g_mode.startswith("자동"):
    g_auto = cf_res["summary"].get("cagr5") or cf_res["summary"].get("cagr_max") \
        or cf_res["summary"].get("cagr3")
    g1 = float(np.clip(g_auto, -0.02, 0.15)) if g_auto is not None else 0.05
    g1_label = f"자동 {pct(g1)} (과거 FCF CAGR 기반, 상하한 −2%~15%)"
    g1_caution = (g_auto is not None and g_auto > 0.12)
else:
    g1, g1_label = g_manual, f"수동 {pct(g_manual)}"
    g1_caution = g_manual > 0.12

scen = dcf_m.run_scenarios(fcf0, wacc, g1, gT, net_debt, fd.shares, fd.price) \
    if fcf0 else None
fair_base = scen["기준"]["fair"] if scen else None
sens = dcf_m.sensitivity(fcf0, wacc, g1, net_debt, fd.shares) if fcf0 else None
reverse = dcf_m.reverse_dcf(fd.price, wacc, gT, net_debt, fd.shares, fcf0) \
    if fcf0 else None

# ── 채점 ───────────────────────────────────────────────────────────────────
if fd.is_financial:
    fin_reason = ["금융회사 — 일반 모델 부적합(5단계 전용 모델 예정)"]
    components = {k: (None, WEIGHTS[k], fin_reason)
                  for k in ("roic", "fcf", "reinvestment", "debt")}
else:
    components = {
        "roic": score_roic(roic_res, wacc),
        "fcf": score_fcf(cf_res),
        "reinvestment": score_reinvestment(re_res, wacc),
        "debt": score_debt(debt_res),
    }
for k, why in NOT_IMPLEMENTED.items():
    components[k] = (None, WEIGHTS[k], [f"미구현({why})"])
components["valuation"] = score_valuation(fd.price, fair_base,
                                          mult.get("fcf_yield"), rf,
                                          mult.get("per"))

risk_codes = set()
for r in (roic_res, cf_res, re_res, debt_res):
    risk_codes |= r.get("risk_codes", set())
penalty_total, penalty_items = collect_penalties(risk_codes, data_shortage)

scores = aggregate(components, penalty_total)
z_zone = debt_res["altman"]["zone"] if debt_res.get("altman") else None
cls = classify(scores["quality_norm"], scores["val_norm"],
               components.get("debt"), z_zone, fd.is_financial,
               roic_res["summary"]["years"])
concl = nr.conclusion(fd, scores, cls, scen, reverse, mult,
                      roic_res, cf_res, re_res, debt_res, mos_target)

# ── 화면 2: 종합 요약 헤더 ─────────────────────────────────────────────────
h1, h2 = st.columns([3, 2])
with h1:
    st.subheader(f"{fd.name}  ·  {fd.ticker}")
    st.caption(f"{fd.sector or '섹터 N/A'} / {fd.industry or '산업 N/A'} · "
               f"재무통화 {cur or 'N/A'} · 출처 {fd.source}")
with h2:
    st.metric("현재주가", price_fmt(fd.price, fd.currency),
              help="시가총액 " + money(fd.market_cap, fd.currency))

c = st.columns(6)
c[0].metric("종합점수(환산)", f"{scores['total_norm']:.0f}" if scores["total_norm"]
            is not None else "N/A")
c[1].metric("등급", scores["grade"])
c[2].metric("기업의 질", f"{scores['quality_norm']:.0f}" if scores["quality_norm"]
            is not None else "N/A")
c[3].metric("밸류에이션", f"{scores['val_norm']:.0f}" if scores["val_norm"]
            is not None else "N/A")
c[4].metric("적정가치(기준)", price_fmt(fair_base, fd.currency))
c[5].metric("안전마진(기준)", pct(scen["기준"]["mos"]) if scen else "N/A",
            help=f"목표 안전마진 {pct(mos_target,0)}")

st.info(f"**분류(프로그램의 추론):** {cls[0]} — {cls[1]}")
st.markdown(f"**핵심 판단:** {concl['thesis']}")
if scores.get("partial_note"):
    st.warning("⚠️ " + scores["partial_note"] +
               " — 해자·경기방어력이 미평가된 환산 점수이므로 해당 요소가 약한 기업은 "
               "점수가 실제보다 높게 나올 수 있습니다.")

for m in msgs:
    (st.error if m["level"] == "error" else
     st.warning if m["level"] == "warn" else st.caption)(m["msg"])

if fd.is_financial:
    eq_s, ni_s = _g(annual, "equity"), _g(annual, "net_income")
    div_s = _g(annual, "dividends_out")
    roe = (ni_s / eq_s.where(eq_s > 0)).dropna()
    with st.container(border=True):
        st.markdown("**🏦 금융회사 참고 패널** — 은행·보험은 ROE·자본비율(CET1)·"
                    "순이자마진·대손비용 중심으로 평가해야 하며, 아래는 공개 데이터로 "
                    "산출 가능한 참고치입니다(전용 모델은 5단계).")
        if len(roe):
            fm = st.columns(4)
            fm[0].metric("평균 ROE", pct(float(roe.mean())))
            fm[1].metric("최근 ROE", pct(float(roe.iloc[-1])))
            fm[2].metric("PBR", xs(mult["pbr"]))
            payout = (div_s / ni_s.where(ni_s > 0)).dropna()
            fm[3].metric("평균 배당성향", pct(float(payout.mean()), 0)
                         if len(payout) else "N/A")
            fin_df = pd.DataFrame({"ROE": roe.map(lambda v: pct(v))})
            if len(payout):
                fin_df["배당성향"] = payout.map(lambda v: pct(v, 0))
            st.dataframe(fin_df.T, use_container_width=True)
            if mult["pbr"] is not None and roe.mean() is not None:
                st.caption(f"참고: PBR {xs(mult['pbr'])} · 평균 ROE {pct(float(roe.mean()))} — "
                           f"ROE가 자기자본비용(≈Ke {pct(wacc_res['ke'])})을 지속 상회하는지가 "
                           f"PBR 1배 이상을 정당화하는 핵심입니다.")
        else:
            st.caption("ROE 산출 불가(자기자본·순이익 데이터 부족)")

tabs = st.tabs(["요약·점수", "ROIC", "현금흐름", "재투자", "재무 안전성",
                "밸류에이션·DCF", "투자 결론", "데이터·다운로드"])

# ── 요약·점수 ──────────────────────────────────────────────────────────────
with tabs[0]:
    rows = []
    for k in ("roic", "fcf", "reinvestment", "moat", "debt",
              "cyclicality", "valuation"):
        pts, mx, det = components[k]
        rows.append({"항목": COMP_LABEL[k],
                     "점수": "N/A" if pts is None else f"{pts:.1f}",
                     "배점": mx,
                     "비고": det[0] if det else ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(f"획득 {scores['achieved']} / 가용 배점 {scores['available']}  ·  "
                f"감점 {scores['penalty']}  →  환산 "
                f"**{scores['total_norm']}점**" if scores["total_norm"] is not None
                else "환산 불가")
    with st.expander("항목별 채점 근거"):
        for k in components:
            pts, mx, det = components[k]
            st.markdown(f"**{COMP_LABEL[k]}** ({'N/A' if pts is None else pts}/{mx})")
            for line in det:
                st.markdown(f"- {line}")
    if penalty_items:
        st.markdown("**감점 내역**")
        for label, p in penalty_items:
            st.markdown(f"- {label}: {p:+.1f}")

# ── ROIC (화면 3) ──────────────────────────────────────────────────────────
with tabs[1]:
    st.plotly_chart(charts.roic_chart(roic_res["table"], wacc, rf),
                    use_container_width=True)
    s = roic_res["summary"]
    m = st.columns(6)
    m[0].metric("최근 연도", pct(s["latest"]))
    m[1].metric("TTM", pct(s["ttm"]))
    m[2].metric("3년 평균", pct(s["mean3"]))
    m[3].metric(f"기간 평균({s['years']}년)", pct(s["mean_all"]))
    m[4].metric("표준편차", pct(s["std"]))
    m[5].metric("최저치", pct(s["min"]))
    m2 = st.columns(4)
    m2[0].metric("15%↑ 연도 비율", pct(s["pct_ge_15"], 0))
    m2[1].metric("20%↑ 연도 비율", pct(s["pct_ge_20"], 0))
    m2[2].metric("WACC 초과 비율", pct(s["pct_gt_wacc"], 0))
    m2[3].metric("ROIC−WACC", "N/A" if s["spread_wacc"] is None
                 else f"{s['spread_wacc']*100:+.1f}%p")
    if roic_res["decomposition"]:
        d = roic_res["decomposition"]
        st.markdown(f"**분해({d['fy']}):** ROIC = NOPAT마진 {pct(d['nopat_margin'])} × "
                    f"투하자본회전율 {d['ic_turnover']:.2f}회")
    st.caption(f"투하자본 방식: {s['ic_method']}")
    st.markdown(nr.roic_text(roic_res, wacc))
    for f in roic_res["flags"]:
        st.warning(f)
    st.dataframe(fmt_table(roic_res["table"], cur,
                           money_cols=("revenue", "ebit", "nopat", "ic_a",
                                       "ic_b", "ic"),
                           pct_cols=("eff_tax", "roic")),
                 use_container_width=True)

# ── 현금흐름 (화면 4) ──────────────────────────────────────────────────────
with tabs[2]:
    st.plotly_chart(charts.cash_chart(cf_res["table"]), use_container_width=True)
    st.plotly_chart(charts.margin_chart(cf_res["table"]), use_container_width=True)
    s = cf_res["summary"]
    m = st.columns(6)
    m[0].metric("FCF(최근)", money(s["fcf_latest"], cur))
    m[1].metric("FCF(TTM)", money(s["fcf_ttm"], cur))
    m[2].metric("FCF 마진(평균)", pct(s["margin_avg"]))
    m[3].metric("FCF CAGR(기간)", pct(s["cagr_max"]))
    m[4].metric("현금전환율(평균)", pct(s["conv_avg"], 0))
    m[5].metric("주식수 변화", pct(s["share_change"]))
    st.markdown(nr.fcf_text(cf_res))
    for f in cf_res["flags"]:
        st.warning(f)
    st.dataframe(fmt_table(cf_res["table"], cur,
                           money_cols=("revenue", "net_income", "ocf",
                                       "capex_out", "fcf", "sbc_out",
                                       "fcf_adj"),
                           pct_cols=("fcf_margin", "conversion")),
                 use_container_width=True)

# ── 재투자 (화면 5) ────────────────────────────────────────────────────────
with tabs[3]:
    s = re_res["summary"]
    m = st.columns(4)
    m[0].metric("재투자율(평균)", pct(s["rr_avg"], 0),
                help=s["rr_method"] or "산출 불가")
    m[1].metric("증분 ROIC(1년)", pct(s["inc_roic"].get(1)))
    m[2].metric("증분 ROIC(3년)", pct(s["inc_roic"].get(3)))
    m[3].metric("지속가능성장률", pct(s["sgr"]))
    if s["quadrant"]:
        st.markdown(f"**판정:** {s['quadrant'][0]} — {s['quadrant'][1]}")
    st.markdown(nr.reinvest_text(re_res))
    for f in re_res["flags"]:
        st.warning(f)
    st.dataframe(fmt_table(re_res["table"], cur,
                           money_cols=("capex_out", "depreciation", "d_wc",
                                       "acquisitions_out", "dividends_out",
                                       "buybacks_out"),
                           pct_cols=("rr_capex", "rr_payout")),
                 use_container_width=True)

# ── 재무 안전성 (화면 7) ───────────────────────────────────────────────────
with tabs[4]:
    L = debt_res["latest"]
    m = st.columns(6)
    m[0].metric("순부채", money(L["net_debt"], cur))
    m[1].metric("순부채/EBITDA", xs(L["nd_ebitda"]))
    m[2].metric("이자보상배율", "무차입" if L["debt_free"] else xs(L["icov"]))
    m[3].metric("부채/자기자본", xs(L["de"]))
    m[4].metric("유동비율", "N/A" if L["cur_ratio"] is None
                else f"{L['cur_ratio']:.2f}")
    m[5].metric("순부채/FCF", xs(L["nd_fcf"]))
    a, p = debt_res.get("altman"), debt_res.get("piotroski")
    m2 = st.columns(3)
    m2[0].metric("Altman Z", "N/A" if not a else f"{a['z']:.2f} ({a['zone']})")
    m2[1].metric("Piotroski F", "N/A" if not p else f"{p['score']} / {p['valid']}")
    m2[2].metric("단기부채 비중", pct(L["short_share"], 0))
    st.markdown(nr.debt_text(debt_res))
    if debt_res["warnings"]:
        st.markdown("**🔴 적색 경고**")
        for w in debt_res["warnings"]:
            st.error(w)
    else:
        st.success("적색 경고 없음 (명세 10장 계산 가능 항목 기준)")
    if p:
        with st.expander("Piotroski 세부 항목"):
            for name, ok in p["detail"]:
                mark = "✅" if ok else ("❌" if ok is not None else "➖ N/A")
                st.markdown(f"- {name}: {mark}")
    st.plotly_chart(charts.debt_chart(debt_res["net_debt_series"]),
                    use_container_width=True)

# ── 밸류에이션·DCF (화면 9) ────────────────────────────────────────────────
with tabs[5]:
    st.markdown(f"지표 산출 기준 — {mult['basis']}")
    rows = [
        ("PER", xs(mult["per"])), ("Forward PER", xs(mult["forward_pe"])),
        ("PBR", xs(mult["pbr"])), ("PSR", xs(mult["psr"])),
        ("EV/EBITDA", xs(mult["ev_ebitda"])), ("EV/EBIT", xs(mult["ev_ebit"])),
        ("P/FCF", xs(mult["p_fcf"])), ("FCF Yield", pct(mult["fcf_yield"])),
        ("Earnings Yield", pct(mult["earn_yield"])),
        ("주주환원수익률", pct(mult["shareholder_yield"])),
        ("PEG", "N/A(4단계)"),
        ("FCF수익률 − rf", "N/A" if mult["spread_fcf_rf"] is None
         else f"{mult['spread_fcf_rf']*100:+.1f}%p"),
        ("Earnings수익률 − rf", "N/A" if mult["spread_earn_rf"] is None
         else f"{mult['spread_earn_rf']*100:+.1f}%p"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["지표", "값"]),
                 use_container_width=True, hide_index=True)
    st.caption(f"WACC {pct(wacc)} = Ke {pct(wacc_res['ke'])}×{pct(wacc_res['we'],0)} "
               f"+ 세후Kd {pct(wacc_res['kd_after'])}×{pct(wacc_res['wd'],0)} · "
               f"β {wacc_res['beta_used']:.2f}")
    for n in wacc_res["notes"]:
        st.caption("· " + n)

    st.divider()
    st.subheader("DCF")
    st.caption(f"기준 FCF: {money(fcf0, cur)} ({fcf0_label}) · 1단계 성장률: {g1_label} · "
               f"영구성장률 {pct(gT)}")
    if g1_caution:
        st.warning("성장률 가정 주의: 과거 고성장(연 12%↑)의 단순 외삽은 가치평가에서 "
                   "가장 흔한 오류입니다. 보수적 시나리오와 역산 DCF(시장 내재 기대치)를 "
                   "기준으로 판단하세요.")
    if scen:
        sdf = pd.DataFrame(scen).T
        sdf_disp = pd.DataFrame({
            "성장률(5년)": sdf["g1"].map(lambda v: pct(v)),
            "WACC": sdf["wacc"].map(lambda v: pct(v)),
            "영구성장률": sdf["gT"].map(lambda v: pct(v)),
            "적정가치": sdf["fair"].map(lambda v: price_fmt(v, fd.currency)),
            "상승여력": sdf["upside"].map(lambda v: pct(v)),
            "안전마진": sdf["mos"].map(lambda v: pct(v)),
            "TV 비중": sdf["tv_share"].map(lambda v: pct(v, 0)),
        })
        st.dataframe(sdf_disp, use_container_width=True)
        tv_b = scen["기준"].get("tv_share")
        if tv_b is not None and tv_b >= 0.75:
            st.warning(f"기준 시나리오 적정가치의 {pct(tv_b,0)}가 6년차 이후 "
                       f"영구성장 구간(Terminal Value)에서 나옵니다 — 적정가치가 "
                       f"영구성장률·WACC 가정에 매우 민감하니 민감도 표와 함께 해석하세요.")
        for name, sc in scen.items():
            if sc.get("note"):
                st.caption(f"· {name}: {sc['note']}")
        st.plotly_chart(charts.dcf_chart(scen, fd.price, fd.currency),
                        use_container_width=True)
        if reverse:
            if reverse["implied_g"] is not None:
                st.metric("역산 DCF — 현 주가 내재 성장률(5년 FCF)",
                          pct(reverse["implied_g"]))
                st.caption("현재 주가가 정당화되려면 필요한 1단계 FCF 성장률 — "
                           "시장 기대치의 근사치입니다.")
            if reverse.get("msg"):
                st.warning(reverse["msg"])
        if sens is not None:
            st.plotly_chart(charts.sens_heatmap(sens), use_container_width=True)
        if fair_base and fd.price:
            st.markdown("**매수가격 구간 (기준 시나리오 적정가치 대비)**")
            st.dataframe(dcf_m.price_zones(fair_base, fd.price),
                         use_container_width=True, hide_index=True)
            st.caption(f"목표 안전마진 {pct(mos_target,0)} 기준 매수 검토가: "
                       f"{price_fmt(fair_base*(1-mos_target), fd.currency)} 이하")
    else:
        st.warning("기준 FCF ≤ 0 또는 데이터 부족 — DCF는 임의값을 대입하지 않고 "
                   "N/A 처리합니다(명세 18·21).")
    if fd.price_history is not None:
        st.plotly_chart(charts.price_chart(fd.price_history),
                        use_container_width=True)

# ── 투자 결론 (화면 10) ────────────────────────────────────────────────────
with tabs[6]:
    a, b = st.columns(2)
    with a:
        st.markdown("**강점**")
        for x in concl["strengths"]:
            st.markdown(f"- {x}")
        st.markdown("**핵심 리스크**")
        for x in concl["risks"]:
            st.markdown(f"- {x}")
        st.markdown("**다음 분기 확인 지표**")
        for x in concl["checkpoints"]:
            st.markdown(f"- {x}")
    with b:
        st.markdown("**약점**")
        for x in concl["weaknesses"]:
            st.markdown(f"- {x}")
        st.markdown("**투자 논리 훼손 조건**")
        for x in concl["breakers"]:
            st.markdown(f"- {x}")
    if scen and fair_base:
        st.divider()
        st.markdown(f"적정가치 범위(보수~낙관): "
                    f"{price_fmt(scen['보수적']['fair'], fd.currency)} ~ "
                    f"{price_fmt(scen['낙관적']['fair'], fd.currency)} · "
                    f"현재 {price_fmt(fd.price, fd.currency)}")

# ── 데이터·다운로드 ────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown(f"**원천 데이터** — {fd.source} (연간, 재무통화 {cur or 'N/A'})")
    st.dataframe(annual, use_container_width=True)
    if fd.ttm:
        st.markdown("**TTM(최근 4개 분기 합)**")
        st.json({k: (None if not np.isfinite(v) else v) for k, v in fd.ttm.items()})
    assumptions = {
        "무위험수익률": pct(rf), "ERP": pct(erp), "대체 법인세율": pct(tax_fb),
        "WACC": pct(wacc), "투하자본 방식": roic_res["summary"]["ic_method"],
        "DCF 기준 FCF": f"{money(fcf0, cur)} ({fcf0_label})",
        "1단계 성장률": g1_label, "영구성장률": pct(gT),
        "목표 안전마진": pct(mos_target, 0),
        "분석기간": roic_res["summary"]["period"],
    }
    st.markdown("**가정(Assumptions)**")
    st.dataframe(pd.DataFrame({"항목": assumptions.keys(),
                               "값": assumptions.values()}),
                 use_container_width=True, hide_index=True)
    xls = build_excel(fd, roic_res, cf_res, re_res, debt_res, mult, scen, sens,
                      scores, penalty_items, assumptions)
    st.download_button("📥 Excel 보고서 다운로드", data=xls,
                       file_name=f"{fd.ticker}_analysis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet")

st.divider()
st.caption("본 프로그램의 결과는 공개 데이터 기반 자동 계산이며 투자 조언이 아닙니다. "
           "확인된 사실·계산 결과·프로그램의 추론을 구분해 표기했으며, 데이터 공백은 "
           "N/A로 처리합니다. 최종 투자 판단과 책임은 사용자에게 있습니다.")

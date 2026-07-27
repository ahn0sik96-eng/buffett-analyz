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
from data import data_fetcher
from data import market_rates
from models import dcf as dcf_m
from reports import narrative_report as nr
from reports.excel_report import build_excel
from analysis_pipeline import analyze
from ui import charts
from ui import mobile as mob
from ui.formatting import pct, xs, money, price_fmt

COMP_LABEL = {"roic": "ROIC", "fcf": "FCF", "reinvestment": "재투자",
              "moat": "경제적 해자", "debt": "부채·안전성",
              "cyclicality": "경기 방어력", "valuation": "밸류에이션"}

st.set_page_config(page_title="버핏·멍거 우량기업 분석기", page_icon="📒",
                   layout="wide", initial_sidebar_state="auto")

mob.inject_css()
if "mobile_view" not in st.session_state:
    st.session_state.mobile_view = mob.detect_mobile()


@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker: str):
    fd = data_fetcher.fetch(ticker)
    import datetime as _dt
    fd.fetched_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return fd


@st.cache_data(ttl=settings.RF_CACHE_TTL, show_spinner=False)
def load_rf(country: str) -> dict:
    """무위험수익률 자동 조회(실패 시 폴백값 + 사유). 6시간 캐시."""
    return market_rates.fetch_rf(country)


def guess_country(raw: str) -> str:
    """입력값 → 국가코드. 데이터 수집기와 동일한 판별 규칙을 재사용한다.

    앞 2글자에 숫자가 있는지 보는 방식은 종목명 검색('삼성전자')에서 오판하므로
    resolve_candidates()를 그대로 쓴다. 여러 종목이 들어오면 첫 종목 기준.
    """
    import re as _re
    first = next((t.strip() for t in _re.split(r"[,\n]", raw or "") if t.strip()),
                 "")
    if not first:
        return "US"
    try:
        return data_fetcher.resolve_candidates(first)[1]
    except Exception:
        return "US"


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
    st.title("📒 종목 분석")
    st.session_state.mobile_view = st.toggle(
        "📱 모바일 보기", value=st.session_state.mobile_view,
        help="지표를 2칸으로 배치하고 차트·표를 작은 화면에 맞춥니다. "
             "접속 기기에 따라 자동 설정되며, 직접 바꿀 수 있습니다.")
    MOBILE = st.session_state.mobile_view
    mode = st.radio("모드", ["단일 분석", "비교·워치리스트"], horizontal=True)
    if mode == "단일 분석":
        ticker_in = st.text_input("티커 / 한국 6자리 코드",
                                  placeholder="예: AAPL, V, 005930, 삼성전자")
        tickers_in = None
        st.caption("미국: 티커 · 한국: 6자리 코드(.KS/.KQ 자동 판별) 또는 대표 종목명")
    else:
        tickers_in = st.text_area(
            "티커 여러 개 (쉼표 또는 줄바꿈으로 구분)",
            placeholder="AAPL, MSFT, V\n005930, 000660",
            height=100)
        ticker_in = None
        st.caption("최대 12개 권장 — 각 종목을 동일 가정으로 채점해 표로 비교합니다.")
    st.divider()
    st.subheader("가정 (수정 가능)")
    _hint_src = ticker_in or (tickers_in or "")
    country_hint = guess_country(_hint_src)
    # 위젯 key에 국가코드를 넣어야 종목을 바꿨을 때 기본값이 새로 반영된다
    # (Streamlit 위젯의 value 인자는 최초 렌더에만 적용되므로).
    _k = country_hint

    rf_auto = st.checkbox("무위험수익률 자동 조회", value=True,
                          help="FRED에서 10년물 국채 수익률을 받아옵니다. "
                               "실패하면 기본값을 쓰고 사유를 표시합니다.")
    if rf_auto:
        _rfd = load_rf(country_hint)
    else:
        _rfd = {"rf": settings.DEFAULT_RF.get(country_hint, 0.04),
                "auto": False, "label": "자동 조회 꺼짐 — 기본값"}
    rf = st.number_input(
        "무위험수익률 (10년물, %)", value=_rfd["rf"] * 100,
        min_value=0.0, max_value=15.0, step=0.1,
        key=f"rf_{_k}_{int(rf_auto)}_{_rfd['rf']:.4f}") / 100
    (st.caption if _rfd["auto"] else st.warning)(
        ("✅ " if _rfd["auto"] else "⚠️ ") + _rfd["label"])

    _erp_def = settings.DEFAULT_ERP_BY_COUNTRY.get(country_hint,
                                                   settings.DEFAULT_ERP)
    erp = st.number_input("주식위험프리미엄 (%)", value=_erp_def * 100,
                          min_value=1.0, max_value=12.0, step=0.25,
                          key=f"erp_{_k}") / 100
    tax_fb = st.number_input(
        "대체 법인세율 (%)",
        value=settings.TAX_FALLBACK.get(country_hint, .25) * 100,
        min_value=0.0, max_value=45.0, step=0.5, key=f"tax_{_k}") / 100

    _ke_preview = rf + erp
    st.caption(f"→ 판별 국가 **{country_hint}** · 자기자본비용(β=1 기준) "
               f"{pct(_ke_preview)}")
    if _ke_preview > 0.105:
        st.warning("무위험수익률과 ERP를 동시에 높게 잡으면 이중계상입니다 — "
                   "금리가 오르면 실현 ERP는 압축되는 것이 정상이라, "
                   "합계(Ke)가 9~10% 밴드에 들어오는지를 기준으로 보세요.")
    ic_method = st.selectbox("투하자본 산정", ["auto", "A", "B"],
                             help="A: 총자산−현금−무이자유동부채 / B: 자기자본+이자부부채−현금")
    fcf_base_opt = st.selectbox("DCF 기준 FCF", ["3년 중앙값", "TTM", "최근 연도"])
    g_mode = st.selectbox("1단계 성장률(5년)", ["자동(과거 FCF CAGR 기반)", "수동"])
    g_manual = st.slider("수동 성장률 (%)", -10.0, 25.0, 6.0, 0.5) / 100
    _gT_def = settings.DEFAULT_TERMINAL_G_BY_COUNTRY.get(
        country_hint, settings.DEFAULT_TERMINAL_G)
    gT = st.number_input("영구성장률 (%)", value=_gT_def * 100,
                         min_value=0.0, max_value=4.0, step=0.25,
                         key=f"gt_{_k}",
                         help="장기 명목 GDP 성장률을 넘을 수 없습니다. "
                              "한국은 인구구조를 반영해 기본값이 더 낮습니다.") / 100

    mos_preset = st.selectbox(
        "목표 안전마진 프리셋", list(settings.MOS_PRESETS.keys()),
        index=list(settings.MOS_PRESETS).index(settings.DEFAULT_MOS_PRESET),
        help="이익 예측가능성이 낮을수록 안전마진을 높게 잡습니다. "
             "반도체·시클리컬에 30%는 사실상 여유가 없는 수준입니다.")
    _mos_val = settings.MOS_PRESETS[mos_preset]
    if _mos_val is None:
        mos_target = st.slider("목표 안전마진 (%)", 10, 60,
                               int(settings.DEFAULT_MOS_TARGET * 100), 5) / 100
    else:
        mos_target = _mos_val
        st.caption(f"목표 안전마진 {pct(mos_target, 0)} 적용")
    run = st.button("분석 실행", type="primary", use_container_width=True)
    if st.button("🔄 데이터 캐시 지우기", use_container_width=True,
                 help="최신 주가·재무를 다시 받아옵니다(1시간 캐시 무시)"):
        st.cache_data.clear()
        st.success("캐시를 비웠습니다. 다시 분석 실행하세요.")

ASSUMPTIONS = dict(rf=rf, erp=erp, tax_fb=tax_fb, ic_method=ic_method,
                   fcf_base_opt=fcf_base_opt, g_mode=g_mode, g_manual=g_manual,
                   gT=gT, mos_target=mos_target)

st.title("버핏·멍거식 우량기업 분석기")
st.caption(f"ROIC · FCF · 재투자 · 재무안전성 · 경기방어(간이) · DCF — 장기 복리 적합성 평가　|　"
           f"**{settings.APP_VERSION}**")

if not run:
    st.info("좌측에서 모드를 고르고 티커를 입력한 뒤 **분석 실행**을 누르세요. "
            "단일 분석은 상세 리포트, 비교·워치리스트는 여러 종목 점수표를 보여줍니다.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  비교·워치리스트 모드
# ══════════════════════════════════════════════════════════════════════════
if mode == "비교·워치리스트":
    import re as _re
    raw = [t.strip() for t in _re.split(r"[,\n]", tickers_in or "") if t.strip()]
    seen, tickers = set(), []
    for t in raw:
        if t.upper() not in seen:
            seen.add(t.upper())
            tickers.append(t)
    if not tickers:
        st.warning("티커를 하나 이상 입력하세요.")
        st.stop()
    tickers = tickers[:12]

    rows, errors = [], []
    prog = st.progress(0.0, text="분석 중…")
    for i, t in enumerate(tickers):
        try:
            fd = load(t)
            r = analyze(fd, ASSUMPTIONS)
            sc = r["scores"]
            base = r["scen"]["기준"] if r["scen"] else {}
            if not base and r.get("fin_val"):
                base = {"mos": r["fin_val"]["summary"].get("mos")}
            rows.append({
                "티커": fd.ticker,
                "기업": (fd.name or "")[:22],
                "분류": r["cls"][0],
                "종합": sc["total_norm"],
                "등급": sc["grade"],
                "질": sc["quality_norm"],
                "밸류": sc["val_norm"],
                "ROIC(평균)": r["roic_res"]["summary"].get("mean_all"),
                "FCF마진": r["cf_res"]["summary"].get("margin_avg"),
                "순부채/EBITDA": r["debt_res"]["latest"].get("nd_ebitda"),
                "안전마진(기준)": base.get("mos"),
                "내재성장률": r["reverse"]["implied_g"] if r["reverse"] else None,
                "경고": len(r["debt_res"].get("warnings", [])),
                "데이터연수": r["roic_res"]["summary"].get("years"),
            })
        except Exception as e:
            errors.append(f"{t}: {e}")
        prog.progress((i + 1) / len(tickers), text=f"분석 중… ({i+1}/{len(tickers)})")
    prog.empty()

    if errors:
        with st.expander(f"⚠️ 분석 실패 {len(errors)}건"):
            for e in errors:
                st.caption(e)
    if not rows:
        st.error("분석된 종목이 없습니다.")
        st.stop()

    df = pd.DataFrame(rows)
    sort_key = st.selectbox("정렬 기준",
                            ["종합", "질", "밸류", "ROIC(평균)", "안전마진(기준)"],
                            index=0)
    df = df.sort_values(sort_key, ascending=False,
                        key=lambda s: pd.to_numeric(s, errors="coerce"),
                        na_position="last").reset_index(drop=True)

    disp = df.copy()
    for c in ("종합", "질", "밸류"):
        disp[c] = disp[c].map(lambda v: "N/A" if v is None else f"{v:.0f}")
    for c in ("ROIC(평균)", "FCF마진", "안전마진(기준)", "내재성장률"):
        disp[c] = disp[c].map(lambda v: pct(v))
    disp["순부채/EBITDA"] = disp["순부채/EBITDA"].map(lambda v: xs(v))
    if MOBILE:
        # 휴대폰: 핵심 열만 (가로 스크롤 최소화). 전체는 CSV로 확인
        disp = disp[["티커", "종합", "등급", "질", "밸류", "안전마진(기준)"]]
        st.caption("📱 모바일 보기 — 핵심 열만 표시합니다. 전체 지표는 아래 CSV를 받거나 "
                   "사이드바에서 모바일 보기를 끄세요.")
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption("‘질’은 밸류에이션을 제외한 기업 품질 환산점수 — 이 값이 높은데 ‘밸류’가 "
               "낮으면 ‘좋은데 비싼’ 워치리스트 후보입니다. ‘내재성장률’은 현 주가가 "
               "정당화되려면 필요한 5년 FCF 성장률(시장 기대치의 근사)입니다.")

    # 워치리스트 후보 자동 추림: 질 높은데 밸류 낮은
    cand = df[(pd.to_numeric(df["질"], errors="coerce") >= 75) &
              (pd.to_numeric(df["밸류"], errors="coerce") < 55)]
    if len(cand):
        st.markdown("**⭐ 워치리스트 후보 (질 우수·현재 가격 부담):** " +
                    ", ".join(cand["티커"].tolist()) +
                    " — 조정 시 매수 검토 대상")

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 비교표 CSV 다운로드", data=csv,
                       file_name="comparison.csv", mime="text/csv")
    st.divider()
    st.caption("본 결과는 공개 데이터 기반 자동 계산이며 투자 조언이 아닙니다. "
               "동일 가정을 모든 종목에 적용했으므로 개별 종목의 특수성은 단일 분석에서 확인하세요.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  단일 분석 모드
# ══════════════════════════════════════════════════════════════════════════
if not ticker_in:
    st.warning("티커를 입력하세요.")
    st.stop()

try:
    with st.spinner("재무데이터 수집 중…"):
        fd = load(ticker_in.strip())
        R = analyze(fd, ASSUMPTIONS)
except Exception as e:
    st.error(str(e))
    st.stop()

# 파이프라인 결과 언팩 (아래 탭 렌더링이 참조하는 지역변수명 유지)
cur = R["cur"]; msgs = R["msgs"]; data_shortage = R["data_shortage"]
wacc_res = R["wacc_res"]; wacc = R["wacc"]
roic_res = R["roic_res"]; cf_res = R["cf_res"]; re_res = R["re_res"]
debt_res = R["debt_res"]; mult = R["mult"]; cyc_res = R["cyc_res"]
net_debt = R["net_debt"]; annual = fd.annual
fcf0 = R["fcf0"]; fcf0_label = R["fcf0_label"]
g1 = R["g1"]; g1_label = R["g1_label"]; g1_caution = R["g1_caution"]
scen = R["scen"]; fx_sanity_msg = R["fx_sanity_msg"]; fair_base = R["fair_base"]
sens = R["sens"]; reverse = R["reverse"]
components = R["components"]; penalty_items = R["penalty_items"]
scores = R["scores"]; cls = R["cls"]; concl = R["concl"]
fin_val = R.get("fin_val")
penalty_total = scores["penalty"]

# ── 화면 2: 종합 요약 헤더 ─────────────────────────────────────────────────
st.subheader(f"{fd.name}  ·  {fd.ticker}")
st.caption(f"{fd.sector or '섹터 N/A'} / {fd.industry or '산업 N/A'} · "
           f"재무통화 {cur or 'N/A'} · 출처 {fd.source}"
           + (f" · 수집 {fd.fetched_at}" if getattr(fd, 'fetched_at', '') else ""))

mob.metric_row([
    ("현재주가", price_fmt(fd.price, fd.currency),
     "시가총액 " + money(fd.market_cap, fd.currency)),
    ("종합점수(환산)", f"{scores['total_norm']:.0f}"
     if scores["total_norm"] is not None else "N/A"),
    ("등급", scores["grade"]),
    ("기업의 질", f"{scores['quality_norm']:.0f}"
     if scores["quality_norm"] is not None else "N/A"),
    ("밸류에이션", f"{scores['val_norm']:.0f}"
     if scores["val_norm"] is not None else "N/A"),
    ("적정가치(기준)", price_fmt(fair_base, fd.currency)),
    ("안전마진(기준)", pct(scen["기준"]["mos"]) if scen else "N/A",
     f"목표 안전마진 {pct(mos_target,0)}"),
], MOBILE, per_row_desktop=4, per_row_mobile=2)

st.info(f"**분류(프로그램의 추론):** {cls[0]} — {cls[1]}")
st.markdown(f"**핵심 판단:** {concl['thesis']}")
if scores.get("partial_note"):
    st.warning("⚠️ " + scores["partial_note"] +
               " — 해자·경기방어력이 미평가된 환산 점수이므로 해당 요소가 약한 기업은 "
               "점수가 실제보다 높게 나올 수 있습니다.")

for m in msgs:
    (st.error if m["level"] == "error" else
     st.warning if m["level"] == "warn" else st.caption)(m["msg"])

def render_financial_panel(fv, container_border=True):
    """금융업 초과수익모형 패널 — 요약 헤더와 밸류·DCF 탭에서 공유."""
    s = fv["summary"]
    ctx = st.container(border=True) if container_border else st.container()
    with ctx:
        st.markdown(f"**🏦 금융회사 전용 모델 — {fv['subtype']}**　"
                    f"적정 PBR = (ROE − g) / (Ke − g)")
        if not s:
            for f in fv["flags"]:
                st.warning(f)
            return
        mob.metric_row([
            ("지속가능 ROE", pct(s["roe_med"]),
             f"최근 {min(settings.FIN_ROE_YEARS, s['years'])}년 중앙값"),
            ("자기자본비용 Ke", pct(s["ke"])),
            ("ROE − Ke", "N/A" if s["spread"] is None
             else f"{s['spread']*100:+.1f}%p", "양수여야 가치 창출"),
            ("ROA(평균)", pct(s["roa_avg"])),
            ("배당성향(평균)", pct(s["payout_avg"], 0)),
            ("적용 성장률 g", pct(s["g_used"])),
        ], MOBILE)
        mob.metric_row([
            ("BPS", price_fmt(s["bps"], fd.currency)),
            ("현재 PBR", xs(s["pbr_now"])),
            ("적정 PBR", xs(s["pb_fair"])),
            ("적정주가", price_fmt(s["fair"], fd.currency)),
            ("안전마진", pct(s["mos"])),
            ("연간 초과수익", money(s["excess_return"], cur)),
        ], MOBILE)
        for f in fv["flags"]:
            st.warning(f)
        for n in fv["notes"]:
            st.caption("· " + n)


if fd.is_financial and fin_val:
    render_financial_panel(fin_val)

tabs = st.tabs(["요약", "ROIC", "현금", "재투자", "안전성",
                "밸류·DCF", "결론", "데이터"] if MOBILE else
               ["요약·점수", "ROIC", "현금흐름", "재투자", "재무 안전성",
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
    mob.chart(charts.roic_chart(roic_res["table"], wacc, rf), MOBILE)
    s = roic_res["summary"]
    mob.metric_row([
        ("최근 연도", pct(s["latest"])),
        ("TTM", pct(s["ttm"])),
        ("3년 평균", pct(s["mean3"])),
        (f"기간 평균({s['years']}년)", pct(s["mean_all"])),
        ("표준편차", pct(s["std"])),
        ("최저치", pct(s["min"])),
    ], MOBILE)
    mob.metric_row([
        ("15%↑ 연도 비율", pct(s["pct_ge_15"], 0)),
        ("20%↑ 연도 비율", pct(s["pct_ge_20"], 0)),
        ("WACC 초과 비율", pct(s["pct_gt_wacc"], 0)),
        ("ROIC−WACC", "N/A" if s["spread_wacc"] is None
         else f"{s['spread_wacc']*100:+.1f}%p"),
    ], MOBILE, per_row_desktop=4)
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
    mob.chart(charts.cash_chart(cf_res["table"]), MOBILE)
    mob.chart(charts.margin_chart(cf_res["table"]), MOBILE)
    s = cf_res["summary"]
    mob.metric_row([
        ("FCF(최근)", money(s["fcf_latest"], cur)),
        ("FCF(TTM)", money(s["fcf_ttm"], cur)),
        ("FCF 마진(평균)", pct(s["margin_avg"])),
        ("FCF CAGR(기간)", pct(s["cagr_max"])),
        ("현금전환율(평균)", pct(s["conv_avg"], 0)),
        ("주식수 변화", pct(s["share_change"])),
    ], MOBILE)
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
    mob.metric_row([
        ("재투자율(평균)", pct(s["rr_avg"], 0), s["rr_method"] or "산출 불가"),
        ("증분 ROIC(1년)", pct(s["inc_roic"].get(1))),
        ("증분 ROIC(3년)", pct(s["inc_roic"].get(3))),
        ("지속가능성장률", pct(s["sgr"])),
    ], MOBILE, per_row_desktop=4)
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
    a, p = debt_res.get("altman"), debt_res.get("piotroski")
    mob.metric_row([
        ("순부채", money(L["net_debt"], cur)),
        ("순부채/EBITDA", xs(L["nd_ebitda"])),
        ("이자보상배율", "무차입" if L["debt_free"] else xs(L["icov"])),
        ("부채/자기자본", xs(L["de"])),
        ("유동비율", "N/A" if L["cur_ratio"] is None else f"{L['cur_ratio']:.2f}"),
        ("순부채/FCF", xs(L["nd_fcf"])),
    ], MOBILE)
    mob.metric_row([
        ("Altman Z", "N/A" if not a else f"{a['z']:.2f} ({a['zone']})"),
        ("Piotroski F", "N/A" if not p else f"{p['score']} / {p['valid']}"),
        ("단기부채 비중", pct(L["short_share"], 0)),
    ], MOBILE, per_row_desktop=3)
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
    mob.chart(charts.debt_chart(debt_res["net_debt_series"]), MOBILE)

# ── 밸류에이션·DCF (화면 9) ────────────────────────────────────────────────
with tabs[5]:
    if fx_sanity_msg:
        st.error("🚫 " + fx_sanity_msg)
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
    if fd.is_financial:
        st.warning("금융회사에서는 EV·EV/EBITDA·EV/EBIT·P/FCF·FCF Yield가 "
                   "성립하지 않습니다 — 현금과 부채가 영업의 원재료라 "
                   "기업가치(EV) 정의 자체가 무의미합니다. **PBR·PER·"
                   "배당수익률만 참고**하세요. WACC도 쓰지 않으며, 아래 "
                   "적정가치는 자기자본비용(Ke)만 사용합니다.")
    st.caption(f"WACC {pct(wacc)} = Ke {pct(wacc_res['ke'])}×{pct(wacc_res['we'],0)} "
               f"+ 세후Kd {pct(wacc_res['kd_after'])}×{pct(wacc_res['wd'],0)} · "
               f"β {wacc_res['beta_used']:.2f}")
    for n in wacc_res["notes"]:
        st.caption("· " + n)

    st.divider()
    if fd.is_financial and fin_val:
        st.subheader("적정가치 — 초과수익모형 (DCF 미실행)")
        render_financial_panel(fin_val, container_border=False)
        _fv = fin_val["summary"].get("fair")
        if _fv and fd.price:
            st.markdown("**매수가격 구간 (적정주가 대비)**")
            st.dataframe(dcf_m.price_zones(_fv, fd.price),
                         use_container_width=True, hide_index=True)
            st.caption(f"목표 안전마진 {pct(mos_target,0)} 기준 매수 검토가: "
                       f"{price_fmt(_fv*(1-mos_target), fd.currency)} 이하")
    else:
        st.subheader("DCF")
        st.caption(f"기준 FCF: {money(fcf0, cur)} ({fcf0_label}) · "
                   f"1단계 성장률: {g1_label} · 영구성장률 {pct(gT)}")
    if g1_caution and not fd.is_financial:
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
        mob.chart(charts.dcf_chart(scen, fd.price, fd.currency), MOBILE)
        if reverse:
            if reverse["implied_g"] is not None:
                ig = reverse["implied_g"]
                mob.metric_row([
                    ("① 시장 내재 성장률", pct(ig),
                     "현재 주가가 정당화되려면 필요한 5년 FCF 성장률 — 시장 기대치 근사"),
                    ("② 내 가정 성장률", pct(g1),
                     f"사이드바에서 설정한 1단계 성장률 ({g1_label})"),
                ], MOBILE, per_row_desktop=2, per_row_mobile=2)
                gap = g1 - ig
                if gap >= 0.02:
                    st.success(f"내 가정({pct(g1)})이 시장 기대({pct(ig)})보다 "
                               f"{gap*100:+.1f}%p 높습니다 → 내 전망이 맞다면 현 주가는 "
                               f"저평가. 단, 내 성장 가정이 과도하지 않은지 재확인 필요.")
                elif gap <= -0.02:
                    st.warning(f"시장 기대({pct(ig)})가 내 가정({pct(g1)})보다 "
                               f"{-gap*100:.1f}%p 높습니다 → 시장은 나보다 낙관적. "
                               f"그 성장을 못 내면 디레이팅 위험, 관망이 안전.")
                else:
                    st.info(f"시장 기대({pct(ig)})와 내 가정({pct(g1)})이 거의 일치 "
                            f"→ 현 주가에 큰 왜곡 없음. 성장 실현 여부가 관건.")
            if reverse.get("msg"):
                st.warning(reverse["msg"])
        if sens is not None:
            mob.chart(charts.sens_heatmap(sens), MOBILE)
        if fair_base and fd.price:
            st.markdown("**매수가격 구간 (기준 시나리오 적정가치 대비)**")
            st.dataframe(dcf_m.price_zones(fair_base, fd.price),
                         use_container_width=True, hide_index=True)
            st.caption(f"목표 안전마진 {pct(mos_target,0)} 기준 매수 검토가: "
                       f"{price_fmt(fair_base*(1-mos_target), fd.currency)} 이하")
    elif not fd.is_financial:
        st.warning("기준 FCF ≤ 0 또는 데이터 부족 — DCF는 임의값을 대입하지 않고 "
                   "N/A 처리합니다(명세 18·21).")
    if fd.price_history is not None:
        mob.chart(charts.price_chart(fd.price_history), MOBILE)

# ── 투자 결론 (화면 10) ────────────────────────────────────────────────────
with tabs[6]:
    def _bullets(title, items):
        st.markdown(f"**{title}**")
        for x in items:
            st.markdown(f"- {x}")

    if MOBILE:
        # 모바일: 읽는 순서대로 세로 배치 (강점→약점→리스크→훼손조건→확인지표)
        _bullets("강점", concl["strengths"])
        _bullets("약점", concl["weaknesses"])
        _bullets("핵심 리스크", concl["risks"])
        _bullets("투자 논리 훼손 조건", concl["breakers"])
        _bullets("다음 분기 확인 지표", concl["checkpoints"])
    else:
        a, b = st.columns(2)
        with a:
            _bullets("강점", concl["strengths"])
            _bullets("핵심 리스크", concl["risks"])
            _bullets("다음 분기 확인 지표", concl["checkpoints"])
        with b:
            _bullets("약점", concl["weaknesses"])
            _bullets("투자 논리 훼손 조건", concl["breakers"])
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

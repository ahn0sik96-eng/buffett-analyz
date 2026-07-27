"""매수시점 분류.

기업의 질·내재가치 등급과 실제 진입 시점을 분리한다. 가격이 하락하면
밸류에이션 점수는 기계적으로 좋아질 수 있으므로, 주봉 추세·시장 대비
상대강도·최근 저점 안정·재무 둔화를 함께 확인한다.

이 모듈의 결과는 매수 지시가 아니라 '추가 확인이 필요한 단계'를 표시하는
규칙 기반 보조지표다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _finite(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _series(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if not len(s):
        return s.astype(float)
    try:
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
    except (TypeError, ValueError):
        pass
    return s.astype(float)


def _return(s: pd.Series, weeks: int):
    if len(s) <= weeks or s.iloc[-weeks - 1] <= 0:
        return None
    return _finite(s.iloc[-1] / s.iloc[-weeks - 1] - 1)


def _growth(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2 or s.iloc[-2] <= 0:
        return None
    return _finite(s.iloc[-1] / s.iloc[-2] - 1)


def _annual_col(annual: pd.DataFrame | None, col: str) -> pd.Series:
    if annual is None or col not in annual.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(annual[col], errors="coerce").dropna()


def _fundamental_check(annual, ttm, roic_res, risk_codes) -> dict:
    revenue = _annual_col(annual, "revenue")
    fcf = _annual_col(annual, "fcf")
    rev_growth = _growth(revenue)
    fcf_growth = _growth(fcf)

    ttm_rev_change = ttm_fcf_change = None
    if ttm and len(revenue) and revenue.iloc[-1] > 0:
        ttm_rev = _finite(ttm.get("revenue"))
        if ttm_rev is not None:
            ttm_rev_change = _finite(ttm_rev / revenue.iloc[-1] - 1)
    if ttm and len(fcf) and fcf.iloc[-1] > 0:
        ttm_fcf = _finite(ttm.get("fcf"))
        if ttm_fcf is not None:
            ttm_fcf_change = _finite(ttm_fcf / fcf.iloc[-1] - 1)

    roic_trend = None
    if roic_res:
        roic_trend = _finite(roic_res.get("summary", {}).get("trend"))

    cautions = []
    severe = []
    if rev_growth is not None and rev_growth <= -0.05:
        severe.append(f"최근 연간 매출 {rev_growth:.1%}")
    elif rev_growth is not None and rev_growth < 0:
        cautions.append(f"최근 연간 매출 {rev_growth:.1%}")

    if fcf_growth is not None and fcf_growth <= -0.30:
        cautions.append(f"최근 연간 FCF {fcf_growth:.1%}")
    if len(fcf) and fcf.iloc[-1] <= 0:
        severe.append("최근 연간 FCF 적자")

    # TTM과 최근 회계연도는 기간이 상당 부분 겹친다. 따라서 확정 YoY가 아닌
    # '최신 흐름 경고'로만 사용하고 단독으로 강제 매수보류시키지 않는다.
    if ttm_rev_change is not None and ttm_rev_change <= -0.05:
        cautions.append(f"TTM 매출이 최근 FY보다 {ttm_rev_change:.1%}")
    if ttm_fcf_change is not None and ttm_fcf_change <= -0.25:
        cautions.append(f"TTM FCF가 최근 FY보다 {ttm_fcf_change:.1%}")

    if roic_trend is not None and roic_trend <= -0.015:
        cautions.append(f"ROIC 추세 연 {roic_trend*100:+.1f}%p")

    codes = set(risk_codes or ())
    if "FCF_NEG_2Y" in codes:
        severe.append("FCF 2년 연속 적자")
    if "ROIC_3Y_DOWN" in codes:
        cautions.append("ROIC 3년 하락")
    if "NI_FCF_DIVERGE" in codes:
        cautions.append("순이익·FCF 괴리")

    # 서로 다른 둔화 신호가 겹칠 때만 펀더멘털 보류로 올린다.
    weak = bool(severe) or len(cautions) >= 2
    level = "악화 경고" if weak else "주의" if cautions else "양호"
    return {
        "level": level,
        "weak": weak,
        "cautions": severe + cautions,
        "rev_growth": rev_growth,
        "fcf_growth": fcf_growth,
        "ttm_rev_change": ttm_rev_change,
        "ttm_fcf_change": ttm_fcf_change,
        "roic_trend": roic_trend,
    }


def _trend_check(price_history, benchmark_history, current_price) -> dict:
    s = _series(price_history)
    if len(s) < 41:
        return {
            "available": False,
            "label": "가격이력 부족",
            "score": None,
            "reasons": [f"주봉 {len(s)}개 — 최소 41개 필요"],
        }

    price = _finite(current_price)
    # 현재가와 최근 주봉이 합리적인 범위 안이면 마지막 관측치에 반영한다.
    if price is not None and s.iloc[-1] > 0 and 0.75 <= price / s.iloc[-1] <= 1.25:
        s.iloc[-1] = price
    price = float(s.iloc[-1])

    ma13s = s.rolling(13).mean()
    ma40s = s.rolling(40).mean()
    ma13, ma40 = _finite(ma13s.iloc[-1]), _finite(ma40s.iloc[-1])
    ma13_old = _finite(ma13s.iloc[-5]) if len(ma13s.dropna()) >= 5 else None
    ma40_old = _finite(ma40s.iloc[-5]) if len(ma40s.dropna()) >= 5 else None
    ma13_slope = _finite(ma13 / ma13_old - 1) if ma13 and ma13_old else None
    ma40_slope = _finite(ma40 / ma40_old - 1) if ma40 and ma40_old else None

    ret13, ret26 = _return(s, 13), _return(s, 26)
    window13 = s.iloc[-13:]
    low13, high52 = float(window13.min()), float(s.iloc[-52:].max())
    weeks_since_low = len(window13) - 1 - int(np.argmin(window13.values))
    new_low = bool(price <= low13 * 1.005)
    rebound_low = _finite(price / low13 - 1) if low13 > 0 else None
    drawdown52 = _finite(price / high52 - 1) if high52 > 0 else None

    bench = _series(benchmark_history)
    bench_ret13 = bench_ret26 = rel13 = rel26 = None
    if len(bench) >= 27:
        bench_ret13, bench_ret26 = _return(bench, 13), _return(bench, 26)
        if ret13 is not None and bench_ret13 is not None:
            rel13 = ret13 - bench_ret13
        if ret26 is not None and bench_ret26 is not None:
            rel26 = ret26 - bench_ret26

    above13 = bool(ma13 and price >= ma13)
    above40 = bool(ma40 and price >= ma40)
    short_up = bool(ma13_slope is not None and ma13_slope > 0)
    long_up = bool(ma40_slope is not None and ma40_slope >= 0)
    active_downtrend = bool(
        not above13 and ma13_slope is not None and ma13_slope < 0
        and ret13 is not None and ret13 < 0
    )
    deep_downtrend = bool(
        active_downtrend and not above40
        and ma40_slope is not None and ma40_slope < 0
    )
    stabilized = bool(
        not new_low and weeks_since_low >= 3
        and rebound_low is not None and rebound_low >= 0.03
        and (ma13_slope is None or ma13_slope >= -0.01)
    )
    turning = bool(above13 and short_up and not new_low)
    confirmed = bool(above13 and above40 and short_up and long_up
                     and (rel13 is None or rel13 >= 0))

    score = 0
    score += 20 if above13 else 0
    score += 20 if above40 else 0
    score += 15 if short_up else 7 if ma13_slope is not None and ma13_slope >= -0.01 else 0
    score += 10 if long_up else 0
    score += 10 if ret13 is not None and ret13 > 0 else 0
    score += 10 if rel13 is None or rel13 >= 0 else 0
    score += 10 if not new_low else 0
    score += 5 if weeks_since_low >= 3 else 0

    if deep_downtrend:
        label = "장기 하락"
    elif active_downtrend:
        label = "하락 진행"
    elif confirmed:
        label = "상승 확인"
    elif turning:
        label = "추세 전환"
    elif stabilized:
        label = "바닥 확인 중"
    else:
        label = "횡보·혼조"

    reasons = [
        f"현재가 {'13주선 위' if above13 else '13주선 아래'}",
        f"현재가 {'40주선 위' if above40 else '40주선 아래'}",
        f"13주 수익률 {ret13:.1%}" if ret13 is not None else "13주 수익률 N/A",
    ]
    if rel13 is not None:
        reasons.append(f"시장 대비 13주 상대수익 {rel13:+.1%}")
    reasons.append(
        "최근 13주 신저가" if new_low
        else f"13주 저점 후 {weeks_since_low}주 경과"
    )

    return {
        "available": True,
        "label": label,
        "score": score,
        "price": price,
        "ma13": ma13,
        "ma40": ma40,
        "ma13_slope": ma13_slope,
        "ma40_slope": ma40_slope,
        "ret13": ret13,
        "ret26": ret26,
        "bench_ret13": bench_ret13,
        "bench_ret26": bench_ret26,
        "relative13": rel13,
        "relative26": rel26,
        "drawdown52": drawdown52,
        "weeks_since_low13": weeks_since_low,
        "rebound_from_low13": rebound_low,
        "new_low13": new_low,
        "active_downtrend": active_downtrend,
        "deep_downtrend": deep_downtrend,
        "stabilized": stabilized,
        "turning": turning,
        "confirmed": confirmed,
        "reasons": reasons,
    }


def _valuation_check(price, fair_value, mos_target, val_norm) -> dict:
    price, fair_value = _finite(price), _finite(fair_value)
    mos = None
    if price is not None and fair_value is not None and fair_value > 0:
        available = True
        mos = _finite((fair_value - price) / fair_value)
        attractive = mos >= mos_target
        near = mos >= max(mos_target - 0.10, 0)
        expensive = mos < 0
        reason = f"안전마진 {mos:.1%} (목표 {mos_target:.0%})"
    else:
        v = _finite(val_norm)
        available = v is not None
        attractive = bool(v is not None and v >= 70)
        near = bool(v is not None and v >= 45)
        expensive = bool(v is not None and v < 35)
        reason = "적정가치 N/A — 밸류에이션 점수로 대체" if v is not None \
            else "밸류에이션 산출 불가"
    return {
        "available": available,
        "mos": mos,
        "attractive": attractive,
        "near": near,
        "expensive": expensive,
        "reason": reason,
    }


def classify_buy_timing(
    *,
    price_history,
    benchmark_history=None,
    current_price=None,
    fair_value=None,
    mos_target=0.30,
    quality_norm=None,
    val_norm=None,
    annual=None,
    ttm=None,
    roic_res=None,
    risk_codes=None,
) -> dict:
    """가격·가치·재무를 결합한 매수시점 단계 분류."""
    trend = _trend_check(price_history, benchmark_history, current_price)
    valuation = _valuation_check(current_price, fair_value, mos_target, val_norm)
    fundamental = _fundamental_check(annual, ttm, roic_res, risk_codes)
    quality = _finite(quality_norm)

    code = "HOLD"
    label = "➖ 판단 보류"
    tone = "info"
    summary = "가격이력 또는 핵심 점수가 부족합니다."

    if quality is None or not trend["available"]:
        pass
    elif quality < 60:
        code, label, tone = "HOLD_QUALITY", "🔴 매수 보류", "error"
        summary = "기업의 질 점수가 낮아 진입 시점보다 사업의 질 재검토가 우선입니다."
    elif fundamental["weak"]:
        code, label, tone = "HOLD_FUNDAMENTAL", "🔴 펀더멘털 확인 필요", "error"
        summary = "최근 재무 둔화 신호가 겹쳐 다음 실적 확인 전 매수를 보류합니다."
    elif trend["active_downtrend"]:
        code, label, tone = "WAIT_TREND", "🟠 하락 추세 대기", "warning"
        summary = ("가격 매력은 있어도 하락 추세가 진행 중입니다. "
                   "13주선 회복과 저점 안정이 먼저입니다.")
    elif not valuation["available"]:
        code, label, tone = "HOLD", "➖ 판단 보류", "info"
        summary = "적정가치와 밸류에이션 점수가 없어 현재 가격의 매력도를 판단할 수 없습니다."
    elif valuation["expensive"] or not valuation["near"]:
        code, label, tone = "WAIT_PRICE", "⚪ 가격 부담 관망", "info"
        summary = "추세보다 목표 안전마진을 충족할 가격 조정이 먼저입니다."
    elif trend["confirmed"] and valuation["attractive"]:
        code, label, tone = "ACCUMULATE", "🟢 분할매수 검토", "success"
        summary = "목표 안전마진과 중·장기 추세가 함께 확인됐습니다."
    elif (trend["turning"] or trend["stabilized"]) and valuation["attractive"]:
        code, label, tone = "BASE_CONFIRMING", "🟡 바닥 확인 중", "warning"
        summary = ("가격 매력은 확보됐고 단기 안정 신호가 나타났습니다. "
                   "40주선 또는 시장 대비 상대강도 확인이 남았습니다.")
    elif trend["confirmed"] and valuation["near"]:
        code, label, tone = "WATCH_ENTRY", "🟡 관심 가격대", "warning"
        summary = "추세는 확인됐지만 목표 안전마진에는 아직 부족합니다."
    else:
        code, label, tone = "WAIT_CONFIRMATION", "🟡 추가 확인 대기", "warning"
        summary = "가격·추세 조건이 동시에 충족되지 않았습니다."

    checkpoints = []
    if trend.get("active_downtrend"):
        checkpoints.append("주가가 13주 이동평균선을 회복")
    if trend.get("new_low13"):
        checkpoints.append("최소 3주 동안 13주 신저가 미갱신")
    if trend.get("relative13") is not None and trend["relative13"] < 0:
        checkpoints.append("13주 시장 대비 상대수익률이 0 이상으로 전환")
    if not valuation["attractive"]:
        checkpoints.append(valuation["reason"])
    if fundamental["cautions"]:
        checkpoints.append("다음 실적에서 " + ", ".join(fundamental["cautions"][:2]) + " 확인")
    if not checkpoints:
        checkpoints.append("분할 접근 후 다음 분기 가이던스 유지 여부 확인")

    return {
        "code": code,
        "label": label,
        "tone": tone,
        "summary": summary,
        "trend": trend,
        "valuation": valuation,
        "fundamental": fundamental,
        "checkpoints": checkpoints,
    }

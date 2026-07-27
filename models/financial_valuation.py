"""금융업 전용 밸류에이션 — 초과수익모형(Excess Return).

은행·보험·BDC는 부채가 조달수단이 아니라 원재료이므로 FCF·투하자본·EV·WACC가
모두 정의되지 않는다. 따라서 DCF 대신 자기자본 기준 모형을 쓴다.

    적정 PBR = (ROE − g) / (Ke − g)
    적정주가 = 적정 PBR × BPS
    초과수익 = (ROE − Ke) × 자기자본

ROE가 Ke를 지속 상회하지 못하면 적정 PBR은 1배 미만이 되는 것이 정상이다.
데이터가 부족하거나 가정이 발산하는 구간에서는 임의값을 넣지 않고 N/A로 둔다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from analysis.roic import _f, _g


def detect_subtype(fd) -> str:
    """industry/sector 문자열로 금융업 세부 분류를 추정."""
    hay = f"{fd.industry or ''} {fd.sector or ''} {fd.name or ''}".lower()
    for label, keys in settings.FINANCIAL_SUBTYPES:
        if any(k in hay for k in keys):
            return label
    return settings.FINANCIAL_SUBTYPE_DEFAULT


def _series(annual: pd.DataFrame) -> pd.DataFrame:
    """연도별 ROE·ROA·배당성향 테이블."""
    eq = _g(annual, "equity")
    ni = _g(annual, "net_income")
    ta = _g(annual, "total_assets")
    div = _g(annual, "dividends_out")

    eq_pos = eq.where(eq > 0)
    t = pd.DataFrame({
        "equity": eq,
        "net_income": ni,
        "total_assets": ta,
        "roe": ni / eq_pos,
        "roa": ni / ta.where(ta > 0),
        "payout": (div / ni.where(ni > 0)).clip(0, 1.5),
    })
    return t


def compute_financial_valuation(fd, annual: pd.DataFrame, ke: float | None,
                                price: float | None) -> dict:
    """금융회사 밸류에이션. 반환 dict는 UI가 그대로 렌더링할 수 있는 형태."""
    out = {
        "subtype": detect_subtype(fd),
        "table": None, "summary": {}, "flags": [], "notes": [],
        "applicable": False,
    }
    t = _series(annual)
    out["table"] = t

    roe = t["roe"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(roe) == 0:
        out["flags"].append("자기자본·순이익 데이터 부족 — ROE 산출 불가")
        return out

    n = settings.FIN_ROE_YEARS
    roe_recent = roe.iloc[-n:]
    roe_med = float(roe_recent.median())
    roe_avg = float(roe_recent.mean())
    roe_latest = float(roe.iloc[-1])
    roe_std = float(roe_recent.std()) if len(roe_recent) > 1 else None

    roa = t["roa"].replace([np.inf, -np.inf], np.nan).dropna()
    payout = t["payout"].replace([np.inf, -np.inf], np.nan).dropna()
    payout_avg = float(payout.iloc[-n:].mean()) if len(payout) else None

    # 지속가능 성장률 g = ROE × 사내유보율
    retention = 1 - (payout_avg if payout_avg is not None else 0.35)
    retention = float(np.clip(retention, 0.0, 1.0))
    g_raw = roe_med * retention
    g = float(np.clip(g_raw, 0.0, settings.FIN_G_CAP))
    if payout_avg is None:
        out["notes"].append("배당성향 데이터 부재 — 사내유보율 65%로 가정")

    summary = {
        "years": int(len(roe)), "roe_med": roe_med, "roe_avg": roe_avg,
        "roe_latest": roe_latest, "roe_std": roe_std,
        "roa_avg": float(roa.iloc[-n:].mean()) if len(roa) else None,
        "payout_avg": payout_avg, "g_used": g, "ke": ke,
        "bps": None, "pbr_now": None, "pb_fair": None, "fair": None,
        "upside": None, "mos": None, "excess_return": None,
        "spread": (roe_med - ke) if ke is not None else None,
    }

    eq_latest = _f(t["equity"].dropna().iloc[-1]) if t["equity"].notna().any() else None
    shares = _f(fd.shares)
    if eq_latest and shares and shares > 0 and eq_latest > 0:
        summary["bps"] = eq_latest / shares
        if price:
            summary["pbr_now"] = price / summary["bps"]
    if eq_latest and ke is not None:
        summary["excess_return"] = (roe_med - ke) * eq_latest

    # ── 적정 PBR ────────────────────────────────────────────────────────────
    if ke is None:
        out["flags"].append("자기자본비용(Ke) 산출 불가 — 적정 PBR 계산 생략")
    elif ke - g < settings.FIN_KE_GAP_MIN:
        g_adj = ke - settings.FIN_KE_GAP_MIN
        if g_adj <= 0:
            out["flags"].append(
                f"Ke({ke:.2%})가 너무 낮아 적정 PBR이 발산 — 계산 생략")
        else:
            out["notes"].append(
                f"성장률이 Ke에 근접해 g를 {g:.2%}→{g_adj:.2%}로 하향 조정")
            g = g_adj
            summary["g_used"] = g

    if ke is not None and ke - g >= settings.FIN_KE_GAP_MIN:
        pb = (roe_med - g) / (ke - g)
        if not np.isfinite(pb) or pb <= 0:
            out["flags"].append(
                f"지속가능 ROE({roe_med:.1%})가 성장률({g:.1%}) 이하 — "
                f"적정 PBR 산출 불가(가치파괴 구간)")
        elif pb > settings.FIN_PB_SANE_MAX:
            out["flags"].append(
                f"적정 PBR {pb:.1f}배로 비현실적 — ROE 가정 재확인 필요, N/A 처리")
        else:
            summary["pb_fair"] = float(pb)
            if summary["bps"]:
                fair = pb * summary["bps"]
                summary["fair"] = fair
                if price:
                    summary["upside"] = price and (fair / price - 1)
                    summary["mos"] = (fair - price) / fair if fair else None
            out["applicable"] = True

    # ── 해석 플래그 ─────────────────────────────────────────────────────────
    if ke is not None:
        if roe_med < ke:
            out["flags"].append(
                f"지속가능 ROE {roe_med:.1%} < Ke {ke:.1%} — 자기자본 대비 "
                f"가치를 파괴하는 구간입니다. PBR 1배 미만이 정상이며, "
                f"'PBR이 낮으니 싸다'는 판단은 성립하지 않습니다.")
        elif roe_med >= settings.FIN_ROE_EXCELLENT:
            out["notes"].append(
                f"지속가능 ROE {roe_med:.1%} — 버핏식 은행 기준선(15%) 충족")
        elif roe_med >= settings.FIN_ROE_GOOD:
            out["notes"].append(f"지속가능 ROE {roe_med:.1%} — 양호(12% 이상)")

    if summary["roa_avg"] is not None and out["subtype"] == "은행":
        if summary["roa_avg"] >= settings.FIN_ROA_GOOD:
            out["notes"].append(
                f"ROA {summary['roa_avg']:.2%} — 은행 기준선(1.3%) 충족")
        else:
            out["flags"].append(
                f"ROA {summary['roa_avg']:.2%} — 은행 기준선(1.3%) 미달. "
                f"높은 ROE가 레버리지에서 나온 것은 아닌지 확인 필요.")

    if roe_std is not None and roe_med and roe_std > abs(roe_med) * 0.5:
        out["flags"].append(
            f"ROE 변동성 σ {roe_std:.1%}로 평균 대비 큼 — 중앙값 기반 "
            f"적정 PBR의 신뢰도가 낮습니다.")

    if summary["years"] < settings.MIN_YEARS:
        out["flags"].append(
            f"ROE 데이터 {summary['years']}년치뿐 — 사이클 전체를 담지 못했을 "
            f"가능성이 높습니다(대손비용은 사이클 후반에 집중 발생).")

    out["notes"].append(
        "Yahoo 공개 데이터로는 CET1·NIM·NPL·대손비용률·합산비율을 조회할 수 "
        "없습니다. 이 지표들 없이는 금융회사 평가가 불완전하므로 "
        "감독당국 공시(DART/EDGAR)를 별도 확인하세요.")

    out["summary"] = summary
    return out

"""간이 경제적 해자 채점 — 마진 지속성 기반 (경쟁사 비교 미포함).

정식 해자 평가(명세 4단계)는 경쟁사 대비 마진·점유율 데이터가 필요하다.
본 모듈은 EDGAR/DART 연동으로 확보된 장기 이력(12~20년)에서 계산 가능한
절반 — '마진 지속성' — 만 채점한다. 넓은 해자의 재무적 흔적은
  (1) 높은 마진 수준
  (2) 사이클을 관통하는 마진 안정성
  (3) 침식 없는 추세
로 나타난다는 버핏/멍거식 관찰을 정량화한 것이다.

한계 — 반드시 인지할 것:
  · 경쟁사 비교가 없어 '업계 전체가 고마진'인 경우를 구분하지 못한다.
  · 마진은 해자의 결과이지 원인이 아니다(전환비용·네트워크효과·브랜드 등
    해자의 원천은 판별하지 않는다).
  · 따라서 점수 근거에 '간이'를 명시하고, 정식 4단계 구현 전까지의
    근사치로만 쓴다. 8년 미만 이력은 채점하지 않는다(사이클 미포함 위험).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.roic import _g

MIN_YEARS = 8
W_LEVEL, W_PERSIST, W_TREND = 0.40, 0.35, 0.25


def compute_moat(annual: pd.DataFrame) -> dict:
    """연간 재무 → {'score_frac'(0~1|None), 'summary', 'reasons', 'flags'}."""
    out = {"score_frac": None, "summary": {}, "reasons": [], "flags": []}

    rev = _g(annual, "revenue")
    op = _g(annual, "operating_income")
    opm = (op / rev.where(rev != 0)).replace([np.inf, -np.inf], np.nan).dropna()
    years = int(len(opm))
    if years < MIN_YEARS:
        out["flags"].append(
            f"영업이익률 이력 {years}년 — 간이 해자 채점에 최소 {MIN_YEARS}년 "
            f"필요(사이클 미포함 위험)")
        return out

    med = float(opm.median())
    std = float(opm.std())
    loss_years = int((opm < 0).sum())

    # ① 마진 수준 — 구조적 수익성. 20% 이상이면 가격결정력 강한 구간.
    level = 0.0 if med <= 0 else float(np.interp(
        med, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        [0.0, 0.20, 0.50, 0.75, 0.95, 1.00]))

    # ② 지속성 — 변동계수(σ/중앙값). 사이클을 관통해 마진이 유지되는가.
    #    적자 연도가 있으면 상한 0.4 — 손실을 내는 해자는 없다.
    cv = std / abs(med) if med else float("inf")
    persist = float(np.interp(cv, [0.10, 0.20, 0.35, 0.60, 1.00],
                              [1.00, 0.85, 0.60, 0.30, 0.10]))
    if loss_years:
        persist = min(persist, 0.40)

    # ③ 추세 — 초기 3년 대비 최근 3년 중앙값. 침식되는 해자는 감점.
    k = min(3, years // 2)
    trend_pp = float(opm.iloc[-k:].median() - opm.iloc[:k].median())
    trend = float(np.interp(trend_pp, [-0.06, -0.03, 0.00, 0.02],
                            [0.00, 0.35, 0.70, 1.00]))

    frac = W_LEVEL * level + W_PERSIST * persist + W_TREND * trend
    # 적자 이력 상한 — 손실을 낸 적 있는 해자는 넓지 않다(버핏 기준).
    # 호황기 고마진으로 중앙값이 높아도 이 상한이 커머디티를 걸러낸다.
    if loss_years >= 3:
        frac = min(frac, 0.30)
    elif loss_years >= 1:
        frac = min(frac, 0.50)

    gp = _g(annual, "gross_profit")
    gpm = (gp / rev.where(rev != 0)).replace([np.inf, -np.inf], np.nan).dropna()
    gpm_med = float(gpm.median()) if len(gpm) >= MIN_YEARS else None

    out["summary"] = dict(
        years=years, opm_med=med, opm_std=std, cv=cv, loss_years=loss_years,
        trend_pp=trend_pp, gpm_med=gpm_med,
        level=level, persist=persist, trend=trend, frac=frac)

    loss_txt = f" · 적자 {loss_years}회" if loss_years else ""
    gpm_txt = f" · GPM {gpm_med:.0%}" if gpm_med is not None else ""
    out["reasons"] = [
        f"간이(마진 지속성 {years}년): 영업이익률 중앙값 {med:.1%} · "
        f"σ {std * 100:.1f}%p · 추세 {trend_pp * 100:+.1f}%p{loss_txt}{gpm_txt}"
        f" — 경쟁사 비교 미포함, 정식 4단계 아님"]
    out["score_frac"] = float(np.clip(frac, 0.0, 1.0))
    return out

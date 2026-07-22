"""수집 데이터 검증 — 명세 4.2(기간)·18(예외) 대응 경고 생성."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from data.data_fetcher import FinancialData, CRITICAL_FIELDS, _g


HOLDING_ASSET_SHARE = 0.30      # 지분법·장기지분투자 / 총자산 임계치
HOLDING_NAME_KEYWORDS = ("지주", "홀딩스")   # 한국 지주사 명명 관례(강한 신호)


def is_holding(fd: FinancialData) -> bool:
    """지주회사/투자회사 휴리스틱 감지.

    ① 지분법·장기지분투자 자산이 총자산의 30% 이상, 또는
    ② (한국 종목) 회사명에 '지주'/'홀딩스' 포함.
    영문 'Holdings'는 일반 사업회사 상호에도 흔해 명칭만으로는 판정하지 않는다.
    """
    df = fd.annual
    try:
        inv = _g(df, "equity_investments").dropna()
        ta = _g(df, "total_assets").dropna()
        if len(inv) and len(ta):
            y = inv.index.intersection(ta.index)
            if len(y):
                share = float(inv.loc[y[-1]]) / float(ta.loc[y[-1]]) \
                    if ta.loc[y[-1]] else 0.0
                if share >= HOLDING_ASSET_SHARE:
                    return True
    except Exception:
        pass
    name = (fd.name or "")
    if fd.country == "KR" and any(k in name for k in HOLDING_NAME_KEYWORDS):
        return True
    return False


def validate(fd: FinancialData) -> tuple[list[dict], bool]:
    """(경고 목록[{level,msg}], 데이터부족_감점_여부) 반환.

    level: "error"(신뢰도 심각) | "warn"(주의) | "info"
    """
    out: list[dict] = []
    df = fd.annual
    years = int(_g(df, "revenue").notna().sum())

    shortage = years < settings.MIN_YEARS
    if shortage:
        out.append({"level": "error",
                    "msg": f"연간 데이터 {years}개년 — 최소 {settings.MIN_YEARS}개년 미만으로 "
                           f"분석 신뢰도가 낮습니다(점수 −5점 반영). 10개년 분석은 5단계 "
                           f"SEC/OpenDART 연동에서 지원됩니다."})
    elif years < settings.TARGET_YEARS:
        out.append({"level": "warn",
                    "msg": f"연간 데이터 {years}개년 — 목표 {settings.TARGET_YEARS}개년 대비 짧아 "
                           f"장기 지표(10년 평균 등)는 가용 기간 기준으로 계산됩니다."})

    latest = df.iloc[-1] if len(df) else pd.Series(dtype=float)
    missing = [f for f in CRITICAL_FIELDS
               if f not in df.columns or pd.isna(latest.get(f, np.nan))]
    if missing:
        out.append({"level": "warn",
                    "msg": "최근 연도 핵심 항목 누락: " + ", ".join(missing) +
                           " → 해당 지표는 N/A 처리"})

    if fd.currency and fd.fin_currency and fd.currency != fd.fin_currency:
        out.append({"level": "warn",
                    "msg": f"주가 통화({fd.currency})와 재무제표 통화({fd.fin_currency}) 불일치 — "
                           f"ADR 가능성. 밸류에이션 배수는 환율 왜곡에 유의하세요."})

    eq = _g(df, "equity")
    if len(eq.dropna()) and (eq.dropna() <= 0).any():
        out.append({"level": "warn",
                    "msg": "자기자본이 0 이하인 연도 존재 — ROE·부채비율 등 일부 지표 N/A 처리"})

    if is_holding(fd):
        out.append({"level": "warn",
                    "msg": "지주회사/투자회사 특성 감지 — 가치의 상당 부분이 자회사·피투자회사 "
                           "지분에 있어 ROIC·DCF 기반 점수가 왜곡될 수 있습니다. "
                           "순자산가치(NAV)·자회사 지분가치 기준 평가를 병행하세요. "
                           "(감지 근거: 지분법·장기지분투자 자산 비중 또는 지주사 명칭 — 휴리스틱)"})

    if fd.is_financial:
        out.append({"level": "error",
                    "msg": "금융회사로 감지됨 — 일반기업 ROIC·부채 모델이 부적절합니다. "
                           "ROE 중심 참고 지표만 제한적으로 표시하며 점수 신뢰도는 낮습니다. "
                           "(전용 모델은 5단계)"})

    for m in fd.messages:
        out.append({"level": "info", "msg": m})

    return out, shortage

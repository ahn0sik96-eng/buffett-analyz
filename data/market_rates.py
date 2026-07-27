"""무위험수익률(10년물) 자동 조회.

FRED의 무인증 CSV 엔드포인트를 사용한다(API 키 불필요).
  US : DGS10           — 미 국채 10년물, 일별
  KR : IRLTLT01KRM156N — 한국 장기국채 수익률(OECD), 월별(1~2개월 시차)

조회 실패·이상값·네트워크 차단 상황에서는 예외를 밖으로 던지지 않고
settings.DEFAULT_RF 폴백값과 사유 문자열을 함께 반환한다 —
가격 산출이 조용히 틀린 값으로 진행되지 않도록 UI가 출처를 항상 표시한다.
"""
from __future__ import annotations

import io

import pandas as pd

from config import settings

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={cosd}"
_UA = {"User-Agent": "buffett-analyzer/1.4 (personal research)"}

LOOKBACK_DAYS = 400        # 최신값만 필요하므로 최근 구간만 요청
HTTP_TIMEOUT = 12.0
HTTP_RETRIES = 2


def _fred_latest(series_id: str, timeout: float = HTTP_TIMEOUT) -> tuple[float, str]:
    """FRED 시계열의 최신 유효 관측치를 (소수 비율, 날짜) 로 반환.

    cosd(관측 시작일)를 붙여 최근 구간만 받는다 — 파라미터 없이 요청하면
    DGS10은 1962년부터 전체 시계열(수만 행)을 내려주어 ReadTimeout이 난다.
    """
    import datetime as dt

    import requests  # yfinance 의존성으로 이미 설치됨

    cosd = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    url = FRED_CSV.format(sid=series_id, cosd=cosd)

    last_err: Exception | None = None
    r = None
    for _ in range(HTTP_RETRIES):
        try:
            r = requests.get(url, timeout=timeout, headers=_UA)
            r.raise_for_status()
            break
        except Exception as e:
            last_err, r = e, None
            continue
    if r is None:
        raise last_err or RuntimeError("요청 실패")

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or df.shape[1] < 2:
        raise ValueError("빈 응답")
    date_col, val_col = df.columns[0], df.columns[-1]
    # FRED는 결측을 "." 으로 표기 — 숫자 변환 후 제거
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    if df.empty:
        raise ValueError("유효 관측치 없음")
    last = df.iloc[-1]
    return float(last[val_col]) / 100.0, str(last[date_col])[:10]


def fetch_rf(country: str) -> dict:
    """국가코드 → {'rf', 'label', 'auto'} .

    auto=False면 폴백값이며, label에 사유가 들어간다.
    """
    country = country if country in settings.DEFAULT_RF else "OTHER"
    fallback = settings.DEFAULT_RF[country]
    sid = settings.RF_SOURCES.get(country)

    if not settings.RF_AUTO_ENABLED or not sid:
        return {"rf": fallback, "auto": False, "reason": "none",
                "label": f"기본값 {fallback:.2%} — 이 시장은 자동 조회 대상이 "
                         f"아닙니다. 국고채 10년물 최신값을 직접 입력하세요."}

    lo, hi = settings.RF_SANE_RANGE
    try:
        v, d = _fred_latest(sid)
        if not (lo < v < hi):
            raise ValueError(f"이상값 {v:.4f}")
        note = " · 월별 시계열이라 최대 2개월 시차" if country == "KR" else ""
        return {"rf": v, "auto": True, "reason": "ok",
                "label": f"FRED {sid} · {d} 기준 {v:.2%}{note}"}
    except Exception as e:                      # 네트워크·포맷·이상값 모두 폴백
        return {"rf": fallback, "auto": False, "reason": "fail",
                "label": f"자동 조회 실패({type(e).__name__}) — "
                         f"기본값 {fallback:.2%} 사용, 직접 확인 권장"}

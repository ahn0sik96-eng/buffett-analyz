"""전역 설정값. UI에서 대부분 재정의 가능."""

APP_VERSION = "v1.4 (2026-07-27) · 금리자동·국가별가정·금융업모델"

TARGET_YEARS = 10          # 목표 분석 기간
MIN_YEARS = 5              # 이 미만이면 신뢰도 경고 + 데이터 부족 감점

# ── 무위험수익률 ────────────────────────────────────────────────────────────
# 자동 조회(FRED) 실패 시 쓰는 폴백값. 자동 조회가 켜져 있으면 UI가 최신값으로 덮어씀.
DEFAULT_RF = {"US": 0.046, "KR": 0.044, "OTHER": 0.042}

# FRED 시계열 ID (무인증 CSV 엔드포인트 사용)
#   DGS10           : 미 국채 10년물, 일별
#   IRLTLT01KRM156N : 한국 장기국채 수익률(OECD), 월별 — 1~2개월 시차 있음
RF_AUTO_ENABLED = True
RF_SOURCES = {"US": "DGS10", "KR": "IRLTLT01KRM156N"}
RF_SANE_RANGE = (0.001, 0.20)   # 이 범위를 벗어난 조회값은 폐기하고 폴백 사용
RF_CACHE_TTL = 6 * 3600

# ── 주식위험프리미엄 ────────────────────────────────────────────────────────
DEFAULT_ERP = 0.050        # 하위호환용 스칼라 폴백
# 국가별 기본 ERP — 한국은 지배구조·유동성 디스카운트를 반영해 상향
DEFAULT_ERP_BY_COUNTRY = {"US": 0.045, "KR": 0.060, "OTHER": 0.055}

# 유효세율 계산 불가 시 사용할 대체 세율
# KR 0.264 = 법인세 최고세율 24% + 지방소득세 2.4%p
TAX_FALLBACK = {"US": 0.24, "KR": 0.264, "OTHER": 0.25}

EFF_TAX_MIN, EFF_TAX_MAX = 0.0, 0.45
BETA_MIN, BETA_MAX, BETA_DEFAULT = 0.4, 2.5, 1.0
KD_SPREAD_MAX = 0.06       # 부채비용 상한 = rf + 6%p
KD_FALLBACK_SPREAD = 0.015
WACC_FLOOR = 0.05

DCF_YEARS = 5
DEFAULT_TERMINAL_G = 0.025          # 하위호환용 스칼라 폴백
# 국가별 영구성장률 — 장기 명목 GDP 성장률 상한 룰. 한국은 인구구조 반영해 하향.
DEFAULT_TERMINAL_G_BY_COUNTRY = {"US": 0.025, "KR": 0.018, "OTHER": 0.020}
TERMINAL_GAP_MIN = 0.015   # WACC - g 최소 간격
SCENARIO_G_DELTA = 0.03    # 보수/낙관 성장률 가감
SCENARIO_WACC_DELTA = 0.01
TV_SHARE_WARN = 0.75       # 터미널밸류 비중 경고 임계값

DEFAULT_MOS_TARGET = 0.30  # 목표 안전마진
# 예측가능성에 연동한 안전마진 프리셋 (이익 변동성이 클수록 높게)
MOS_PRESETS = {
    "필수소비재·저변동 (25%)": 0.25,
    "일반 (30%)": 0.30,
    "시클리컬·반도체 (45%)": 0.45,
    "직접 입력": None,
}
DEFAULT_MOS_PRESET = "일반 (30%)"

# 한국 6자리 종목코드 → 야후 티커 접미사 시도 순서
KR_SUFFIXES = [".KS", ".KQ"]
# 종목명 간이 매핑(대표 종목만). 전체 이름검색은 5단계 OpenDART에서 지원.
KR_NAME_MAP = {
    "삼성전자": "005930", "SK하이닉스": "000660", "네이버": "035420",
    "카카오": "035720", "현대차": "005380", "기아": "000270",
    "LG에너지솔루션": "373220", "삼성바이오로직스": "207940",
    "셀트리온": "068270", "POSCO홀딩스": "005490",
    "KB금융": "105560", "신한지주": "055550", "리노공업": "058470",
}

# ── SEC EDGAR (미국 재무제표 장기 이력) ─────────────────────────────────────
SEC_ENABLED = True
SEC_MIN_YEARS = 5          # 이보다 적게 나오면 야후 데이터를 유지
# EDGAR는 User-Agent에 실제 연락처를 요구한다(없으면 403).
# 코드에 직접 적지 말고 Streamlit Secrets에 SEC_USER_AGENT로 넣을 것.
SEC_USER_AGENT_FALLBACK = ""


def sec_user_agent() -> str:
    """Streamlit Secrets → 환경변수 → 폴백 순으로 EDGAR User-Agent를 찾는다."""
    import os
    try:
        import streamlit as st
        v = st.secrets.get("SEC_USER_AGENT", "")
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get("SEC_USER_AGENT", "") or SEC_USER_AGENT_FALLBACK


FINANCIAL_SECTORS = {"Financial Services", "Financial"}
FINANCIAL_KEYWORDS = ("bank", "insurance", "capital markets", "credit")

# ── 금융업 세부 분류 (industry 문자열 소문자 매칭, 위에서부터 우선) ──────────
FINANCIAL_SUBTYPES = [
    ("은행", ("bank", "thrift", "savings", "diversified banks")),
    ("보험", ("insurance", "insurer", "reinsurance")),
    ("BDC·대체신용", ("asset management", "credit services")),
    ("증권·자본시장", ("capital markets", "brokerage", "financial data")),
]
FINANCIAL_SUBTYPE_DEFAULT = "기타 금융"

# 초과수익모형(Excess Return) 파라미터
FIN_ROE_YEARS = 5              # 지속가능 ROE 산출에 쓸 최근 연수
FIN_ROE_EXCELLENT = 0.15       # 버핏식 은행 기준선(우수)
FIN_ROE_GOOD = 0.12
FIN_ROA_GOOD = 0.013           # 은행 ROA 1.3%
FIN_KE_GAP_MIN = 0.015         # Ke − g 최소 간격 (적정 PBR 발산 방지)
FIN_G_CAP = 0.06               # 금융업 지속가능 성장률 상한
FIN_PB_SANE_MAX = 6.0          # 적정 PBR 상한(이를 넘으면 N/A 처리)

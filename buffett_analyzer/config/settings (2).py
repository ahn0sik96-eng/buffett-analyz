"""전역 설정값. UI에서 대부분 재정의 가능."""

TARGET_YEARS = 10          # 목표 분석 기간
MIN_YEARS = 5              # 이 미만이면 신뢰도 경고 + 데이터 부족 감점

# 무위험수익률 기본값(10년물, 실행 시 사용자가 최신값으로 수정 권장)
DEFAULT_RF = {"US": 0.043, "KR": 0.033, "OTHER": 0.040}
DEFAULT_ERP = 0.050        # 주식위험프리미엄
# 유효세율 계산 불가 시 사용할 대체 세율
TAX_FALLBACK = {"US": 0.24, "KR": 0.26, "OTHER": 0.25}

EFF_TAX_MIN, EFF_TAX_MAX = 0.0, 0.45
BETA_MIN, BETA_MAX, BETA_DEFAULT = 0.4, 2.5, 1.0
KD_SPREAD_MAX = 0.06       # 부채비용 상한 = rf + 6%p
KD_FALLBACK_SPREAD = 0.015
WACC_FLOOR = 0.05

DCF_YEARS = 5
DEFAULT_TERMINAL_G = 0.025
TERMINAL_GAP_MIN = 0.015   # WACC - g 최소 간격
SCENARIO_G_DELTA = 0.03    # 보수/낙관 성장률 가감
SCENARIO_WACC_DELTA = 0.01

DEFAULT_MOS_TARGET = 0.30  # 목표 안전마진

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

FINANCIAL_SECTORS = {"Financial Services", "Financial"}
FINANCIAL_KEYWORDS = ("bank", "insurance", "capital markets", "credit")

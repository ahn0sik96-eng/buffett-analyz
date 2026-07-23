"""배점·등급·매수구간 규칙 (명세 9·12장)."""

WEIGHTS = {
    "roic": 20, "fcf": 15, "reinvestment": 15, "moat": 15,
    "debt": 10, "cyclicality": 10, "valuation": 15,
}
QUALITY_KEYS = ["roic", "fcf", "reinvestment", "moat", "debt", "cyclicality"]

# 4단계 이후 구현 예정 → MVP에서는 항상 N/A(미채점, 환산 제외)
NOT_IMPLEMENTED = {"moat": "4단계(경쟁사·마진 지속성 데이터 필요)",
                   "cyclicality": "4단계(거시지표 상관 분석 필요)"}

GRADE_BANDS = [
    (90, "S급 장기 복리 기업"),
    (80, "A급 우량기업"),
    (70, "B급 조건부 우량기업"),
    (60, "C급 관찰 대상"),
    (50, "D급 투자 주의"),
    (0,  "회피 권고"),
]

PENALTY_CAP = -10          # 경고 기반 감점 합계 하한
DATA_SHORTAGE_PENALTY = -5 # 5개년 미만

# 매수가격 구간 (명세 12장) — 적정가치 F 대비 배수
PRICE_ZONES = [
    ("강력 매수 검토", 0.00, 0.70),
    ("매수 검토",     0.70, 0.80),
    ("적정(할인)",    0.80, 0.90),
    ("관망",          0.90, 1.10),
    ("고평가",        1.10, 1.30),
    ("과열",          1.30, 99.0),
]

def grade_of(score: float) -> str:
    for cut, name in GRADE_BANDS:
        if score >= cut:
            return name
    return GRADE_BANDS[-1][1]

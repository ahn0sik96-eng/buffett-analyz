"""경고 기반 감점 (명세 9장 가감점 + 10장 경고와 연동).

플래그 감점 합계는 −10점 하한, 데이터 부족(5개년 미만)은 별도 −5점.
"""
from __future__ import annotations

from config.scoring_rules import PENALTY_CAP, DATA_SHORTAGE_PENALTY

PENALTY_DEFS = {
    "NI_FCF_DIVERGE":   ("순이익 증가·FCF 감소 괴리", -2.0),
    "AR_FASTER":        ("매출채권이 매출보다 빠르게 증가", -1.5),
    "INV_FASTER":       ("재고자산이 매출보다 빠르게 증가", -1.5),
    "SHARES_RISING":    ("주식수 지속 증가(희석)", -1.5),
    "SBC_HEAVY":        ("주식보상이 FCF의 과도한 비중", -2.0),
    "ROIC_3Y_DOWN":     ("ROIC 3년 연속 하락", -2.0),
    "FCF_NEG_2Y":       ("FCF 적자 2년 이상 지속", -3.0),
    "INC_ROIC_LT_WACC": ("증분 ROIC < WACC", -2.0),
    "BUYBACK_DEBT":     ("부채 조달 자사주매입 의심", -2.0),
    "PAYOUT_GT_OCF":    ("주주환원이 영업현금흐름 초과", -1.5),
    "ICOV_LOW":         ("이자보상배율 3배 미만", -1.5),
    "GOODWILL_HEAVY":   ("영업권·무형자산 과다", -1.0),
}


def collect(risk_codes: set[str], data_shortage: bool):
    items = [(PENALTY_DEFS[c][0], PENALTY_DEFS[c][1])
             for c in sorted(risk_codes) if c in PENALTY_DEFS]
    flag_total = max(sum(p for _, p in items), PENALTY_CAP)
    data_pen = DATA_SHORTAGE_PENALTY if data_shortage else 0
    if data_pen:
        items.append(("데이터 부족(5개년 미만)", data_pen))
    return flag_total + data_pen, items

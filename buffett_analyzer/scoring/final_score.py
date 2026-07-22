"""최종 점수 집계·등급·분류 (명세 2·9장).

미구현/산출불가 항목은 N/A로 두고 가용 배점 기준으로 100점 환산한다
(부분평가임을 명시 — 허위 정확성 방지). 기업의 질 점수와 밸류에이션 점수는
분리해 표시한다.
"""
from __future__ import annotations

from config.scoring_rules import WEIGHTS, QUALITY_KEYS, NOT_IMPLEMENTED, grade_of


def aggregate(components: dict[str, tuple[float | None, int, list[str]]],
              penalty_total: float) -> dict:
    """components: {키: (점수|None, 만점, 근거)} — moat/cyclicality는 (None, w, [사유])."""
    avail = sum(w for _, (pts, w, _d) in components.items() if pts is not None)
    achieved = sum(pts for pts, _w, _d in components.values() if pts is not None)

    q_avail = sum(w for k, (pts, w, _d) in components.items()
                  if k in QUALITY_KEYS and pts is not None)
    q_pts = sum(pts for k, (pts, _w, _d) in components.items()
                if k in QUALITY_KEYS and pts is not None)
    v = components.get("valuation", (None, WEIGHTS["valuation"], []))

    quality_norm = round(100 * q_pts / q_avail, 1) if q_avail else None
    val_norm = round(100 * v[0] / v[1], 1) if v[0] is not None else None

    raw = max(achieved + penalty_total, 0.0)
    total_norm = round(min(100 * raw / avail, 100), 1) if avail else None

    missing = [k for k, (pts, _w, _d) in components.items() if pts is None]
    partial_note = None
    if missing:
        labels = [f"{k}({NOT_IMPLEMENTED.get(k, '데이터 부족')})" for k in missing]
        partial_note = "미채점 항목 제외 후 환산: " + ", ".join(labels)

    return {
        "achieved": round(achieved, 1), "available": avail,
        "penalty": round(penalty_total, 1),
        "total_norm": total_norm,
        "grade": grade_of(total_norm) if total_norm is not None else "N/A",
        "quality_norm": quality_norm, "val_norm": val_norm,
        "partial_note": partial_note,
    }


def classify(quality_norm, val_norm, debt_component, z_zone,
             is_financial: bool, years: int) -> tuple[str, str]:
    """명세 2장의 7분류 + 금융회사 예외. (분류, 근거) 반환 — 프로그램의 추론."""
    if is_financial:
        return ("금융회사 — 일반 모델 부적합",
                "ROE·자본비율 중심 별도 평가 필요(5단계). 아래 점수는 참고용.")
    debt_ratio = (debt_component[0] / debt_component[1]) if (
        debt_component and debt_component[0] is not None) else None
    if (debt_ratio is not None and debt_ratio < 0.35) or z_zone == "부실 위험":
        return ("재무 위험 기업", "부채 안전성 점수 미달 또는 Z-score 부실 구간.")
    if quality_norm is None or val_norm is None:
        return ("판단 보류(데이터 부족)", "핵심 축 산출 불가 — 임의 분류하지 않음.")
    if quality_norm >= 85 and val_norm >= 55:
        if years >= 7:
            return ("강력한 장기 복리 기업", "질·가격 양 축 모두 상위 구간.")
        return ("우량 기업", "질·가격 모두 양호하나 데이터 기간이 짧아 상위 등급 유보.")
    if quality_norm >= 75 and val_norm >= 45:
        return ("우량 기업", "기업의 질이 우수하고 가격 부담이 과도하지 않음.")
    if quality_norm >= 75:
        return ("좋은 기업이지만 주가가 비쌈", "질은 우수하나 현재 가격의 안전마진 부족.")
    if val_norm >= 70 and quality_norm < 55:
        return ("저평가 가능성이 있으나 기업의 질이 낮음",
                "가격 매력은 있으나 자본효율·현금창출력이 약함.")
    if quality_norm >= 60:
        return ("조건부 투자 가능", "일부 축이 보통 수준 — 약점 항목의 개선 확인 필요.")
    return ("경기민감·관찰 필요",
            "질 점수 하위 구간. 경기민감도 정량 분석(4단계) 전까지 관찰 권고.")

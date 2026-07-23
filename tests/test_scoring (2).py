import pytest

from scoring.valuation_score import score_valuation
from scoring.risk_penalties import collect
from scoring.final_score import aggregate, classify


def test_valuation_bands():
    pts, mx, _ = score_valuation(price=70.0, fair_base=100.0,
                                 fcf_yield=None, rf=0.04, per=None)
    assert (pts, mx) == (15, 15)
    pts2, _, _ = score_valuation(price=140.0, fair_base=100.0,
                                 fcf_yield=None, rf=0.04, per=None)
    assert pts2 == 2


def test_penalties_cap_and_data_shortage():
    codes = {"FCF_NEG_2Y", "ROIC_3Y_DOWN", "NI_FCF_DIVERGE", "SBC_HEAVY",
             "BUYBACK_DEBT", "SHARES_RISING", "AR_FASTER", "INV_FASTER"}
    total, items = collect(codes, data_shortage=True)
    assert total == -15                       # 플래그 −10 하한 + 데이터 −5
    assert any("데이터 부족" in i[0] for i in items)


def test_aggregate_partial_normalization():
    comps = {
        "roic": (18.0, 20, []), "fcf": (12.0, 15, []),
        "reinvestment": (None, 15, ["미채점"]), "moat": (None, 15, ["미구현"]),
        "debt": (8.0, 10, []), "cyclicality": (None, 10, ["미구현"]),
        "valuation": (9.0, 15, []),
    }
    s = aggregate(comps, penalty_total=-2.0)
    assert s["available"] == 60
    assert s["achieved"] == pytest.approx(47.0)
    assert s["total_norm"] == pytest.approx(round(100 * 45.0 / 60, 1))
    assert s["quality_norm"] == pytest.approx(round(100 * 38.0 / 45, 1))
    assert s["val_norm"] == pytest.approx(60.0)
    assert "미채점" in s["partial_note"]


def test_classify_good_but_expensive():
    label, _ = classify(quality_norm=88.0, val_norm=30.0,
                        debt_component=(8.0, 10, []), z_zone="안전",
                        is_financial=False, years=8)
    assert label == "좋은 기업이지만 주가가 비쌈"
    label2, _ = classify(90.0, 70.0, (9.0, 10, []), "안전", False, 8)
    assert label2 == "강력한 장기 복리 기업"
    label3, _ = classify(90.0, 70.0, (2.0, 10, []), "부실 위험", False, 8)
    assert label3 == "재무 위험 기업"

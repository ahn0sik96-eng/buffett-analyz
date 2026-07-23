import pytest

from models.dcf import dcf_value, per_share, reverse_dcf
from models.wacc import compute_wacc


def test_dcf_perpetuity_identity():
    # g1=0, gT=0, WACC=10% → EV = 100/0.1 = 1000 (영구연금 항등식)
    res = dcf_value(100.0, 0.10, 0.0, 0.0, years=5)
    assert res["ev"] == pytest.approx(1000.0, abs=1e-6)
    ps, err = per_share(res["ev"], net_debt=200.0, shares=10.0)
    assert err is None and ps == pytest.approx(80.0)


def test_dcf_rejects_nonpositive_fcf():
    assert dcf_value(-5.0, 0.09, 0.05, 0.02) is None
    assert dcf_value(None, 0.09, 0.05, 0.02) is None


def test_terminal_gap_clamp():
    res = dcf_value(100.0, 0.05, 0.02, 0.045)
    assert res["gT_used"] == pytest.approx(0.05 - 0.015)
    assert res["note"] is not None


def test_reverse_dcf_recovers_growth():
    base = dcf_value(100.0, 0.10, 0.06, 0.02)
    price, _ = per_share(base["ev"], 200.0, 10.0)
    r = reverse_dcf(price, 0.10, 0.02, 200.0, 10.0, 100.0)
    assert r["implied_g"] == pytest.approx(0.06, abs=1e-4)


def test_wacc():
    r = compute_wacc(market_cap=800.0, total_debt=200.0, equity_book=500.0,
                     beta=1.0, interest_expense=8.0, rf=0.04, erp=0.05,
                     tax=0.25)
    assert r["ke"] == pytest.approx(0.09)
    assert r["kd_pre"] == pytest.approx(0.04)          # rf 하한 클립
    assert r["wacc"] == pytest.approx(0.8 * 0.09 + 0.2 * 0.04 * 0.75)


def test_scenarios_expose_tv_share():
    from models.dcf import run_scenarios
    scen = run_scenarios(100.0, 0.10, 0.05, 0.02, net_debt=0.0,
                         shares=10.0, price=90.0)
    for sc in scen.values():
        assert sc["tv_share"] is not None and 0.4 < sc["tv_share"] < 0.95


def test_sanity_filter_blocks_adr_currency_bug():
    from models.dcf import sanity_filter
    # TSM 사례 재현: TWD 미환산으로 적정가치가 현재가의 10배로 왜곡된 상황
    scen = {"보수적": {"fair": 3000.0, "upside": 6.0, "mos": 0.86},
            "기준":   {"fair": 4206.87, "upside": 8.9, "mos": 0.90},
            "낙관적": {"fair": 5200.0, "upside": 11.2, "mos": 0.92}}
    out, msg = sanity_filter(scen, price=424.61, fx_adjusted=True)
    assert msg is not None and "ADR" in msg
    assert all(sc["fair"] is None for sc in out.values())


def test_sanity_filter_passes_normal_case():
    from models.dcf import sanity_filter
    scen = {"기준": {"fair": 150.0, "upside": 0.1, "mos": 0.1}}
    out, msg = sanity_filter(scen, price=156.0, fx_adjusted=True)
    assert msg is None and out["기준"]["fair"] == 150.0
    # fx_adjusted=False면 검사 자체를 하지 않음
    scen2 = {"기준": {"fair": 4000.0, "upside": 9, "mos": 0.9}}
    out2, msg2 = sanity_filter(scen2, price=400.0, fx_adjusted=False)
    assert msg2 is None and out2["기준"]["fair"] == 4000.0

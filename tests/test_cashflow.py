import pandas as pd
import pytest

from analysis.cashflow import compute_cashflow, cagr


def test_fcf_conversion_margin_cagr():
    df = pd.DataFrame({
        "revenue":    {2021: 900.0, 2022: 950.0, 2023: 1000.0},
        "net_income": {2021: 90.0,  2022: 95.0,  2023: 100.0},
        "ocf":        {2021: 180.0, 2022: 195.0, 2023: 200.0},
        "capex_out":  {2021: 80.0,  2022: 85.0,  2023: 79.0},
        "fcf":        {2021: 100.0, 2022: 110.0, 2023: 121.0},
        "sbc_out":    {2021: 5.0,   2022: 5.0,   2023: 5.0},
        "shares_out": {2021: 10.0,  2022: 10.0,  2023: 10.0},
    })
    res = compute_cashflow(df)
    s = res["summary"]
    assert s["cagr_max"] == pytest.approx(0.10, abs=1e-9)      # 100→121, 2년
    assert res["table"].loc[2023, "conversion"] == pytest.approx(1.21)
    assert res["table"].loc[2023, "fcf_margin"] == pytest.approx(0.121)
    assert s["neg_count"] == 0
    assert res["table"].loc[2023, "fcf_adj"] == pytest.approx(116.0)


def test_cagr_guards():
    assert cagr(pd.Series({2022: -10.0, 2023: 120.0})) is None
    assert cagr(pd.Series({2023: 100.0})) is None


def test_share_change_uses_recent_comparable_period():
    """Old pre-split facts must not manufacture dilution (AAPL-type case)."""
    years = list(range(2007, 2026))
    shares = [0.9] * 13 + [17.5, 16.9, 16.3, 15.8, 15.3, 14.9]
    df = pd.DataFrame(index=years)
    df["revenue"] = 100.0
    df["net_income"] = 10.0
    df["ocf"] = 20.0
    df["capex_out"] = 5.0
    df["fcf"] = 15.0
    df["diluted_shares"] = shares

    s = compute_cashflow(df)["summary"]
    assert s["share_period"] == "2020–2025"
    assert s["share_change"] == pytest.approx(14.9 / 17.5 - 1)


def test_recent_split_break_does_not_trigger_dilution_penalty():
    df = pd.DataFrame({
        "revenue": {2021: 100, 2022: 100, 2023: 100, 2024: 100, 2025: 100},
        "net_income": {2021: 10, 2022: 10, 2023: 10, 2024: 10, 2025: 10},
        "ocf": {2021: 20, 2022: 20, 2023: 20, 2024: 20, 2025: 20},
        "capex_out": {2021: 5, 2022: 5, 2023: 5, 2024: 5, 2025: 5},
        "fcf": {2021: 15, 2022: 15, 2023: 15, 2024: 15, 2025: 15},
        "shares_out": {2021: 100, 2022: 100, 2023: 400, 2024: 390, 2025: 380},
    })
    res = compute_cashflow(df)
    assert res["summary"]["share_change"] == pytest.approx(380 / 400 - 1)
    assert "SHARES_RISING" not in res["risk_codes"]

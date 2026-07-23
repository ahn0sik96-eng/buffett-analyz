import pandas as pd
import pytest

from analysis.roic import compute_roic


def sample():
    return pd.DataFrame({
        "revenue":            {2022: 950.0, 2023: 1000.0},
        "ebit":               {2022: 140.0, 2023: 150.0},
        "pretax_income":      {2022: 130.0, 2023: 140.0},
        "tax_provision":      {2022: 26.0,  2023: 28.0},
        "total_assets":       {2022: 950.0, 2023: 1000.0},
        "cash":               {2022: 90.0,  2023: 100.0},
        "current_liabilities": {2022: 190.0, 2023: 200.0},
        "current_debt":       {2022: 40.0,  2023: 50.0},
        "equity":             {2022: 470.0, 2023: 500.0},
        "total_debt":         {2022: 330.0, 2023: 350.0},
    })


def test_nopat_ic_roic():
    res = compute_roic(sample(), tax_fallback=0.30, wacc=0.08)
    t = res["table"]
    # 2023: 유효세율 28/140=0.20 → NOPAT 120, IC(A)=1000-100-(200-50)=750
    assert t.loc[2023, "eff_tax"] == pytest.approx(0.20)
    assert t.loc[2023, "nopat"] == pytest.approx(120.0)
    assert t.loc[2023, "ic_a"] == pytest.approx(750.0)
    assert t.loc[2023, "ic_b"] == pytest.approx(750.0)   # 500+350-100
    assert t.loc[2023, "roic"] == pytest.approx(0.16)
    assert t.loc[2022, "roic"] == pytest.approx(112.0 / 710.0)
    s = res["summary"]
    assert s["mean_all"] == pytest.approx((0.16 + 112.0 / 710.0) / 2)
    assert s["spread_wacc"] == pytest.approx(s["mean_all"] - 0.08)


def test_negative_ic_is_na():
    df = sample()
    df.loc[2023, "total_assets"] = 100.0   # IC(A) 음수 유도
    res = compute_roic(df, ic_method="A")
    assert pd.isna(res["table"].loc[2023, "roic"])

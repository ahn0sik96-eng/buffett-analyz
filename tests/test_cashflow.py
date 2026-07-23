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

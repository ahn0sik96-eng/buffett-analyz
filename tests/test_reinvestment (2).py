import pandas as pd
import pytest

from analysis.roic import compute_roic
from analysis.reinvestment import compute_reinvestment
from tests.test_roic import sample


def test_incremental_roic_and_rr():
    df = sample()
    df["capex_out"] =        pd.Series({2022: 60.0, 2023: 70.0})
    df["depreciation"] =     pd.Series({2022: 30.0, 2023: 32.0})
    df["working_capital"] =  pd.Series({2022: 100.0, 2023: 112.0})
    df["acquisitions_out"] = pd.Series({2022: 0.0, 2023: 0.0})
    df["net_income"] =       pd.Series({2022: 95.0, 2023: 100.0})
    df["dividends_out"] =    pd.Series({2022: 20.0, 2023: 20.0})
    df["buybacks_out"] =     pd.Series({2022: 10.0, 2023: 10.0})
    roic_res = compute_roic(df, wacc=0.08)
    res = compute_reinvestment(df, roic_res, wacc=0.08)
    s = res["summary"]
    # 2023 RR1 = (70-32+12+0)/120 = 50/120
    assert res["table"].loc[2023, "rr_capex"] == pytest.approx(50.0 / 120.0)
    # 1년 증분 ROIC = (120-112)/(750-710) = 0.20
    assert s["inc_roic"][1] == pytest.approx(0.20)
    assert s["quadrant"] is not None

import sys, types

try:
    import yfinance  # noqa: F401
except ImportError:
    sys.modules["yfinance"] = types.ModuleType("yfinance")

import pandas as pd



def test_normalize_converts_money_not_shares():
    import data.data_fetcher as dfm
    original = dfm._fx_rate
    dfm._fx_rate = lambda fin, cur: 0.031   # 1 TWD = 0.031 USD
    try:
        annual = pd.DataFrame({
            "revenue": {2023: 1000.0}, "fcf": {2023: 200.0},
            "shares_out": {2023: 25_930_000_000.0},
        })
        msgs = []
        out, ttm = dfm._normalize_currency(annual, {"fcf": 200.0}, "TWD", "USD", msgs)
    finally:
        dfm._fx_rate = original
    assert out.loc[2023, "revenue"] == 31.0
    assert out.loc[2023, "fcf"] == 6.2
    assert out.loc[2023, "shares_out"] == 25_930_000_000.0   # 주식수는 미환산
    assert ttm["fcf"] == 6.2
    assert any("TWD" in m and "USD" in m for m in msgs)


def test_normalize_noop_when_same_currency():
    from data.data_fetcher import _normalize_currency
    annual = pd.DataFrame({"revenue": {2023: 1000.0}})
    out, ttm = _normalize_currency(annual, {"fcf": 1.0}, "USD", "USD", [])
    assert out.loc[2023, "revenue"] == 1000.0
    assert ttm["fcf"] == 1.0

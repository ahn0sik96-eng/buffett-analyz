import numpy as np
import pandas as pd

from analysis.buy_timing import classify_buy_timing


INDEX = pd.date_range("2025-01-03", periods=60, freq="W-FRI")
BENCHMARK = pd.Series(np.linspace(90.0, 110.0, 60), index=INDEX)
HEALTHY_ANNUAL = pd.DataFrame(
    {"revenue": [100.0, 105.0, 110.0], "fcf": [15.0, 16.0, 18.0]},
    index=[2023, 2024, 2025],
)


def classify(prices, *, fair_value, quality=90.0, annual=HEALTHY_ANNUAL):
    history = pd.Series(prices, index=INDEX[:len(prices)])
    return classify_buy_timing(
        price_history=history,
        benchmark_history=BENCHMARK,
        current_price=float(prices[-1]),
        fair_value=fair_value,
        mos_target=0.30,
        quality_norm=quality,
        val_norm=80.0,
        annual=annual,
    )


def test_confirmed_uptrend_and_safety_margin_is_accumulate():
    result = classify(np.linspace(60.0, 120.0, 60), fair_value=180.0)

    assert result["code"] == "ACCUMULATE"
    assert result["trend"]["label"] == "상승 확인"
    assert result["trend"]["relative13"] > 0
    assert result["valuation"]["attractive"] is True


def test_active_downtrend_overrides_cheap_valuation():
    result = classify(np.linspace(120.0, 60.0, 60), fair_value=120.0)

    assert result["code"] == "WAIT_TREND"
    assert result["trend"]["deep_downtrend"] is True
    assert result["valuation"]["attractive"] is True
    assert any("13주 이동평균선" in x for x in result["checkpoints"])


def test_confirmed_uptrend_but_expensive_price_is_wait_price():
    result = classify(np.linspace(60.0, 120.0, 60), fair_value=100.0)

    assert result["code"] == "WAIT_PRICE"
    assert result["valuation"]["expensive"] is True


def test_confirmed_uptrend_near_target_is_watch_entry():
    result = classify(np.linspace(60.0, 120.0, 60), fair_value=160.0)

    assert result["code"] == "WATCH_ENTRY"
    assert result["valuation"]["near"] is True
    assert result["valuation"]["attractive"] is False


def test_fundamental_deterioration_overrides_price_setup():
    weak = pd.DataFrame(
        {"revenue": [100.0, 90.0], "fcf": [15.0, 10.0]},
        index=[2024, 2025],
    )
    result = classify(
        np.linspace(60.0, 120.0, 60), fair_value=180.0, annual=weak,
    )

    assert result["code"] == "HOLD_FUNDAMENTAL"
    assert result["fundamental"]["weak"] is True


def test_low_quality_is_hold_even_with_good_price_trend():
    result = classify(
        np.linspace(60.0, 120.0, 60), fair_value=180.0, quality=55.0,
    )

    assert result["code"] == "HOLD_QUALITY"


def test_short_price_history_is_explicitly_undetermined():
    prices = np.linspace(80.0, 100.0, 20)
    result = classify(prices, fair_value=150.0)

    assert result["code"] == "HOLD"
    assert result["trend"]["available"] is False
    assert result["trend"]["label"] == "가격이력 부족"


def test_missing_valuation_is_not_mislabeled_as_expensive():
    history = pd.Series(np.linspace(60.0, 120.0, 60), index=INDEX)
    result = classify_buy_timing(
        price_history=history,
        benchmark_history=BENCHMARK,
        current_price=120.0,
        fair_value=None,
        quality_norm=90.0,
        val_norm=None,
        annual=HEALTHY_ANNUAL,
    )

    assert result["code"] == "HOLD"
    assert result["valuation"]["available"] is False
    assert "판단할 수 없습니다" in result["summary"]

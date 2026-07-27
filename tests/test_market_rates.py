from data import market_rates


def test_us_rate_uses_yahoo_when_fred_fails(monkeypatch):
    def fred_fail(*_args, **_kwargs):
        raise TimeoutError("blocked")

    monkeypatch.setattr(market_rates, "_fred_latest", fred_fail)
    monkeypatch.setattr(
        market_rates, "_yahoo_us10y",
        lambda: (0.0432, "2026-07-27"),
    )
    out = market_rates.fetch_rf("US")
    assert out["auto"] is True
    assert out["rf"] == 0.0432
    assert "Yahoo Finance" in out["label"]


def test_rate_falls_back_when_both_sources_fail(monkeypatch):
    def fail(*_args, **_kwargs):
        raise TimeoutError("blocked")

    monkeypatch.setattr(market_rates, "_fred_latest", fail)
    monkeypatch.setattr(market_rates, "_yahoo_us10y", fail)
    out = market_rates.fetch_rf("US")
    assert out["auto"] is False
    assert out["reason"] == "fail"

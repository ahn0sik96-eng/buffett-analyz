import pandas as pd
import pytest

from analysis_pipeline import pick_fcf0


def test_default_fcf_base_auto_normalizes_cyclical_company():
    df = pd.DataFrame(index=range(2016, 2026))
    df["revenue"] = 100.0
    df["fcf"] = [10, 20, 5, 18, 7, 22, 6, 19, 8, 30]
    cyc = {"summary": {"level": "경기 민감"}}

    value, label = pick_fcf0(
        df, {"fcf_ttm": 35.0}, "3년 중앙값", cyc
    )
    assert value == pytest.approx(14.0)
    assert "경기민감 자동 정규화" in label

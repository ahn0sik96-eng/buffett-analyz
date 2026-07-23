import sys, types

try:
    import yfinance  # noqa: F401  (사용자 환경)
except ImportError:  # 오프라인 테스트 환경용 스텁
    sys.modules["yfinance"] = types.ModuleType("yfinance")

import pandas as pd

from data.data_validator import is_holding


def _fd(name="테스트", country="US", inv=None, ta=1000.0):
    annual = pd.DataFrame({"total_assets": {2023: ta}})
    if inv is not None:
        annual["equity_investments"] = pd.Series({2023: inv})
    return types.SimpleNamespace(annual=annual, name=name, country=country)


def test_holding_by_asset_share():
    assert is_holding(_fd(inv=500.0)) is True          # 50% ≥ 30%
    assert is_holding(_fd(inv=50.0)) is False          # 5%


def test_holding_by_kr_name():
    assert is_holding(_fd(name="OO금융지주", country="KR")) is True
    assert is_holding(_fd(name="OO홀딩스", country="KR")) is True
    assert is_holding(_fd(name="Acme Holdings", country="US")) is False  # 영문명 단독 불충분

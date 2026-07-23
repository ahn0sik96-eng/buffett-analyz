"""ui/mobile.py 단위 검증 + metric_row 배치 로직 확인 (streamlit 목업)."""
import sys, types

# streamlit 목업
fake = types.ModuleType("streamlit")
_calls = {"columns": [], "metric": [], "markdown": 0}

class _Col:
    def metric(self, label, value, help=None):
        _calls["metric"].append((label, value))

def _columns(n):
    k = n if isinstance(n, int) else len(n)
    _calls["columns"].append(k)
    return [_Col() for _ in range(k)]

fake.columns = _columns
fake.markdown = lambda *a, **k: _calls.__setitem__("markdown", _calls["markdown"] + 1)
fake.plotly_chart = lambda *a, **k: None
class _Ctx:
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Safari"}
fake.context = _Ctx()
sys.modules["streamlit"] = fake

sys.path.insert(0, ".")
from ui import mobile as mob


def test_mobile_layout_helpers():
    # 1) UA 감지

    assert mob.detect_mobile() is True, "iPhone UA는 모바일로 감지돼야 함"
    fake.context.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64) Chrome"}
    assert mob.detect_mobile() is False, "데스크톱 UA는 False"
    fake.context = None  # 헤더 접근 불가 환경
    assert mob.detect_mobile() is False, "예외 시 안전하게 False"

    # 2) metric_row 배치
    items = [(f"L{i}", f"V{i}") for i in range(6)]
    _calls["columns"].clear()
    mob.metric_row(items, mobile=True)          # 2칸씩 → 3줄
    assert _calls["columns"] == [2, 2, 2], _calls["columns"]

    _calls["columns"].clear()
    mob.metric_row(items, mobile=False)         # 6칸 1줄
    assert _calls["columns"] == [6], _calls["columns"]

    _calls["columns"].clear()
    mob.metric_row(items[:4], mobile=False, per_row_desktop=4)
    assert _calls["columns"] == [4]

    # 3) 홀수 개수 처리 (마지막 줄이 1개)
    _calls["columns"].clear()
    mob.metric_row(items[:3], mobile=True)
    assert _calls["columns"] == [2, 1], _calls["columns"]

    # 4) help 파라미터 있는 튜플도 처리
    _calls["metric"].clear()
    mob.metric_row([("라벨", "값", "도움말")], mobile=True)
    assert _calls["metric"] == [("라벨", "값")]

    # 5) 차트 높이 조정
    class _Fig:
        def __init__(self): self.layout = {}
        def update_layout(self, **kw): self.layout.update(kw)
        def update_xaxes(self, **kw): pass
        def update_yaxes(self, **kw): pass
    f = _Fig()
    mob.chart(f, mobile=True)
    assert f.layout["height"] == 280, f.layout
    f2 = _Fig()
    mob.chart(f2, mobile=False)
    assert "height" not in f2.layout, "데스크톱은 원래 높이 유지"

    # 6) CSS에 핵심 미디어쿼리 포함 확인
    assert "@media (max-width: 640px)" in mob.RESPONSIVE_CSS
    assert "stHorizontalBlock" in mob.RESPONSIVE_CSS
    assert "stMetricValue" in mob.RESPONSIVE_CSS



"""모바일 대응 레이아웃 유틸.

- User-Agent로 모바일 여부를 1차 자동 감지(실패 시 데스크톱 가정)
- CSS 미디어쿼리로 여백·글자·탭·표를 화면폭에 맞춰 자동 조정
- 지표(metric)를 화면에 맞는 열 수로 자동 배치
"""
from __future__ import annotations

import streamlit as st

_MOBILE_UA = ("Mobile", "Android", "iPhone", "iPad", "iPod", "Windows Phone")


def detect_mobile() -> bool:
    """요청 헤더의 User-Agent로 모바일 추정. 구버전 Streamlit이면 False."""
    try:
        ua = st.context.headers.get("User-Agent", "")
    except Exception:
        try:
            ua = st.context.headers.get("user-agent", "")
        except Exception:
            return False
    return any(k in ua for k in _MOBILE_UA)


RESPONSIVE_CSS = """
<style>
/* ── 좁은 화면(휴대폰) 대응 ───────────────────────────────── */
@media (max-width: 640px) {
  /* 본문 여백 축소 — 화면을 최대한 넓게 */
  .block-container { padding: 0.6rem 0.7rem 3rem 0.7rem !important; }

  /* 열(column)을 1줄 1개가 아니라 2개씩 배치 → 스크롤 길이 절반 */
  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.4rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 calc(50% - 0.4rem) !important;
    min-width: calc(50% - 0.4rem) !important;
    width: calc(50% - 0.4rem) !important;
  }

  /* 지표 글자 크기 축소 (숫자가 잘리지 않게) */
  [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
  [data-testid="stMetricLabel"] p { font-size: 0.72rem !important; }
  [data-testid="stMetric"] { padding: 0.2rem 0 !important; }

  /* 탭: 줄바꿈 없이 가로 스크롤 + 작은 글자 */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    scrollbar-width: none;
  }
  [data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    white-space: nowrap !important;
    padding: 0.4rem 0.6rem !important;
  }
  [data-testid="stTabs"] [data-baseweb="tab"] p { font-size: 0.82rem !important; }

  /* 표: 글자 축소해 열이 더 많이 보이게 */
  [data-testid="stDataFrame"] { font-size: 0.75rem !important; }

  /* 제목 축소 */
  h1 { font-size: 1.5rem !important; }
  h2 { font-size: 1.2rem !important; }
  h3 { font-size: 1.05rem !important; }

  /* 경고/안내 박스 여백 축소 */
  [data-testid="stAlert"] { padding: 0.55rem 0.7rem !important; }
  [data-testid="stAlert"] p { font-size: 0.85rem !important; }

  /* 사이드바 입력 요소 간격 축소 */
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }

  /* 차트가 가로로 넘치지 않게 */
  .js-plotly-plot, .plot-container { max-width: 100% !important; }
}

/* 아주 좁은 화면(360px 이하)에서는 1열로 */
@media (max-width: 360px) {
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    min-width: 100% !important;
    width: 100% !important;
  }
}
</style>
"""


def inject_css() -> None:
    st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)


def metric_row(items, mobile: bool, per_row_desktop: int = 6,
               per_row_mobile: int = 2) -> None:
    """items: (label, value) 또는 (label, value, help) 튜플 리스트."""
    n = per_row_mobile if mobile else per_row_desktop
    n = max(1, min(n, len(items)))
    for i in range(0, len(items), n):
        chunk = items[i:i + n]
        cols = st.columns(len(chunk))
        for col, it in zip(cols, chunk):
            label, value = it[0], it[1]
            helptext = it[2] if len(it) > 2 else None
            col.metric(label, value, help=helptext)


def chart(fig, mobile: bool) -> None:
    """모바일에서는 차트 높이를 줄이고 범례를 아래로."""
    if mobile:
        fig.update_layout(height=280, margin=dict(l=30, r=12, t=42, b=30),
                          legend=dict(orientation="h", y=-0.25,
                                      font=dict(size=10)),
                          title=dict(font=dict(size=13)))
        fig.update_xaxes(tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False} if mobile else None)

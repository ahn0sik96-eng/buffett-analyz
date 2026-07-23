"""실행 위치 전환용 파일 (shim).

Streamlit 앱이 이 폴더(buffett_analyzer/)의 app.py를 실행하도록 설정돼 있지만,
최신 코드는 저장소 최상위에 있습니다. 이 파일은 실행을 최상위 app.py로 넘깁니다.

- 이 폴더(구버전)를 모듈 검색 경로에서 제외해, 옛 모듈이 잘못 불려오는 것을 막습니다.
- 최상위를 최우선 경로로 지정한 뒤 최상위 app.py를 실행합니다.
"""
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))   # .../buffett_analyzer
ROOT = os.path.dirname(HERE)                        # 저장소 최상위

# 구버전 폴더가 경로에 남아 있으면 옛 모듈(config, analysis 등)이 먼저 잡힌다 → 제거
sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != HERE]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TARGET = os.path.join(ROOT, "app.py")

if not os.path.exists(TARGET):
    import streamlit as st
    st.error(
        "최상위 app.py를 찾지 못했습니다.\n\n"
        f"확인한 경로: {TARGET}\n\n"
        "저장소 최상위에 app.py가 있는지 확인해 주세요."
    )
    st.stop()

runpy.run_path(TARGET, run_name="__main__")

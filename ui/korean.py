"""한국어 조사 자동 확정 — '재무구조을(를)' 병기 표기를 앞말 받침에 맞춰
하나로 고른다.

narrative_report 등 문장 생성부는 앞말이 무엇일지 모른 채 '을(를)' 형태를
쓴다. 생성부를 일일이 고치는 대신 렌더링 직전에 이 모듈로 후처리한다.

판별 규칙:
  · 한글: 유니코드 분해로 받침 유무 판정. '(으)로'는 ㄹ 받침 예외 처리.
  · 숫자: 한국어 독음 기준(1·7·8 → ㄹ 받침, 0·3·6 → 기타 받침, 2·4·5·9 → 없음).
  · 앞이 닫는 괄호면 괄호 묶음을 건너뛰고 그 앞 글자로 판정
    — '추세(연 -0.6%p)이(가)' → '추세'의 '세'로 판정.
  · 영문·기호 등 판별 불가 문자는 병기 표기를 그대로 둔다(오교정 방지).
"""
from __future__ import annotations

import re

# 병기 표기 → (받침 있을 때, 없을 때)
_PAIRS = {
    "은(는)": ("은", "는"), "는(은)": ("은", "는"),
    "이(가)": ("이", "가"), "가(이)": ("이", "가"),
    "을(를)": ("을", "를"), "를(을)": ("을", "를"),
    "과(와)": ("과", "와"), "와(과)": ("과", "와"),
    "으로(로)": ("으로", "로"), "로(으로)": ("으로", "로"),
}
_RE = re.compile("|".join(
    re.escape(k) for k in sorted(_PAIRS, key=len, reverse=True)))

# 숫자 독음: (받침 있음, ㄹ 받침)
_DIGITS = {"0": (True, False), "1": (True, True), "2": (False, False),
           "3": (True, False), "4": (False, False), "5": (False, False),
           "6": (True, False), "7": (True, True), "8": (True, True),
           "9": (False, False)}


def _batchim(ch: str) -> tuple[bool, bool] | None:
    """문자 → (받침 유무, ㄹ 받침 여부). 판별 불가면 None."""
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        j = (o - 0xAC00) % 28
        return (j != 0, j == 8)
    return _DIGITS.get(ch)


def _prev_char(text: str, i: int) -> str | None:
    """조사 위치 i 직전의 '판정 대상 글자'. 괄호 묶음·따옴표는 건너뛴다."""
    i -= 1
    depth = 0
    while i >= 0:
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth == 0:
                return None
            depth -= 1
        elif depth == 0:
            if ch in "'\"’”":
                i -= 1
                continue
            return ch
        i -= 1
    return None


def fix_josa(text: str) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(0)
        ch = _prev_char(text, m.start())
        if ch is None:
            return key
        info = _batchim(ch)
        if info is None:                      # 영문·기호 → 그대로 둠
            return key
        has, rieul = info
        if key in ("으로(로)", "로(으로)"):
            return "으로" if (has and not rieul) else "로"
        with_b, without = _PAIRS[key]
        return with_b if has else without

    return _RE.sub(repl, text)


def fix_josa_deep(obj):
    """문자열이 들어 있는 임의 구조(dict/list/tuple)에 재귀 적용."""
    if isinstance(obj, str):
        return fix_josa(obj)
    if isinstance(obj, dict):
        return {k: fix_josa_deep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(fix_josa_deep(v) for v in obj)
    return obj

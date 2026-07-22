"""표시 포맷 유틸 — 통화 인지형 숫자 표기, N/A 일관 처리."""
from __future__ import annotations

import math


def _na(v) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def pct(v, d: int = 1) -> str:
    return "N/A" if _na(v) else f"{v*100:.{d}f}%"


def xs(v, d: int = 1) -> str:
    return "N/A" if _na(v) else f"{v:.{d}f}배"


def num(v, d: int = 2) -> str:
    return "N/A" if _na(v) else f"{v:,.{d}f}"


def money(v, currency: str | None) -> str:
    if _na(v):
        return "N/A"
    cur = (currency or "").upper()
    sign = "-" if v < 0 else ""
    a = abs(v)
    if cur == "KRW":
        if a >= 1e12:
            return f"{sign}{a/1e12:,.1f}조원"
        if a >= 1e8:
            return f"{sign}{a/1e8:,.0f}억원"
        return f"{sign}{a:,.0f}원"
    unit = {"USD": "$", "EUR": "€", "JPY": "¥"}.get(cur, "")
    if a >= 1e9:
        return f"{sign}{unit}{a/1e9:,.2f}B"
    if a >= 1e6:
        return f"{sign}{unit}{a/1e6:,.1f}M"
    return f"{sign}{unit}{a:,.0f}"


def price_fmt(v, currency: str | None) -> str:
    if _na(v):
        return "N/A"
    cur = (currency or "").upper()
    if cur == "KRW":
        return f"{v:,.0f}원"
    unit = {"USD": "$", "EUR": "€", "JPY": "¥"}.get(cur, "")
    return f"{unit}{v:,.2f}"

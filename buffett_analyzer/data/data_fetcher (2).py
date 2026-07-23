"""재무데이터 수집 (MVP 소스: Yahoo Finance).

명세 4.1의 소스 우선순위(SEC/DART 우선)는 5단계에서 sec_fetcher/dart_fetcher로
확장한다. 본 모듈은 소스 교체가 가능하도록 표준화된 FinancialData 구조를 반환한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from config import settings

# ── 야후 파이낸스 행 이름 별칭 (버전에 따라 명칭이 달라 다중 후보를 순차 탐색) ──
IS_ALIASES = {
    "revenue":          ["Total Revenue", "Operating Revenue"],
    "gross_profit":     ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "ebit":             ["EBIT"],
    "ebitda":           ["EBITDA", "Normalized EBITDA"],
    "pretax_income":    ["Pretax Income"],
    "tax_provision":    ["Tax Provision"],
    "net_income":       ["Net Income", "Net Income Common Stockholders",
                         "Net Income Continuous Operations"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "diluted_shares":   ["Diluted Average Shares", "Basic Average Shares"],
}
BS_ALIASES = {
    "total_assets":        ["Total Assets"],
    "current_assets":      ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "cash":                ["Cash Cash Equivalents And Short Term Investments",
                            "Cash And Cash Equivalents"],
    "current_debt":        ["Current Debt And Capital Lease Obligation", "Current Debt"],
    "total_debt":          ["Total Debt"],
    "long_term_debt":      ["Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
    "equity":              ["Common Stock Equity", "Stockholders Equity",
                            "Total Equity Gross Minority Interest"],
    "retained_earnings":   ["Retained Earnings"],
    "inventory":           ["Inventory"],
    "receivables":         ["Accounts Receivable", "Receivables"],
    "goodwill_intangibles": ["Goodwill And Other Intangible Assets", "Goodwill"],
    "equity_investments":  ["Investments In Other Ventures Under Equity Method",
                            "Long Term Equity Investment",
                            "Investmentsin Associatesat Cost",
                            "Investments And Advances"],
    "total_liabilities":   ["Total Liabilities Net Minority Interest"],
    "shares_out":          ["Ordinary Shares Number", "Share Issued"],
    "working_capital":     ["Working Capital"],
}
CF_ALIASES = {
    "ocf":            ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex":          ["Capital Expenditure"],
    "depreciation":   ["Depreciation And Amortization", "Depreciation Amortization Depletion",
                       "Depreciation"],
    "sbc":            ["Stock Based Compensation"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "buybacks":       ["Repurchase Of Capital Stock"],
    "acquisitions":   ["Purchase Of Business", "Net Business Purchase And Sale"],
    "fcf_reported":   ["Free Cash Flow"],
    "change_wc":      ["Change In Working Capital"],
}

CRITICAL_FIELDS = ["revenue", "ebit", "net_income", "ocf", "capex",
                   "total_assets", "equity"]


@dataclass
class FinancialData:
    ticker: str
    name: str
    currency: str | None
    fin_currency: str | None
    price: float | None
    market_cap: float | None
    shares: float | None
    beta: float | None
    sector: str | None
    industry: str | None
    trailing_pe: float | None
    forward_pe: float | None
    annual: pd.DataFrame            # index=회계연도(int, 오름차순)
    ttm: dict | None                # 최근 4개 분기 합산 흐름 항목
    price_history: pd.Series | None
    is_financial: bool
    country: str                    # "US" | "KR" | "OTHER"
    source: str = "Yahoo Finance"
    messages: list[str] = field(default_factory=list)
    fx_adjusted: bool = False        # 재무제표 통화를 주가 통화로 환산했는지 여부
    fetched_at: str = ""             # 데이터 수집 시각(캐시 기준)


# ── 내부 유틸 ────────────────────────────────────────────────────────────────
def _pick(df: pd.DataFrame | None, names: list[str]) -> pd.Series | None:
    """행 이름 후보를 순서대로 탐색해 첫 번째로 존재하는 행을 반환."""
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n]
            if isinstance(s, pd.DataFrame):     # 동일 이름 중복 행
                s = s.iloc[0]
            return pd.to_numeric(s, errors="coerce")
    return None


def _collect_annual(inc, bs, cf) -> pd.DataFrame:
    data: dict[str, dict[int, float]] = {}
    for df, amap in ((inc, IS_ALIASES), (bs, BS_ALIASES), (cf, CF_ALIASES)):
        if df is None or getattr(df, "empty", True):
            continue
        for fname, aliases in amap.items():
            s = _pick(df, aliases)
            if s is None:
                continue
            for ts, val in s.items():
                if pd.isna(val):
                    continue
                data.setdefault(fname, {})[pd.Timestamp(ts).year] = float(val)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data).sort_index()
    df.index.name = "fy"
    return _derive(df)


def _g(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, dtype=float)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """파생 항목: EBIT/EBITDA 보완, CAPEX 부호 정규화, FCF, 유출액 절대값."""
    df = df.copy()
    df["ebit"] = _g(df, "ebit").fillna(_g(df, "operating_income"))
    dep = _g(df, "depreciation")
    df["ebitda"] = _g(df, "ebitda").fillna(df["ebit"] + dep)
    df["capex_out"] = _g(df, "capex").abs()
    df["fcf"] = _g(df, "ocf") - df["capex_out"]
    for src, dst in (("dividends_paid", "dividends_out"),
                     ("buybacks", "buybacks_out"),
                     ("acquisitions", "acquisitions_out"),
                     ("sbc", "sbc_out")):
        df[dst] = _g(df, src).abs()
    return df


def _ttm(tk: yf.Ticker) -> dict | None:
    """최근 4개 분기 합산(흐름 항목). 4개 분기 미만이면 항목별 NaN."""
    try:
        qi, qc = tk.quarterly_income_stmt, tk.quarterly_cashflow
    except Exception:
        return None

    def s4(df, aliases):
        s = _pick(df, aliases)
        if s is None:
            return np.nan
        s = s.dropna().sort_index()
        return float(s.iloc[-4:].sum()) if len(s) >= 4 else np.nan

    out = {
        "revenue":    s4(qi, IS_ALIASES["revenue"]),
        "ebit":       s4(qi, IS_ALIASES["ebit"] + IS_ALIASES["operating_income"]),
        "ebitda":     s4(qi, IS_ALIASES["ebitda"]),
        "net_income": s4(qi, IS_ALIASES["net_income"]),
        "ocf":        s4(qc, CF_ALIASES["ocf"]),
        "capex_out":  abs(s4(qc, CF_ALIASES["capex"])),
        "sbc_out":    abs(s4(qc, CF_ALIASES["sbc"])),
    }
    ocf, cap = out["ocf"], out["capex_out"]
    out["fcf"] = ocf - cap if np.isfinite(ocf) and np.isfinite(cap) else np.nan
    return out if any(np.isfinite(v) for v in out.values()) else None


def _safe_info(tk: yf.Ticker) -> dict:
    try:
        return tk.get_info() or {}
    except Exception:
        try:
            return tk.info or {}
        except Exception:
            return {}


def _get_price(tk: yf.Ticker, info: dict) -> float | None:
    try:
        p = tk.fast_info["lastPrice"]
        if p and np.isfinite(float(p)):
            return float(p)
    except Exception:
        pass
    for k in ("currentPrice", "regularMarketPrice", "previousClose"):
        v = info.get(k)
        if v:
            return float(v)
    return None


def resolve_candidates(user_input: str) -> tuple[list[str], str]:
    """입력값 → 시도할 야후 티커 목록과 국가 코드."""
    t = user_input.strip()
    if t in settings.KR_NAME_MAP:
        t = settings.KR_NAME_MAP[t]
    tu = t.upper()
    if re.fullmatch(r"\d{6}", tu):                       # 한국 종목코드
        return [tu + sfx for sfx in settings.KR_SUFFIXES], "KR"
    if tu.endswith(".KS") or tu.endswith(".KQ"):
        return [tu], "KR"
    return [tu], "US" if re.fullmatch(r"[A-Z.\-]{1,10}", tu) else "OTHER"


def _fx_rate(fin_currency: str, currency: str) -> float | None:
    """fin_currency 1단위 → currency 단위 환율. 야후 FX 페어(예: TWDUSD=X)로 조회.

    ADR(TSMC 등)처럼 재무제표 통화와 주가 통화가 다른 종목의 절대금액을
    주가 기준 통화로 맞추기 위함. 조회 실패 시 None(호출측에서 미환산 처리).
    """
    if not fin_currency or not currency or fin_currency == currency:
        return 1.0
    for pair, invert in ((f"{fin_currency}{currency}=X", False),
                        (f"{currency}{fin_currency}=X", True)):
        try:
            t = yf.Ticker(pair)
            r = None
            try:
                r = t.fast_info.get("lastPrice")
            except Exception:
                pass
            if not r:
                h = t.history(period="5d")["Close"]
                r = float(h.iloc[-1]) if len(h) else None
            if r and np.isfinite(r) and r > 0:
                return float(1 / r) if invert else float(r)
        except Exception:
            continue
    return None


_SHARE_COUNT_COLS = {"shares_out", "diluted_shares"}


def _normalize_currency(annual: pd.DataFrame, ttm: dict | None,
                        fin_currency: str | None, currency: str | None,
                        msgs: list[str]) -> tuple[pd.DataFrame, dict | None]:
    """재무제표 통화를 주가 통화로 환산(주식수 등 비금액 항목은 제외)."""
    if not fin_currency or not currency or fin_currency == currency:
        return annual, ttm
    rate = _fx_rate(fin_currency, currency)
    if rate is None:
        msgs.append(f"환율 조회 실패 — 재무제표({fin_currency})와 주가({currency}) 통화가 "
                    f"달라 절대금액·적정가치 결과의 신뢰도가 낮습니다(비율 지표는 영향 없음).")
        return annual, ttm
    annual = annual.copy()
    for c in annual.columns:
        if c not in _SHARE_COUNT_COLS:
            annual[c] = annual[c] * rate
    if ttm:
        ttm = {k: (v * rate if np.isfinite(v) else v) for k, v in ttm.items()}
    msgs.append(f"재무제표를 {fin_currency}→{currency}로 환산(환율 {rate:.4g}, 조회 시점 기준). "
                f"단, ADR 환산비율(예: TSMC 1 ADR=보통주 5주 등)은 확인이 불가해 별도 조정하지 "
                f"않았습니다 — 주당 적정가치는 비정상적일 경우 자동으로 N/A 처리됩니다.")
    return annual, ttm


def fetch(user_input: str) -> FinancialData:
    candidates, country = resolve_candidates(user_input)
    last_err: Exception | None = None
    for tick in candidates:
        try:
            tk = yf.Ticker(tick)
            annual = _collect_annual(tk.income_stmt, tk.balance_sheet, tk.cashflow)
            if annual.empty or _g(annual, "revenue").notna().sum() == 0:
                raise ValueError(f"{tick}: 재무제표를 찾을 수 없음")
            info = _safe_info(tk)
            msgs: list[str] = []

            price = _get_price(tk, info)
            shares = info.get("sharesOutstanding")
            if not shares:
                so = _g(annual, "shares_out").dropna()
                shares = float(so.iloc[-1]) if len(so) else None
                if shares:
                    msgs.append("발행주식수를 재무상태표에서 대체 조회함")
            mcap = info.get("marketCap")
            if not mcap and price and shares:
                mcap = price * shares

            sector = info.get("sector")
            industry = (info.get("industry") or "")
            is_fin = (sector in settings.FINANCIAL_SECTORS) or any(
                k in industry.lower() for k in settings.FINANCIAL_KEYWORDS)

            fin_ccy = info.get("financialCurrency")
            ccy = info.get("currency")
            ttm_raw = _ttm(tk)
            fx_adjusted = bool(fin_ccy and ccy and fin_ccy != ccy)
            annual, ttm_raw = _normalize_currency(annual, ttm_raw, fin_ccy, ccy, msgs)

            try:
                hist = tk.history(period="5y", interval="1wk")["Close"]
                hist = hist if len(hist) else None
            except Exception:
                hist = None
                msgs.append("주가 이력 조회 실패(차트 생략)")

            return FinancialData(
                ticker=tick,
                name=info.get("longName") or info.get("shortName") or tick,
                currency=ccy,
                fin_currency=fin_ccy,
                price=price, market_cap=mcap,
                shares=float(shares) if shares else None,
                beta=info.get("beta"),
                sector=sector, industry=industry or None,
                trailing_pe=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                annual=annual, ttm=ttm_raw, price_history=hist,
                is_financial=bool(is_fin), country=country, messages=msgs,
                fx_adjusted=fx_adjusted,
            )
        except Exception as e:                            # 다음 후보(.KQ 등) 시도
            last_err = e
            continue
    raise ValueError(
        f"'{user_input}' 데이터 수집 실패: {last_err}. "
        "티커(예: AAPL) 또는 한국 6자리 종목코드(예: 005930)를 확인하세요."
    )

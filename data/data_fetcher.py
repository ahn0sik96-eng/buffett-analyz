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
    """야후 info. 간헐적으로 빈 dict가 오므로 1회 재시도한다."""
    import time as _t
    for attempt in range(2):
        for getter in ("get_info", "info"):
            try:
                v = getattr(tk, getter)
                v = v() if callable(v) else v
                if v:
                    return v
            except Exception:
                continue
        if attempt == 0:
            _t.sleep(0.6)
    return {}


def _usable_years(df: pd.DataFrame) -> int:
    """핵심 필드(매출·영업현금흐름·CAPEX)가 모두 있는 연도 수.

    행 수만 비교하면 '연도는 많지만 태그가 비어 항목이 전부 NaN'인
    EDGAR/DART 데이터가 야후의 짧지만 온전한 4개년을 밀어내는 사고가 난다
    — 그 경우 ROIC·FCF·재투자·해자가 한꺼번에 미채점으로 무너진다.
    교체 판단은 반드시 이 가용연수 기준으로 한다."""
    need = ["revenue", "ocf", "capex"]
    cols = [c for c in need if c in df.columns]
    if len(cols) < len(need):
        return 0
    return int(df[cols].notna().all(axis=1).sum())


def _merge_ttm(base: dict | None, override: dict | None) -> dict | None:
    """1차 소스(EDGAR/DART) TTM으로 야후 TTM을 항목별 대체.

    야후에만 있는 항목(sbc 등)은 남기고, 겹치는 항목은 1차 소스를 쓴다.
    fcf는 병합 후 재계산해 ocf·capex와의 정합을 보장한다."""
    if not override:
        return base
    out = dict(base or {})
    for k, v in override.items():
        try:
            if v is not None and np.isfinite(v):
                out[k] = v
        except TypeError:
            continue
    ocf, cap = out.get("ocf"), out.get("capex_out")
    if ocf is not None and cap is not None             and np.isfinite(ocf) and np.isfinite(cap):
        out["fcf"] = ocf - cap
    return out or None


def _carry_shares(new_df: pd.DataFrame, old_df: pd.DataFrame) -> pd.DataFrame:
    """EDGAR/DART 데이터로 교체할 때 야후에만 있는 주식수 항목을 넘겨받는다.

    DART 표준계정에는 발행주식수가 없어서, 이걸 빠뜨리면 주당 적정가치가
    통째로 N/A가 된다(야후 info까지 비면 대체 경로가 사라짐)."""
    out = new_df.copy()
    for c in _SHARE_COUNT_COLS:
        if c in old_df.columns and (c not in out.columns
                                    or out[c].isna().all()):
            out[c] = old_df[c].reindex(out.index)
    return out


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
    if re.fullmatch(r"[A-Z.\-]{1,10}", tu):
        # 클래스 주식 표기 정규화 — 언론·거래소는 'BRK.B'(점), 야후는
        # 'BRK-B'(하이픈)를 쓴다. 점 표기가 들어오면 하이픈 변형을 먼저
        # 시도한다(야후가 점 표기에 빈 재무제표를 돌려주므로).
        cands = ([tu.replace(".", "-")] if "." in tu else []) + [tu]
        seen: set = set()
        cands = [c for c in cands if not (c in seen or seen.add(c))]
        return cands, "US"
    return [tu], "OTHER"


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


def _kr_name(tick: str) -> str | None:
    """005930.KS → '삼성전자'. 야후 info가 비었을 때 이름이라도 살린다."""
    code = re.sub(r"\D", "", tick)[:6]
    for nm, c in settings.KR_NAME_MAP.items():
        if c == code:
            return nm
    return None


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
            source = "Yahoo Finance"
            ttm_override: dict | None = None
            ttm_src_msg: str | None = None

            # ── 미국 종목: EDGAR로 장기 이력 교체 시도 ──────────────────────
            # 야후는 연간 4개년뿐이라 시클리컬 종목의 정상이익 추정이 불가능하다.
            # 실패해도 야후 데이터로 계속 진행하되, 사유를 반드시 표기한다.
            if country == "US" and settings.SEC_ENABLED:
                try:
                    from data import sec_fetcher
                    sec_df, sec_msg, sec_ttm, sec_ttm_msg = \
                        sec_fetcher.fetch_annual_and_ttm(tick)
                    u_new, u_old = _usable_years(sec_df), _usable_years(annual)
                    if u_new >= max(settings.SEC_MIN_YEARS, u_old):
                        annual = _derive(_carry_shares(sec_df, annual))
                        source = "SEC EDGAR (재무) + Yahoo Finance (주가·정보)"
                        msgs.append(sec_msg)
                        if sec_ttm:
                            ttm_override, ttm_src_msg = sec_ttm, sec_ttm_msg
                    else:
                        msgs.append(
                            f"EDGAR 가용연수 {u_new}년(매출·현금흐름·CAPEX "
                            f"동시 존재 기준) ≤ 야후 {u_old}년 — 비표준 태그로 "
                            f"핵심 항목이 비어 야후 데이터를 유지했습니다.")
                except Exception as e:
                    msgs.append(f"SEC EDGAR 조회 실패({type(e).__name__}: {e}) — "
                                f"야후 {len(annual)}개년 데이터로 진행합니다.")

            # ── 한국 종목: DART로 장기 이력 교체 시도 ──────────────────────
            if country == "KR" and settings.DART_ENABLED:
                try:
                    from data import dart_fetcher
                    code = re.sub(r"\D", "", tick)[:6]
                    dart_df, dart_msg = dart_fetcher.fetch_annual(
                        code, years_back=settings.DART_YEARS_BACK)
                    u_new, u_old = _usable_years(dart_df), _usable_years(annual)
                    if u_new >= max(settings.DART_MIN_YEARS, u_old):
                        annual = _derive(_carry_shares(dart_df, annual))
                        source = "DART OpenAPI (재무) + Yahoo Finance (주가·정보)"
                        msgs.append(dart_msg)
                        try:
                            fy_last = int(dart_df.index.max())
                            ttm_override, ttm_src_msg = dart_fetcher.fetch_ttm(
                                code, fy_last, dart_df.loc[fy_last].to_dict())
                        except Exception as e:
                            msgs.append(f"DART 분기 TTM 조회 실패"
                                        f"({type(e).__name__}) — 야후 TTM 유지")
                    else:
                        msgs.append(
                            f"DART 가용연수 {u_new}년(매출·현금흐름·CAPEX "
                            f"동시 존재 기준) ≤ 야후 {u_old}년 — 핵심 계정 "
                            f"미검출로 야후 데이터를 유지했습니다.")
                except Exception as e:
                    msgs.append(f"DART 조회 실패({type(e).__name__}: {e}) — "
                                f"야후 {len(annual)}개년 데이터로 진행합니다.")

            price = _get_price(tk, info)
            shares = info.get("sharesOutstanding")
            if not shares:
                try:
                    shares = tk.fast_info.get("shares")
                except Exception:
                    shares = None
            if not shares:
                so = _g(annual, "shares_out").dropna()
                shares = float(so.iloc[-1]) if len(so) else None
                if shares:
                    msgs.append("발행주식수를 재무상태표에서 대체 조회함")
            mcap = info.get("marketCap")
            if not mcap and price and shares:
                mcap = price * shares

            if not info:
                msgs.append("야후 종목정보(섹터·통화·발행주식수) 조회 실패 — "
                            "일시적 장애일 수 있습니다. 사이드바의 '데이터 캐시 "
                            "지우기' 후 재시도해 보세요.")
            sector = info.get("sector")
            industry = (info.get("industry") or "")
            is_fin = (sector in settings.FINANCIAL_SECTORS) or any(
                k in industry.lower() for k in settings.FINANCIAL_KEYWORDS)

            fin_ccy = info.get("financialCurrency")
            ccy = info.get("currency")
            if not ccy:
                try:
                    ccy = tk.fast_info.get("currency")
                except Exception:
                    ccy = None
            if not ccy:
                ccy = "KRW" if tick.endswith((".KS", ".KQ")) else None
            if not fin_ccy:
                fin_ccy = ccy
            ttm_raw = _merge_ttm(_ttm(tk), ttm_override)
            if ttm_src_msg:
                msgs.append(ttm_src_msg)
            fx_adjusted = bool(fin_ccy and ccy and fin_ccy != ccy)
            annual, ttm_raw = _normalize_currency(annual, ttm_raw, fin_ccy, ccy, msgs)

            try:
                hist = tk.history(period="5y", interval="1wk")["Close"]
                hist = hist if len(hist) else None
            except Exception:
                hist = None
                msgs.append("주가 이력 조회 실패(차트 생략)")

            # ── 시세 정합 가드 ──────────────────────────────────────────
            # 야후 쿼트는 액면분할·무상증자 후 조정을 놓치고 스테일 값을
            # 돌려주는 경우가 있다(예: 리노공업 5:1 분할). 이력(차트 API)은
            # 분할 조정이 되므로, 현재가가 최근 종가의 2배를 벗어나면
            # 쿼트를 버리고 최근 종가로 대체한다 — 틀린 가격으로 안전마진을
            # 계산하는 것보다 1주 이내 종가가 항상 낫다.
            if price and hist is not None and len(hist):
                try:
                    _last = float(hist.dropna().iloc[-1])
                except Exception:
                    _last = None
                if _last and _last > 0:
                    _ratio = price / _last
                    if not (0.5 <= _ratio <= 2.0):
                        msgs.append(
                            f"⚠️ 야후 현재가({price:,.0f})가 최근 종가"
                            f"({_last:,.0f})의 {_ratio:.1f}배 — 액면분할·"
                            f"무상증자 등 기업행동 미반영 스테일 쿼트로 보고 "
                            f"현재가를 최근 종가로 대체했습니다. 실제 시세와 "
                            f"발행주식수를 증권사 앱에서 교차 확인하세요.")
                        price = _last
                        if shares:
                            mcap = price * shares

            return FinancialData(
                ticker=tick,
                name=(info.get("longName") or info.get("shortName")
                      or _kr_name(tick) or tick),
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
                fx_adjusted=fx_adjusted, source=source,
            )
        except Exception as e:                            # 다음 후보(.KQ 등) 시도
            last_err = e
            continue
    raise ValueError(
        f"'{user_input}' 데이터 수집 실패: {last_err}. "
        "티커(예: AAPL) 또는 한국 6자리 종목코드(예: 005930)를 확인하세요."
    )

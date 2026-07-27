"""SEC EDGAR XBRL 재무데이터 수집 (미국 상장사, 무인증).

야후는 연간 4개년만 제공하지만 EDGAR companyfacts는 2009년경부터
전체 이력을 준다. 반환 형태는 data_fetcher._collect_annual()과 동일한
(index=회계연도, columns=표준 항목명) DataFrame이라 그대로 교체 가능하다.

주의 — EDGAR 이용 규칙:
  · User-Agent 헤더에 실제 연락처(이메일)를 넣어야 한다. 없으면 403.
  · 초당 10회 제한. 본 모듈은 종목당 요청 2회(티커맵 1회는 캐시)로 설계.

XBRL 태그 정규화 문제:
  같은 '매출'도 회사마다 Revenues / RevenueFromContractWithCustomer... /
  SalesRevenueNet 등으로 다르게 태깅한다. TAGS의 후보 리스트를 순서대로
  탐색해 첫 번째로 값이 있는 태그를 채택한다.

소급수정(restatement):
  동일 회계연도가 여러 10-K에 반복 등장하므로 filed(제출일)가 가장 늦은
  값을 채택해 최신 수정치를 쓴다.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from config import settings

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# ── 표준항목 → us-gaap 태그 후보 (앞에서부터 우선) ──────────────────────────
# kind: "d"=기간(손익·현금흐름), "i"=시점(재무상태표)
TAGS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("d", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"]),
    "gross_profit": ("d", ["GrossProfit"]),
    "operating_income": ("d", ["OperatingIncomeLoss"]),
    "pretax_income": ("d", [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]),
    "tax_provision": ("d", ["IncomeTaxExpenseBenefit"]),
    "net_income": ("d", ["NetIncomeLoss", "ProfitLoss"]),
    "interest_expense": ("d", [
        "InterestExpense", "InterestExpenseDebt",
        "InterestExpenseNonoperating"]),
    "diluted_shares": ("d", ["WeightedAverageNumberOfDilutedSharesOutstanding"]),

    "total_assets": ("i", ["Assets"]),
    "current_assets": ("i", ["AssetsCurrent"]),
    "current_liabilities": ("i", ["LiabilitiesCurrent"]),
    "_cash_only": ("i", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
    "_short_term_inv": ("i", [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "MarketableSecuritiesCurrent"]),
    "current_debt": ("i", ["LongTermDebtCurrent", "DebtCurrent"]),
    "long_term_debt": ("i", ["LongTermDebtNoncurrent", "LongTermDebt"]),
    "equity": ("i", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "retained_earnings": ("i", ["RetainedEarningsAccumulatedDeficit"]),
    "inventory": ("i", ["InventoryNet"]),
    "receivables": ("i", [
        "AccountsReceivableNetCurrent", "ReceivablesNetCurrent"]),
    "_goodwill": ("i", ["Goodwill"]),
    "_intangibles": ("i", [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet"]),
    "equity_investments": ("i", [
        "EquityMethodInvestments", "LongTermInvestments"]),
    "total_liabilities": ("i", ["Liabilities"]),
    "shares_out": ("i", [
        "CommonStockSharesOutstanding", "CommonStockSharesIssued"]),

    "ocf": ("d", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]),
    "capex": ("d", [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets"]),
    "depreciation": ("d", [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet", "Depreciation"]),
    "sbc": ("d", ["ShareBasedCompensation"]),
    "dividends_paid": ("d", [
        "PaymentsOfDividendsCommonStock", "PaymentsOfDividends"]),
    "buybacks": ("d", ["PaymentsForRepurchaseOfCommonStock"]),
    "acquisitions": ("d", ["PaymentsToAcquireBusinessesNetOfCashAcquired"]),
    "change_wc": ("d", ["IncreaseDecreaseInOperatingCapital"]),
}


# ── HTTP ────────────────────────────────────────────────────────────────────
def _headers() -> dict:
    ua = settings.sec_user_agent()
    if not ua or "@" not in ua:
        raise ValueError(
            "SEC EDGAR는 User-Agent에 실제 연락처 이메일을 요구합니다. "
            "Streamlit Secrets에 SEC_USER_AGENT를 설정하세요 "
            '(예: SEC_USER_AGENT = "hong gildong hong@example.com").')
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate",
            "Host": None}


def _get(url: str, timeout: float = 15.0, retries: int = 2):
    import requests

    h = {k: v for k, v in _headers().items() if v}
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=h, timeout=timeout)
            if r.status_code == 403:
                raise PermissionError(
                    "EDGAR 403 — User-Agent 형식을 확인하세요(이름 + 이메일).")
            r.raise_for_status()
            return r
        except PermissionError:
            raise
        except Exception as e:
            last = e
            time.sleep(0.4 * (i + 1))       # 초당 10회 제한 여유
    raise last or RuntimeError("EDGAR 요청 실패")


@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, int]:
    """티커 → CIK. 세션당 1회만 조회."""
    data = json.loads(_get(TICKER_MAP_URL).text)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def get_cik(ticker: str) -> int | None:
    base = ticker.upper().split(".")[0]
    m = _ticker_map()
    if base in m:
        return m[base]
    return m.get(base.replace("-", ""))     # BRK-B → BRKB 형태 보정


# ── XBRL 파싱 ───────────────────────────────────────────────────────────────
def _units(fact: dict) -> list[dict]:
    """단위 딕셔너리에서 금액/주식수 시계열을 꺼낸다."""
    u = fact.get("units") or {}
    for key in ("USD", "shares", "pure"):
        if key in u:
            return u[key]
    return next(iter(u.values()), [])


def _annual_series(facts: dict, tags: list[str], kind: str) -> dict[int, float]:
    """태그 후보를 순서대로 시도해 {회계연도: 값} 반환."""
    gaap = facts.get("us-gaap", {})
    for tag in tags:
        if tag not in gaap:
            continue
        rows = _units(gaap[tag])
        best: dict[int, dict] = {}
        for e in rows:
            form = str(e.get("form", ""))
            if not form.startswith("10-K"):          # 10-K, 10-K/A만
                continue
            end, val = e.get("end"), e.get("val")
            if end is None or val is None:
                continue
            start = e.get("start")
            if kind == "d":
                if not start:
                    continue
                try:
                    days = (dt.date.fromisoformat(end)
                            - dt.date.fromisoformat(start)).days
                except ValueError:
                    continue
                if not (330 <= days <= 400):         # 연간 기간만
                    continue
            else:
                if start:                            # 시점 항목은 start 없음
                    continue
            y = int(end[:4])
            prev = best.get(y)
            # 같은 해: 기말일이 늦은 것 → 그다음 제출일이 늦은 것(소급수정 반영)
            key = (end, str(e.get("filed", "")))
            if prev is None or key > (prev["end"], str(prev.get("filed", ""))):
                best[y] = e
        if best:
            return {y: float(e["val"]) for y, e in best.items()}
    return {}


def build_annual(facts: dict) -> pd.DataFrame:
    """companyfacts JSON → 표준 연간 재무 DataFrame."""
    data: dict[str, dict[int, float]] = {}
    for name, (kind, tags) in TAGS.items():
        s = _annual_series(facts, tags, kind)
        if s:
            data[name] = s
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).sort_index()
    df.index.name = "fy"

    # 야후 정의에 맞춘 합성 항목
    def col(c):
        return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

    df["cash"] = col("_cash_only").fillna(0) + col("_short_term_inv").fillna(0)
    df.loc[col("_cash_only").isna() & col("_short_term_inv").isna(), "cash"] = np.nan
    df["goodwill_intangibles"] = (col("_goodwill").fillna(0)
                                  + col("_intangibles").fillna(0))
    df.loc[col("_goodwill").isna() & col("_intangibles").isna(),
           "goodwill_intangibles"] = np.nan
    df["total_debt"] = col("current_debt").fillna(0) + col("long_term_debt").fillna(0)
    df.loc[col("current_debt").isna() & col("long_term_debt").isna(),
           "total_debt"] = np.nan
    df["working_capital"] = col("current_assets") - col("current_liabilities")

    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def fetch_annual(ticker: str) -> tuple[pd.DataFrame, str]:
    """티커 → (연간 재무 DataFrame, 안내 메시지). 실패 시 예외."""
    cik = get_cik(ticker)
    if cik is None:
        raise ValueError(f"{ticker}: EDGAR 티커맵에 없음(ADR·비상장 가능성)")
    facts = json.loads(_get(FACTS_URL.format(cik=cik)).text).get("facts", {})
    df = build_annual(facts)
    if df.empty or "revenue" not in df.columns:
        raise ValueError(f"{ticker}: EDGAR에서 표준 태그를 찾지 못함")
    yrs = df.index.tolist()
    msg = (f"SEC EDGAR XBRL {len(yrs)}개년({min(yrs)}~{max(yrs)}) 적용 "
           f"· CIK {cik:010d} · 10-K 기준, 소급수정 반영")
    return df, msg

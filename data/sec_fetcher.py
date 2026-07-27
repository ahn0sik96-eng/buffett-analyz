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
    """SEC 티커맵 조회. SEC는 클래스 주식을 'BRK-B'(하이픈)로 등재한다.

    점 표기·접미사·구분자 제거 순으로 시도 — split('.')만 쓰면 'BRK.B'가
    'BRK'로 잘려 조회에 실패한다."""
    m = _ticker_map()
    tu = ticker.upper()
    for cand in (tu, tu.replace(".", "-"), tu.split(".")[0],
                 tu.replace("-", "").replace(".", "")):
        if cand and cand in m:
            return m[cand]
    return None


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


def _dur_days(e: dict) -> int | None:
    try:
        return (dt.date.fromisoformat(e["end"])
                - dt.date.fromisoformat(e["start"])).days
    except Exception:
        return None


def _ttm_one(rows: list[dict]) -> tuple[float, str] | None:
    """단일 태그의 TTM = 직전 FY + 당기 YTD − 전년 동기 YTD.

    10-Q에는 3·6·9개월 YTD가 섞여 있고 Q4는 아예 없으므로(연차보고로 대체)
    분기 4개 합산 방식은 쓸 수 없다. FY+YTD−전년동기 방식은 기간 정의만
    맞으면 결측 분기가 없다.
    """
    dur = [e for e in rows
           if e.get("start") and e.get("end") and e.get("val") is not None]
    ann = [e for e in dur if str(e.get("form", "")).startswith("10-K")
           and 330 <= (_dur_days(e) or 0) <= 400]
    if not ann:
        return None
    fy = max(ann, key=lambda e: (e["end"], str(e.get("filed", ""))))
    fy_end = dt.date.fromisoformat(fy["end"])

    cur = [e for e in dur if dt.date.fromisoformat(e["start"]) > fy_end]
    if not cur:                                  # 새 회계연도 10-Q 미제출
        return None
    ytd = max(cur, key=lambda e: (e["end"], str(e.get("filed", ""))))
    d = _dur_days(ytd) or 0
    ytd_end = dt.date.fromisoformat(ytd["end"])

    prv = [e for e in dur
           if abs((_dur_days(e) or 0) - d) <= 20
           and abs((ytd_end - dt.date.fromisoformat(e["end"])).days - 365) <= 25]
    if not prv:
        return None
    p = max(prv, key=lambda e: str(e.get("filed", "")))
    return (float(fy["val"]) + float(ytd["val"]) - float(p["val"]),
            ytd["end"])


_TTM_FIELDS = [("revenue", "revenue"), ("operating_income", "ebit"),
               ("net_income", "net_income"), ("ocf", "ocf"),
               ("capex", "capex"), ("depreciation", "depreciation")]


def build_ttm(facts: dict) -> tuple[dict | None, str | None]:
    """companyfacts → 야후 ttm dict와 동일한 키 구조의 TTM. 실패 시 (None, None)."""
    gaap = facts.get("us-gaap", {})
    vals: dict[str, float] = {}
    end: str | None = None
    for field, key in _TTM_FIELDS:
        _, tags = TAGS[field]
        for tag in tags:
            if tag not in gaap:
                continue
            r = _ttm_one(_units(gaap[tag]))
            if r:
                vals[key], e = r
                end = max(end, e) if end else e
                break
    if "ocf" not in vals or "capex" not in vals:
        return None, None                        # FCF를 못 만들면 의미 없음
    out = {
        "revenue": vals.get("revenue", np.nan),
        "ebit": vals.get("ebit", np.nan),
        "net_income": vals.get("net_income", np.nan),
        "ocf": vals["ocf"],
        "capex_out": abs(vals["capex"]),
    }
    out["fcf"] = out["ocf"] - out["capex_out"]
    if np.isfinite(vals.get("ebit", np.nan)) and             np.isfinite(vals.get("depreciation", np.nan)):
        out["ebitda"] = vals["ebit"] + abs(vals["depreciation"])
    msg = f"TTM 갱신: 직전 FY + {end} 종료 YTD − 전년 동기 (EDGAR 10-Q)"
    return out, msg


def fetch_annual_and_ttm(ticker: str
                         ) -> tuple[pd.DataFrame, str, dict | None, str | None]:
    """연간 + TTM을 companyfacts 1회 호출로 함께 반환."""
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
    ttm, ttm_msg = build_ttm(facts)
    return df, msg, ttm, ttm_msg


def fetch_annual(ticker: str) -> tuple[pd.DataFrame, str]:
    """호환용 래퍼 — 연간만 필요할 때."""
    df, msg, _, _ = fetch_annual_and_ttm(ticker)
    return df, msg

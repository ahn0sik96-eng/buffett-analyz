"""DART OpenAPI 재무데이터 수집 (국내 상장사).

EDGAR와 다른 점 — 설계가 이래야 하는 이유:
  1) 인증 필요       : crtfc_key 발급 필수(EDGAR는 무인증).
  2) 연도별 호출     : companyfacts처럼 전체 이력을 한 번에 주지 않는다.
                       단 사업보고서 1건에 당기·전기·전전기가 들어 있어
                       3년씩 건너뛰며 호출하면 5회로 15년을 덮는다.
  3) 고유번호 매핑   : 종목코드(005930)가 아니라 DART corp_code(8자리)를
                       써야 한다. corpCode.xml을 ZIP으로 받아 파싱한다.
  4) 계정 식별       : account_id(IFRS 표준계정코드)가 우선이지만
                       '-표준계정코드 미사용-' 인 경우가 많아 한글 계정명
                       매칭을 폴백으로 둔다.
  5) 시작 시점       : API 제공은 2015 사업연도부터. 전전기까지 끌어와도
                       2013년이 하한이라 EDGAR(2008~)보다 이력이 짧다.

반환 형태는 data_fetcher._collect_annual()과 동일한 표준 DataFrame이다.
"""
from __future__ import annotations

import io
import time
import zipfile
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import pandas as pd

from config import settings

BASE = "https://opendart.fss.or.kr/api"
CORP_CODE_URL = f"{BASE}/corpCode.xml"
FNLTT_URL = f"{BASE}/fnlttSinglAcntAll.json"

REPRT_ANNUAL = "11011"          # 사업보고서
# 분기 TTM용 — 제출된 가장 최신 보고서를 앞에서부터 시도
REPRT_QUARTERS = [("11014", "3분기"), ("11012", "반기"), ("11013", "1분기")]
DART_FIRST_YEAR = 2015          # API 제공 시작 사업연도

STATUS_MSG = {
    "010": "등록되지 않은 인증키입니다.",
    "011": "사용할 수 없는 인증키입니다(일시적 사용 중지).",
    "012": "접근할 수 없는 IP입니다.",
    "013": "조회된 데이터가 없습니다.",
    "020": "요청 제한을 초과했습니다(일 20,000건).",
    "021": "조회 가능한 회사 개수가 초과했습니다.",
    "100": "필드의 부적절한 값입니다.",
    "101": "부적절한 접근입니다.",
    "800": "시스템 점검 중입니다.",
    "900": "정의되지 않은 오류가 발생했습니다.",
    "901": "사용자 계정의 개인정보보유기간이 만료되었습니다.",
}

# ── 총차입금 구성 계정 (재무상태표, 계정명 정확 일치·공백 제거 기준) ────────
# IFRS 표준계정에 '총차입금' 합계가 없어 구성 항목을 직접 합산한다.
# 부모 계정('차입금' 단독 등)은 자식과 이중계상되므로 넣지 않는다.
# 같은 이름이 유동·비유동에 각각 나오면(예: '사채', '리스부채') 둘 다 합산
# — 서로 다른 줄이므로 그것이 맞다.
DEBT_COMPONENT_NAMES = {
    "단기차입금", "장기차입금", "유동성장기차입금", "유동성장기부채",
    "사채", "유동성사채", "단기사채", "전환사채", "신주인수권부사채",
    "교환사채", "리스부채", "유동리스부채", "비유동리스부채", "금융리스부채",
}

# ── 표준항목 → (IFRS account_id 후보, 한글 계정명 후보) ─────────────────────
# account_id는 2019년경 'ifrs_' → 'ifrs-full_' 로 접두사가 바뀌어 둘 다 넣는다.
TAGS: dict[str, tuple[list[str], list[str]]] = {
    "revenue": (["ifrs-full_Revenue", "ifrs_Revenue"],
                ["매출액", "수익(매출액)", "영업수익", "매출"]),
    "gross_profit": (["ifrs-full_GrossProfit", "ifrs_GrossProfit"],
                     ["매출총이익"]),
    "operating_income": (["dart_OperatingIncomeLoss",
                          "dart_OperatingIncomeLossAbstract"],
                         ["영업이익", "영업이익(손실)"]),
    "pretax_income": (["ifrs-full_ProfitLossBeforeTax",
                       "ifrs_ProfitLossBeforeTax"],
                      ["법인세비용차감전순이익", "법인세비용차감전순이익(손실)"]),
    "tax_provision": (["ifrs-full_IncomeTaxExpenseContinuingOperations",
                       "ifrs_IncomeTaxExpenseContinuingOperations"],
                      ["법인세비용"]),
    "net_income": (["ifrs-full_ProfitLoss", "ifrs_ProfitLoss"],
                   ["당기순이익", "당기순이익(손실)", "분기순이익"]),
    "interest_expense": ([], ["이자비용", "금융비용"]),

    "total_assets": (["ifrs-full_Assets", "ifrs_Assets"], ["자산총계"]),
    "current_assets": (["ifrs-full_CurrentAssets", "ifrs_CurrentAssets"],
                       ["유동자산"]),
    "current_liabilities": (["ifrs-full_CurrentLiabilities",
                             "ifrs_CurrentLiabilities"], ["유동부채"]),
    "cash": (["ifrs-full_CashAndCashEquivalents", "ifrs_CashAndCashEquivalents"],
             ["현금및현금성자산"]),
    "equity": (["ifrs-full_Equity", "ifrs_Equity"], ["자본총계"]),
    "retained_earnings": (["ifrs-full_RetainedEarnings", "ifrs_RetainedEarnings"],
                          ["이익잉여금", "이익잉여금(결손금)"]),
    "inventory": (["ifrs-full_Inventories", "ifrs_Inventories"], ["재고자산"]),
    "receivables": ([], ["매출채권", "매출채권 및 기타유동채권"]),
    "total_liabilities": (["ifrs-full_Liabilities", "ifrs_Liabilities"],
                          ["부채총계"]),
    "goodwill_intangibles": (["ifrs-full_IntangibleAssetsOtherThanGoodwill"],
                             ["무형자산", "영업권"]),

    "ocf": (["ifrs-full_CashFlowsFromUsedInOperatingActivities",
             "ifrs_CashFlowsFromUsedInOperatingActivities"],
            ["영업활동현금흐름", "영업활동으로인한현금흐름"]),
    "capex": (["ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAs"
               "InvestingActivities"],
              ["유형자산의 취득", "유형자산의취득"]),
    "depreciation": (["ifrs-full_DepreciationAndAmortisationExpense"],
                     ["감가상각비", "감가상각비와 상각비"]),
    "dividends_paid": ([], ["배당금지급", "배당금의 지급"]),
}


# ── 인증·HTTP ───────────────────────────────────────────────────────────────
def api_key() -> str:
    k = settings.dart_api_key()
    if not k:
        raise ValueError(
            "DART API 키가 없습니다. Streamlit Secrets에 "
            'DART_API_KEY = "..." 를 설정하세요 (opendart.fss.or.kr 무료 발급).')
    return k


def _get(url: str, params: dict, timeout: float = 15.0, retries: int = 2):
    import requests

    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(0.3 * (i + 1))
    raise last or RuntimeError("DART 요청 실패")


def _check(js: dict):
    st = str(js.get("status", ""))
    if st and st != "000":
        raise ValueError(f"DART {st}: {STATUS_MSG.get(st, js.get('message',''))}")


# ── 고유번호 매핑 ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _corp_map() -> dict[str, str]:
    """종목코드(6자리) → corp_code(8자리). ZIP+XML이라 세션당 1회만 조회."""
    r = _get(CORP_CODE_URL, {"crtfc_key": api_key()}, timeout=30.0)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        xml = z.read(z.namelist()[0])
    out: dict[str, str] = {}
    for e in ET.fromstring(xml).iter("list"):
        stock = (e.findtext("stock_code") or "").strip()
        corp = (e.findtext("corp_code") or "").strip()
        if stock and corp:                       # 상장사만(비상장은 공란)
            out[stock] = corp
    return out


def get_corp_code(stock_code: str) -> str | None:
    return _corp_map().get(stock_code.strip()[:6])


# ── 금액 파싱 ───────────────────────────────────────────────────────────────
def _amt(v) -> float:
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", "").replace(" ", "")
    if s in ("", "-", "--"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return -x if neg else x


def _match(item: dict, ids: list[str], names: list[str]) -> bool:
    aid = (item.get("account_id") or "").strip()
    if aid and aid in ids:
        return True
    if aid and aid not in ("-표준계정코드 미사용-", ""):
        return False                       # 표준코드가 있는데 다르면 불일치
    nm = (item.get("account_nm") or "").strip().replace(" ", "")
    return any(nm == n.replace(" ", "") for n in names)


def _parse_report(items: list[dict], year: int) -> dict[int, dict[str, float]]:
    """사업보고서 1건 → {연도: {항목: 값}}. 당기·전기·전전기 3개년을 뽑는다."""
    cols = [(year, "thstrm_amount"), (year - 1, "frmtrm_amount"),
            (year - 2, "bfefrmtrm_amount")]
    out: dict[int, dict[str, float]] = {y: {} for y, _ in cols}

    for field, (ids, names) in TAGS.items():
        for it in items:
            if not _match(it, ids, names):
                continue
            for y, key in cols:
                v = _amt(it.get(key))
                if np.isfinite(v) and field not in out[y]:
                    out[y][field] = v
            break                          # 항목당 첫 매칭만 채택

    # 총차입금 = 차입금·사채·리스부채 구성 계정 합산(재무상태표만)
    for it in items:
        if (it.get("sj_div") or "").upper() != "BS":
            continue
        nm = (it.get("account_nm") or "").strip().replace(" ", "")
        if nm not in DEBT_COMPONENT_NAMES:
            continue
        for y, key in cols:
            v = _amt(it.get(key))
            if np.isfinite(v):
                out[y]["total_debt"] = out[y].get("total_debt", 0.0) + v

    return {y: d for y, d in out.items() if d}


# ── 분기 TTM ────────────────────────────────────────────────────────────────
_FLOW_TTM = ("revenue", "operating_income", "net_income", "ocf", "capex",
             "depreciation")


def _fetch_items(corp: str, year: int, reprt: str, fs_div: str) -> list[dict]:
    js = _get(FNLTT_URL, {"crtfc_key": api_key(), "corp_code": corp,
                          "bsns_year": str(year), "reprt_code": reprt,
                          "fs_div": fs_div}).json()
    _check(js)
    return js.get("list") or []


def _ytd_flows(items: list[dict]) -> dict[str, float]:
    """분기·반기보고서에서 누적(YTD) 흐름 항목 추출.

    누적값 우선순위: thstrm_add_amount(손익 누적) → thstrm_amount.
    손익계산서는 반기·3분기 보고서에서 thstrm_amount가 '해당 3개월'이고
    누적은 add_amount에 있다. 현금흐름표는 누적만 공시되므로
    thstrm_amount가 곧 누적이고 add_amount는 비어 있다 — 이 우선순위가
    두 경우를 모두 맞게 처리한다.
    """
    out: dict[str, float] = {}
    for f in _FLOW_TTM:
        ids, names = TAGS[f]
        for it in items:
            if (it.get("sj_div") or "").upper() in ("BS", "SCE"):
                continue
            if not _match(it, ids, names):
                continue
            v = _amt(it.get("thstrm_add_amount"))
            if not np.isfinite(v):
                v = _amt(it.get("thstrm_amount"))
            if np.isfinite(v):
                out[f] = v
            break
    return out


def _assemble_ttm(fy_flows: dict, cur: dict, prv: dict) -> dict | None:
    """TTM = 직전 FY + 당기 누적 − 전년 동기 누적. 야후 ttm dict 키로 반환."""
    key_map = {"operating_income": "ebit"}
    vals: dict[str, float] = {}
    for f in _FLOW_TTM:
        a, b, q = fy_flows.get(f), cur.get(f), prv.get(f)
        if all(x is not None and np.isfinite(x) for x in (a, b, q)):
            vals[key_map.get(f, f)] = float(a) + float(b) - float(q)
    if "ocf" not in vals or "capex" not in vals:
        return None                    # FCF를 못 만들면 TTM 교체 의미 없음
    out = {k: v for k, v in vals.items()
           if k not in ("capex", "depreciation")}
    out["capex_out"] = abs(vals["capex"])
    out["fcf"] = out["ocf"] - out["capex_out"]
    if np.isfinite(vals.get("ebit", np.nan)) and             np.isfinite(vals.get("depreciation", np.nan)):
        out["ebitda"] = vals["ebit"] + abs(vals["depreciation"])
    return out


def fetch_ttm(stock_code: str, fy_year: int, fy_flows: dict
              ) -> tuple[dict | None, str | None]:
    """최신 분기·반기보고서 기준 TTM. 실패 시 (None, None) — 야후 TTM 유지.

    3분기 → 반기 → 1분기 순으로 시도해 '제출된 가장 최신 보고서'를 자동
    선택한다. 반기보고서 법정기한이 8월 중순이라 7월에는 1분기가 최신이고,
    반기가 제출되는 즉시 코드 수정 없이 반기 기준으로 넘어간다.
    전년 동기는 같은 reprt_code·fs_div로 별도 조회한다 — 현재 보고서의
    frmtrm 칼럼은 보고서 유형에 따라 의미가 달라 신뢰할 수 없다.
    """
    corp = get_corp_code(stock_code)
    if corp is None:
        return None, None
    cur_year = fy_year + 1
    cur_items, used_div, used_label = None, None, None
    for reprt, label in REPRT_QUARTERS:
        for fs_div in ("CFS", "OFS"):
            try:
                items = _fetch_items(corp, cur_year, reprt, fs_div)
            except Exception:
                continue
            if items:
                cur_items, used_div, used_label = items, fs_div, label
                used_reprt = reprt
                break
        if cur_items:
            break
        time.sleep(0.1)
    if not cur_items:
        return None, None
    try:
        prv_items = _fetch_items(corp, fy_year, used_reprt, used_div)
    except Exception:
        return None, None
    if not prv_items:
        return None, None
    out = _assemble_ttm(fy_flows, _ytd_flows(cur_items), _ytd_flows(prv_items))
    if out is None:
        return None, None
    msg = (f"TTM 갱신: FY{fy_year} + {cur_year} {used_label} 누적 − "
           f"{fy_year} 동기 (DART {used_label}보고서·{used_div})")
    return out, msg


def fetch_annual(stock_code: str, years_back: int = 12
                 ) -> tuple[pd.DataFrame, str]:
    """종목코드 → (연간 재무 DataFrame, 안내 메시지). 실패 시 예외."""
    corp = get_corp_code(stock_code)
    if corp is None:
        raise ValueError(f"{stock_code}: DART 고유번호를 찾지 못함(비상장·폐지 가능성)")

    import datetime as dt
    latest = dt.date.today().year - 1        # 직전 사업연도부터
    targets, y = [], latest
    while y >= max(DART_FIRST_YEAR, latest - years_back) and len(targets) < 6:
        targets.append(y)
        y -= 3                               # 보고서당 3개년 → 3년씩 건너뛰기

    merged: dict[int, dict[str, float]] = {}
    errors: list[str] = []
    for by in targets:
        got = False
        for fs_div in ("CFS", "OFS"):        # 연결 우선, 없으면 별도
            try:
                js = _get(FNLTT_URL, {
                    "crtfc_key": api_key(), "corp_code": corp,
                    "bsns_year": str(by), "reprt_code": REPRT_ANNUAL,
                    "fs_div": fs_div}).json()
                _check(js)
                parsed = _parse_report(js.get("list", []), by)
                for yy, d in parsed.items():
                    # 최신 보고서가 뒤에 오지 않도록: 기존 값이 없을 때만 채움
                    merged.setdefault(yy, {}).update(
                        {k: v for k, v in d.items() if k not in merged[yy]})
                got = bool(parsed)
                if got:
                    break
            except Exception as e:
                errors.append(f"{by}/{fs_div}: {e}")
        time.sleep(0.1)                      # 호출 간격 여유

    if not merged:
        raise ValueError("DART 재무 데이터 없음 — " + ("; ".join(errors[:2])
                                                   or "응답 비어 있음"))

    df = pd.DataFrame(merged).T.sort_index()
    df.index.name = "fy"
    if "revenue" not in df.columns:
        raise ValueError("DART 응답에서 매출 계정을 찾지 못함")

    # 야후 정의에 맞춘 합성 항목
    def col(c):
        return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

    df["working_capital"] = col("current_assets") - col("current_liabilities")
    if "total_debt" not in df.columns:
        df["total_debt"] = np.nan            # DART 표준계정에 차입금 합계 없음

    yrs = df.index.tolist()
    msg = (f"DART OpenAPI {len(yrs)}개년({min(yrs)}~{max(yrs)}) 적용 · "
           f"고유번호 {corp} · 사업보고서(연결우선) 기준")
    if "total_debt" in df and df["total_debt"].notna().any():
        msg += " · 총차입금은 차입금·사채·리스부채 계정명 합산(근사)"
    else:
        msg += " · 총차입금 계정 미검출 — 순부채는 현금 기준 근사"
    return df, msg

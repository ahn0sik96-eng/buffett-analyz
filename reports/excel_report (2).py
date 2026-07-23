"""엑셀 보고서 (명세 15장의 간이판 — 전체 시트 구성은 6단계에서 완성).

원천 데이터·가정을 함께 담아 계산 추적이 가능하도록 한다.
"""
from __future__ import annotations

from io import BytesIO

import pandas as pd


def _kv(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"항목": list(d.keys()), "값": ["N/A" if v is None else str(v) for v in d.values()]})


def build_excel(fd, roic_res, cf_res, re_res, debt_res, mult, scen, sens,
                scores, penalties, assumptions: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        summary = {
            "기업명": fd.name, "티커": fd.ticker, "통화": fd.currency,
            "현재주가": fd.price, "시가총액": fd.market_cap,
            "섹터": fd.sector, "산업": fd.industry,
            "종합점수(환산)": scores.get("total_norm"),
            "등급": scores.get("grade"),
            "기업의 질(환산)": scores.get("quality_norm"),
            "밸류에이션(환산)": scores.get("val_norm"),
            "감점 합계": scores.get("penalty"),
            "부분평가 주석": scores.get("partial_note"),
        }
        _kv(summary).to_excel(xw, sheet_name="Summary", index=False)
        fd.annual.to_excel(xw, sheet_name="Financial Statements")
        roic_res["table"].to_excel(xw, sheet_name="ROIC")
        cf_res["table"].to_excel(xw, sheet_name="FCF")
        re_res["table"].to_excel(xw, sheet_name="Reinvestment")

        L = dict(debt_res["latest"])
        if debt_res.get("altman"):
            L["altman_z"] = debt_res["altman"]["z"]
            L["altman_zone"] = debt_res["altman"]["zone"]
        if debt_res.get("piotroski"):
            L["piotroski"] = f'{debt_res["piotroski"]["score"]}/{debt_res["piotroski"]["valid"]}'
        _kv(L).to_excel(xw, sheet_name="Debt", index=False)

        _kv(mult).to_excel(xw, sheet_name="Valuation", index=False)

        if scen:
            pd.DataFrame(scen).T.to_excel(xw, sheet_name="DCF")
        if sens is not None:
            sens.to_excel(xw, sheet_name="DCF_Sensitivity")

        pen_df = pd.DataFrame(penalties, columns=["감점 사유", "점수"]) if penalties \
            else pd.DataFrame({"감점 사유": ["없음"], "점수": [0]})
        pen_df.to_excel(xw, sheet_name="Risk Alerts", index=False)

        fd.annual.to_excel(xw, sheet_name="Raw Data")
        _kv(assumptions).to_excel(xw, sheet_name="Assumptions", index=False)
    return bio.getvalue()

"""
utils/convert_ctcae_excel.py
────────────────────────────
Extracts the CTCAE v6.0 terms used by Agentic ChemoTwin from the official
NCI workbook and writes data/rules/ctcae_v6_terms.json, the machine-readable
provenance record for every threshold in utils/ctcae_utils.py.

Workbook: data/guidelines/CTCAE_v6_0_Final_Clean-Tracked-Mapping_w_OS_Jan2026.xlsx
Sheets used: "CTCAE v6.0 Clean Copy" (current definitions) and
             "CTCAE v5.0 to v6.0 Mapping" (terms deleted in v6.0).

REVISION NOTE (Oct 2026): the previous version of this script read the
workbook's FIRST sheet ("Publication Notes", which is empty), looked for the
v5.0 term "Left ventricular systolic dysfunction" (deleted in v6.0), and
indexed columns as "Grade 1" while the real headers are "Grade 1" followed by
two non-breaking spaces. It therefore could not have produced the
ctcae_v6_extracted.json / ctcae_cardiac.json files that were in the repo;
those were hand-written and contained v5.0 values. Both are superseded by
the output of this script.

Usage:  python utils/convert_ctcae_excel.py
"""

import glob
import json
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUIDELINES_DIR = PROJECT_ROOT / "data" / "guidelines"
OUT_FILE = PROJECT_ROOT / "data" / "rules" / "ctcae_v6_terms.json"

CLEAN_SHEET = "CTCAE v6.0 Clean Copy"
MAPPING_SHEET = "CTCAE v5.0 to v6.0 Mapping"

TERMS_USED = [
    "Neutrophil count decreased",
    "Febrile neutropenia",
    "Left ventricular dysfunction",
    "Heart failure",
    "Blood bilirubin increased",
    "Creatinine increased",
    "Alanine aminotransferase increased",
    "Aspartate aminotransferase increased",
]
# v5.0 terms this project used to rely on; documented so nobody re-adds them
REMOVED_V5_TERMS = ["Left ventricular systolic dysfunction", "Ejection fraction decreased"]


def _find_workbook() -> Path:
    hits = sorted(glob.glob(str(GUIDELINES_DIR / "CTCAE*v6*.xlsx")))
    if not hits:
        raise FileNotFoundError(f"No CTCAE v6 workbook found in {GUIDELINES_DIR}")
    return Path(hits[0])


def _clean(v) -> str:
    s = "" if pd.isna(v) else str(v).strip()
    return "" if s == "-" else s


def convert(workbook: Path = None, out_file: Path = OUT_FILE) -> dict:
    workbook = workbook or _find_workbook()
    df = pd.read_excel(workbook, sheet_name=CLEAN_SHEET)
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    term_col = "CTCAE v6.0 MedDRA 28.0 Term"

    terms = {}
    for term in TERMS_USED:
        rows = df[df[term_col].astype(str).str.strip() == term]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one '{term}' row in {CLEAN_SHEET}, found {len(rows)}")
        r = rows.iloc[0]
        terms[term] = {
            "meddra_llt_code": int(r["CTCAE v6.0 MedDRA 28.0 LLT Code"]),
            "soc": _clean(r["CTCAE v6.0 MedDRA 28.0 SOC"]),
            **{f"grade_{g}": _clean(r[f"Grade {g}"]) for g in range(1, 6)},
            "definition": _clean(r["Definition"]),
            "navigational_note": _clean(r["Navigational Note"]),
            "v6_change_note": _clean(r["CTCAE v6.0 Change"]),
        }

    m = pd.read_excel(workbook, sheet_name=MAPPING_SHEET, header=None)
    removed = {}
    for term in REMOVED_V5_TERMS:
        rows = m[m[2].astype(str).str.strip() == term]
        if rows.empty:
            continue
        removed[term] = {
            "v5_meddra_code": int(rows.iloc[0][0]),
            "change": _clean(rows.iloc[0][5]),
            "replaced_by_v6_term": _clean(rows.iloc[0][8]),
        }
    for term in REMOVED_V5_TERMS:
        present = (df[term_col].astype(str).str.strip() == term).any()
        if present:
            raise ValueError(f"'{term}' unexpectedly present in v6.0 clean sheet")

    payload = {
        "ctcae_version": "6.0",
        "source_workbook": workbook.name,
        "source_sheets": [CLEAN_SHEET, MAPPING_SHEET],
        "generated_on": date.today().isoformat(),
        "generated_by": "utils/convert_ctcae_excel.py",
        "terms": terms,
        "removed_in_v6": removed,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


if __name__ == "__main__":
    p = convert()
    print(f"Wrote {OUT_FILE} with {len(p['terms'])} terms; removed-in-v6: {list(p['removed_in_v6'])}")
"""
agentic/layer_b.py
──────────────────
Adapter for the layer B dataset (calibrated synthetic HER2+ breast-cancer
records, generate_layer_b.py) so every arm of the experiment can be run on it.

Two patients are built per case:

  RECORD patient   parsed from the documents exactly AS WRITTEN (clinic note,
                   echo reports, lab table). This is what the agent's tools
                   read. A troponin written in ng/mL is converted (unit
                   normalisation is a legitimate tool job and is noted), an LVEF
                   typo stays a typo, a missing baseline echo stays missing.
                   Nothing the generator knows but the record does not show
                   (clean values, scenario, symptoms only in a nursing note)
                   reaches the tools.
  TRUTH patient    built from the generator's CLEAN values and documented
                   symptom status. The answer key (CTCAE v6.0 grades, ESC 2022
                   CTRCD, management) is computed from it by the same spec
                   module as for the stress set.

What is knowable decides deferral: if the RECORD lacks a safety-critical input
(or contains an implausible one), a correct answer defers and asks, even
though the truth patient has a clean value.

Every arm sees the same record text (render_record); the answer key never
appears in it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional

from state.patient_state import CardiacAssessment, CycleRecord, LabResult, PatientState

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "layer_b_v0.1")
TROP_URL_PG_ML = 30.0          # hs-TnI 99th-percentile URL used by the generator (Sawaya 2012)
SETS = ["boundary", "currency", "missing", "conflict", "fault"]

# Generator ESC category -> CTRCD answers accepted from a system
ACCEPT_CTRCD = {
    "none": {"none"},
    "mild": {"mild"},
    "moderate": {"moderate"},
    "severe": {"severe"},
    "none_lvef_40_49_needs_workup": {"none", "not_evaluable"},
    "indeterminate_missing": {"not_evaluable", "none", "mild"},
}

# What a correct answer must mention for each required flag (searched in all answer text)
FLAG_PATTERNS = {
    "missing_data": None,  # covered by data_requests
    "gls_change_not_computable": r"\bGLS\b.{0,80}(baseline|not (available|done|performed|measured)|missing|cannot)"
                                 r"|(baseline|missing|no).{0,40}\bGLS\b",
    "biomarkers_not_measured": r"troponin|NT-?proBNP|biomarker",
    "baseline_lvef_unavailable": r"baseline.{0,60}(echo|LVEF|report)|(outside|external).{0,40}(echo|report)",
    "surveillance_gap": r"miss(ed|ing).{0,40}(visit|echo|month|surveillance)|gap|not attended|DNA",
    "units_absent": r"\bunits?\b",
    "conflicting_lvef_values": r"discrepan|conflict|discord|differ|verbal|two (different )?(EF|LVEF|values)",
    "symptoms_documented_elsewhere": r"short(ness)? of breath|\bSOB\b|dyspn|oedema|edema|swelling|orthopn|pillow",
    "record_action_inconsistent_with_finding": r"cardioprotect|ACE|ARB|beta[- ]?blocker|bisoprolol|carvedilol|"
                                               r"perindopril|ramipril|candesartan",
    "cross_modality_comparison": r"MUGA|modalit",
    "unit_conversion_needed": r"ng/mL|ng/L|unit|convert",
    "gls_sign_convention_mixed": None,  # informational; correct GLS handling is scored through CTRCD
    "implausible_value": r"typo|implausib|error|verify|confirm|repeat|unlikely|transcription",
    "chronology_error": r"date|chronolog|order",
    "weight_inconsistent_with_dose": r"weight|BSA|dose",
    "outdated_criterion_in_record": None,  # scored through CTRCD (adopting the old label is a CTRCD error)
    "threshold_case": None,
}


# ===========================================================================
# Loading
# ===========================================================================
def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_layer_b(data_dir: Optional[str] = None, sets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    data_dir = data_dir or os.getenv("LAYER_B_DIR") or DEFAULT_DIR
    cases = _load(os.path.join(data_dir, "cases.jsonl"))
    records = {r["case_id"]: r for r in _load(os.path.join(data_dir, "records_for_model.jsonl"))}
    out = []
    for c in cases:
        if sets and c["set"] not in sets:
            continue
        docs = records[c["case_id"]]["documents"]
        record_patient = parse_record(c["case_id"], docs)
        out.append({
            "case_id": c["case_id"], "test_set": f"lb_{c['set']}", "layer": "B",
            "description": f"{c['scenario']} (ESC truth: {c['truth']['esc2022_ctrcd']})",
            "patient": record_patient, "truth_patient": truth_patient(c, record_patient),
            "record_text": render_record(docs), "clinical_note": None, "tags": [c["scenario"]],
            "lb": {"scenario": c["scenario"], "truth": c["truth"], "symptomatic": c.get("symptomatic")},
        })
    return out


# ===========================================================================
# Record text shown to every arm
# ===========================================================================
def render_record(docs: Dict[str, Any]) -> str:
    lines = ["REVIEW REQUESTED: cardio-oncology surveillance - assess cardiac toxicity (CTRCD) and decide on the "
             "current cancer therapy and cardioprotection. Blood counts, temperature, renal and liver tests are "
             "outside the scope of this review; report those CTCAE terms as not_evaluable.", "",
             "ONCOLOGY CLINIC NOTE", docs["clinic_note"]]
    if docs.get("nursing_note"):
        lines += ["", "NURSING NOTE", docs["nursing_note"]]
    lines += ["", "ECHOCARDIOGRAPHY / MUGA REPORTS"] + [f"- {e}" for e in docs.get("echo_reports", [])]
    lines += ["", "CARDIAC BIOMARKERS", "date | hs-TnI | NT-proBNP"]
    lines += [f"{r['date']} | {r['hs-TnI']} | {r['NT-proBNP']}" for r in docs.get("labs", [])]
    lines += [f"Lab note: hs-TnI assay 99th-percentile upper reference limit {TROP_URL_PG_ML:g} pg/mL."]
    return "\n".join(lines)


# ===========================================================================
# Record parsing (as written)
# ===========================================================================
_ECHO = re.compile(r"^(\d{4}-\d{2}-\d{2}) \[(ECHO|MUGA)\]: LVEF (\d+(?:\.\d+)?)%", re.I)
_GLS = re.compile(r"(?:\bGLS\b|global longitudinal strain|LV GLS:?)\s*:?\s*(-?\d+(?:\.\d+)?)", re.I)
_NUM_UNIT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(ng/mL|pg/mL|ng/L)?", re.I)


def _day(d: str, d0: date) -> int:
    return (date.fromisoformat(d) - d0).days


def _parse_value(text: str, notes: List[str], name: str) -> Optional[float]:
    if not text or "not done" in text.lower():
        return None
    m = _NUM_UNIT.search(text)
    if not m:
        return None
    v, unit = float(m.group(1)), (m.group(2) or "").lower()
    if unit == "ng/ml":
        notes.append(f"{name} reported in ng/mL ({text}); converted to {v * 1000:g} ng/L (= pg/mL).")
        return v * 1000.0
    if not unit:
        notes.append(f"{name} reported without units ({text}); pg/mL assumed.")
    return v


def _active_regimen(regimen_line: str, month: int) -> List[str]:
    r = regimen_line.lower()
    if "pertuzumab" in r and "docetaxel" in r:      # metastatic THP
        return ["Docetaxel", "Trastuzumab", "Pertuzumab"]
    if "tch" in r or "carboplatin" in r:
        return ["Docetaxel", "Carboplatin", "Trastuzumab"] if month < 5 else ["Trastuzumab"]
    if "ac ->" in r or "paclitaxel" in r:
        if month < 3:
            return ["Doxorubicin" if "doxorubicin" in r else "Epirubicin", "Cyclophosphamide"]
        return ["Paclitaxel", "Trastuzumab"] if month < 6 else ["Trastuzumab"]
    return ["Trastuzumab"]


def parse_record(case_id: str, docs: Dict[str, Any]) -> PatientState:
    note = docs["clinic_note"]
    notes: List[str] = []
    m_age = re.search(r"^(\d+)F", note, re.M)
    m_month = re.search(r"month (\d+) of treatment", note)
    month = int(m_month.group(1)) if m_month else 3
    m_reg = re.search(r"^Regimen: (.*)$", note, re.M)
    regimen = m_reg.group(1) if m_reg else ""
    m_meds = re.search(r"^Meds: (.*)$", note, re.M)
    home = [] if not m_meds or "nil" in m_meds.group(1).lower() else \
        [x.strip().rstrip(".") for x in m_meds.group(1).split(";") if x.strip()]
    m_cum = re.search(r"(doxorubicin|epirubicin) \(cumulative (\d+) mg/m2\)", regimen, re.I)
    prior = "given for prior cancer" in regimen or "Prior anthracycline" in note
    cum = float(m_cum.group(2)) if (m_cum and not prior) else 0.0
    if prior:
        notes.append("Prior anthracycline course for an earlier cancer: cumulative dose not recorded.")
    m_clinic = re.search(r"ONCOLOGY CLINIC - (\d{4}-\d{2}-\d{2})", note)

    lab_rows = docs.get("labs", [])
    d0 = date.fromisoformat(lab_rows[0]["date"]) if lab_rows else date.fromisoformat(m_clinic.group(1))
    labs = []
    for r in lab_rows:
        labs.append(LabResult(lab_day=_day(r["date"], d0), anc_cells_per_ul=None,
                              troponin_ng_l=_parse_value(r["hs-TnI"], notes, "hs-TnI"),
                              troponin_url_ng_l=TROP_URL_PG_ML,
                              nt_probnp_pg_ml=_parse_value(r["NT-proBNP"], notes, "NT-proBNP")))
    echoes = []
    for e in docs.get("echo_reports", []):
        m = _ECHO.match(e)
        if not m:
            if "outside" in e.lower() or "not available" in e.lower():
                notes.append("Pre-treatment echo done elsewhere; only a verbal 'normal LV function' is recorded.")
            continue
        g = _GLS.search(e[m.end():])
        gls = None if (g is None or "not performed" in e.lower()) else -abs(float(g.group(1)))
        echoes.append((m.group(1), m.group(2).upper(), float(m.group(3)), gls))
    cardiac = []
    has_baseline = bool(echoes) and lab_rows and echoes[0][0] == lab_rows[0]["date"]
    if echoes:
        if has_baseline:
            b = echoes[0]
            cardiac.append(CardiacAssessment(_day(b[0], d0), "Baseline", b[2], b[3]))
        cur = echoes[-1]
        if not has_baseline or cur is not echoes[0]:
            cardiac.append(CardiacAssessment(_day(cur[0], d0), "Current", cur[2], cur[3]))
        if has_baseline and echoes[0][1] != cur[1]:
            notes.append(f"LVEF modality differs: baseline {echoes[0][1]}, current {cur[1]}.")
    decision_day = _day(m_clinic.group(1), d0) if m_clinic else (cardiac[-1].assessment_day if cardiac else 0)
    p = PatientState(
        study_id=case_id, age=int(m_age.group(1)) if m_age else 55, sex="F",
        active_medications=_active_regimen(regimen, month) + home,
        cycles=[CycleRecord(cycle_number=max(1, month), session_day=decision_day,
                            doxorubicin_dose_this_cycle=0.0, cumulative_doxorubicin_dose=cum)],
        cardiac_assessments=cardiac, vitals=[], labs=labs,
        hypertension="HTN" in note, heart_failure_status=None,
    )
    p.assessment_scope = "cardiac"
    p.record_quality_notes = list(dict.fromkeys(notes))
    return p


# ===========================================================================
# Truth patient (clean values)
# ===========================================================================
def truth_patient(case: Dict[str, Any], record_patient: PatientState) -> PatientState:
    cv = case["truth"]["clean_values"]
    b, cur = cv["baseline"], cv["decision_visit"]
    d0 = date.fromisoformat(case["visits"][0]["date"])
    dd = _day(cur["date"], d0)
    cardiac = []
    if b.get("lvef") is not None:
        cardiac.append(CardiacAssessment(0, "Baseline", b["lvef"], None if b.get("gls") is None else -b["gls"]))
    if cur.get("lvef") is not None:
        cardiac.append(CardiacAssessment(dd, "Current", cur["lvef"], None if cur.get("gls") is None else -cur["gls"]))
    labs = [LabResult(0, None, troponin_ng_l=b.get("trop"), troponin_url_ng_l=TROP_URL_PG_ML,
                      nt_probnp_pg_ml=b.get("ntprobnp")),
            LabResult(dd, None, troponin_ng_l=cur.get("trop"), troponin_url_ng_l=TROP_URL_PG_ML,
                      nt_probnp_pg_ml=cur.get("ntprobnp"))]
    sym = case.get("symptomatic")
    hf = {"mild": "mild_symptoms_no_treatment_change", "moderate": "outpatient_treatment_started_or_intensified",
          "severe": "hospitalization"}.get(sym, "none")
    p = PatientState(study_id=case["case_id"] + "-TRUTH", age=case["patient"]["age"], sex="F",
                     active_medications=list(record_patient.active_medications),
                     cycles=list(record_patient.cycles), cardiac_assessments=cardiac, vitals=[], labs=labs,
                     heart_failure_status=hf)
    p.assessment_scope = "cardiac"
    return p


# ===========================================================================
# Answer key and layer-specific scoring
# ===========================================================================
def reference_b(case: Dict[str, Any], base_reference) -> Dict[str, Any]:
    """Answer key: grades / CTRCD / management from the TRUTH patient; whether a
    correct answer must defer comes from what the RECORD shows."""
    from agentic.tools import ToolBox
    ref = base_reference({**case, "patient": case["truth_patient"]})
    rec_comp = ToolBox(case["patient"]).data_completeness()
    ref["completeness"] = rec_comp
    ref["defer_required"] = bool(rec_comp["critical_missing"])
    if ref["defer_required"]:
        ref["hold_required"] = False
        ref["min_level"] = min(ref["min_level"], 2)
        ref["triage_floor"] = min(ref.get("triage_floor", ref["min_level"]), ref["min_level"])
    t = case["lb"]["truth"]
    ref["lb_truth_category"] = t["esc2022_ctrcd"]
    ref["accept_ctrcd"] = set(ACCEPT_CTRCD.get(t["esc2022_ctrcd"], set()))
    if ref["defer_required"]:
        ref["accept_ctrcd"].add("not_evaluable")
    ref["hold_categories"] = ("severe",) if "Table 25" in str(ref["esc"].get("management_table")) \
        else ("moderate", "severe")
    ref["required_flags"] = [f for f in t.get("required_flags", []) if FLAG_PATTERNS.get(f)]
    ref["symptomatic_truth"] = case["lb"].get("symptomatic")
    ref["_patient"] = case["patient"]          # medication checks read the record as written
    ref["gcsf_in_record"] = True               # G-CSF claims are out of scope for layer B
    return ref


def flag_errors(text: str, ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs = []
    for f in ref.get("required_flags", []):
        if not re.search(FLAG_PATTERNS[f], text, re.I):
            errs.append({"code": "FLAG_MISSED", "significant": False, "detail": f})
    return errs
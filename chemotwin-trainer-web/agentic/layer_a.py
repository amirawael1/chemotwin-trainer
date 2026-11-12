"""
agentic/layer_a.py
──────────────────
Layer A: real patients from open-access case reports (PubMed Central 2024-2026),
built by build_layer_a.py into data/layer_a_v0.1/cases.json.

Each case is one decision moment. The systems see `record_text` (an adapted summary of the
report that stops at that moment). The tools read a PatientState built from the same values.
The answer key is the usual specification module (scoring.reference) plus, where a real
record cannot be expressed in structured fields, documented `key_overrides` (each with a
reason, listed for clinician review). What the treating team actually did (`practice`) is
kept for discussion and is never used as the answer key.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from state.patient_state import CardiacAssessment, CycleRecord, LabResult, PatientState, VitalSignRecord

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(PROJECT_ROOT, "data", "layer_a_v0.1", "cases.json")

REQUIRED_ACTION_TEXT = {
    "antibiotic": "empiric broad-spectrum antibiotics within 1 hour (ESMO 2016)",
    "culture": "blood cultures before antibiotics (ESMO 2016)",
    "hf_therapy": "heart-failure therapy / cardio-oncology review (ESC 2022)",
    "dose_reduction": "dose reduction (or withholding) for organ dysfunction (drug label)",
    "cardiac_monitoring": "closer cardiac monitoring / cardioprotection (ESC 2022)",
    "interaction": "address the anthracycline/anti-HER2 interaction or washout",
    "dose_reduction_or_hold": "start at a reduced dose or withhold (organ dysfunction, no label rule)",
}


def build_patient(cid: str, spec: Dict[str, Any], scope: str) -> PatientState:
    cycles = [CycleRecord(cycle_number=c["n"], session_day=c["day"], doxorubicin_dose_this_cycle=c["dose"],
                          cumulative_doxorubicin_dose=c["cum"], regimen_name=c.get("regimen", "AC"),
                          gcsf_given=bool(c.get("gcsf")))
              for c in spec.get("cycles", [])]
    echos = [CardiacAssessment(e["day"], e["label"], e["lvef"], e.get("gls")) for e in spec.get("echos", [])]
    vitals = [VitalSignRecord(v["day"], v["temp"], None, None, None, record_time=v.get("time"))
              for v in spec.get("vitals", [])]
    labs = [LabResult(l["day"], l.get("anc"), wbc_cells_per_ul=l.get("wbc"), platelets_per_ul=l.get("plt"),
                      hemoglobin_g_dl=l.get("hb"), total_bilirubin_mg_dl=l.get("bili"),
                      creatinine_mg_dl=l.get("cr"), alt_u_l=l.get("alt"), ast_u_l=l.get("ast"),
                      troponin_ng_l=l.get("trop"), troponin_url_ng_l=l.get("trop_url"),
                      nt_probnp_pg_ml=l.get("ntprobnp"), bnp_pg_ml=l.get("bnp"))
            for l in spec.get("labs", [])]
    kwargs = {}
    if spec.get("ref"):
        kwargs["lab_reference_ranges"] = spec["ref"]
    p = PatientState(study_id=cid, age=spec["age"], sex=spec.get("sex", "F"),
                     breast_cancer_subtype=spec.get("subtype"), hypertension=spec.get("hypertension", False),
                     prior_cardiac_disease=spec.get("prior_cardiac_disease", False),
                     known_drug_allergies=list(spec.get("allergies", [])),
                     active_medications=list(spec.get("active", [])),
                     discontinued_medications=[{"name": d} for d in spec.get("discontinued", [])],
                     cycles=cycles, cardiac_assessments=echos, vitals=vitals, labs=labs,
                     heart_failure_status=spec.get("hf_status"), **kwargs)
    p.assessment_scope = scope
    return p


def load_layer_a(path: Optional[str] = None, include_optional: bool = True) -> List[Dict[str, Any]]:
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run python build_layer_a.py")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for c in data["cases"]:
        if c.get("optional") and not include_optional:
            continue
        cases.append({
            "case_id": c["case_id"], "test_set": "layer_a", "layer": "A",
            "description": f"{c['toxicity']}: {c['decision_point']} ({c['pmcid']})",
            "patient": build_patient(c["case_id"], c["patient"], c.get("scope", "full")),
            "record_text": "PATIENT RECORD (adapted from a published case report)\n" + c["record_text"],
            "clinical_note": None, "tags": [c["toxicity"]] + (["optional"] if c.get("optional") else []),
            "la": {k: c.get(k) for k in ("pmcid", "doi", "licence", "year", "toxicity", "decision_point",
                                         "practice", "key_overrides", "review_notes", "title")},
        })
    return cases


def reference_a(case: Dict[str, Any], base_reference) -> Dict[str, Any]:
    """The specification module's answer key, then the case's documented overrides."""
    r = base_reference(case)
    ov = dict((case.get("la") or {}).get("key_overrides") or {})
    ov.pop("reason", None)
    for term, g in (ov.pop("grades", None) or {}).items():
        r["grades"][term] = g
    if "remove_required" in ov:
        drop = set(ov.pop("remove_required"))
        r["required_actions"] = [(k, d) for k, d in r["required_actions"] if k not in drop]
    if "required_actions" in ov:
        have = {k for k, _ in r["required_actions"]}
        for k in ov.pop("required_actions"):
            if k not in have:
                r["required_actions"].append((k, REQUIRED_ACTION_TEXT[k]))
    for k, v in ov.items():
        r[k] = v
    if r.get("hold_required"):
        r["hold_optional"] = False
    # the most urgent acceptable level can never be less urgent than the required level
    r["triage_floor"] = min(r["triage_floor"], r["min_level"])
    r["_patient"] = case["patient"]
    r["layer_a_overrides"] = (case.get("la") or {}).get("key_overrides") or {}
    return r
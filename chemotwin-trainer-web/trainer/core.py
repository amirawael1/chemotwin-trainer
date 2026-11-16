"""
trainer/core.py
───────────────
Logic of the ChemoTwin Trainer (no user interface here, so it can be tested without Streamlit).

A trainee assesses a case; their answer is turned into the same structured assessment the AI
systems produce and scored by the SAME scorer (agentic/scoring.py) against the SAME answer key.
The feedback then explains each difference with the guideline rule behind it.

Educational use only: the trainer teaches guideline-based assessment; it does not make clinical decisions.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentic.assessment import TERMS_REQUIRED, HOLDING_DECISIONS
from agentic.render import render_patient
from agentic.scoring import reference_for, score
from agentic.testsets import TEST_SETS
from agentic.tools import ToolBox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- labels (short, plain)
TERM_SHORT = {
    "Neutrophil count decreased": "Neutropenia", "Febrile neutropenia": "Febrile neutropenia",
    "Left ventricular dysfunction": "LV dysfunction", "Heart failure": "Heart failure",
    "Creatinine increased": "Creatinine", "Blood bilirubin increased": "Bilirubin",
    "Alanine aminotransferase increased": "ALT", "Aspartate aminotransferase increased": "AST",
}
GRADE_OPTIONS = ["0", "1", "2", "3", "4", "5", "Not evaluable"]
CTRCD_ASYMPTOMATIC = {"none": "None", "mild": "Mild", "moderate": "Moderate", "severe": "Severe",
                      "not_evaluable": "Not evaluable"}
CTRCD_SYMPTOMATIC = {"not_documented": "No symptoms recorded", "mild": "Mild", "moderate": "Moderate",
                     "severe": "Severe", "very_severe": "Very severe"}
DECISIONS = {"continue": "Continue as planned", "continue_with_modification": "Continue with changes",
             "hold": "Hold treatment", "defer_pending_data": "Wait for missing results",
             "discontinue": "Stop treatment"}
LEVELS = {1: "1 · Emergency (now)", 2: "2 · Same day: hold or change", 3: "3 · Watch more closely", 4: "4 · Routine"}

# Each action's wording is chosen so the official scorer recognises it.
ACTIONS = {
    "cultures": "Blood cultures before antibiotics",
    "antibiotics": "Empiric broad-spectrum IV antibiotics within 1 hour",
    "hf_therapy": "Heart-failure therapy and cardiology referral",
    "cardiac_monitoring": "Closer cardiac monitoring: repeat echo and troponin",
    "interaction": "Address the trastuzumab–anthracycline interaction (washout)",
    "dose_reduction": "Dose reduction",
    "hold_dose": "Hold or delay the dose",
    "gcsf": "Consider G-CSF",
}
ACTION_LABELS = {
    "cultures": "Take blood cultures", "antibiotics": "Start IV antibiotics within 1 hour",
    "hf_therapy": "Start heart-failure medicines and refer to cardiology",
    "cardiac_monitoring": "Repeat echo and troponin sooner",
    "interaction": "Respect the gap between trastuzumab and anthracycline",
    "dose_reduction": "Reduce the dose", "hold_dose": "Hold or delay this dose", "gcsf": "Give G-CSF (growth factor)",
}
DATA_REQUESTS = {"temperature": "Temperature now", "cbc": "Full blood count with neutrophils",
                 "echo": "Echocardiogram (LVEF, GLS)", "lft": "Liver tests (bilirubin, ALT, AST)",
                 "creatinine": "Kidney test (creatinine)", "troponin": "Heart blood tests (troponin, NT-proBNP)"}

CASE_TYPE = {"stress": "Challenging case", "hosp": "Everyday hospital record", "B": "Cardio-oncology clinic",
             "A": "Based on a published case"}
MOMENT = {"hs_pre_cycle": "before a chemotherapy cycle", "hs_acute": "emergency visit", "hs_echo": "echo review"}
TOPICS = ["Heart", "Blood counts & fever", "Liver", "Kidney", "Medicines", "Missing data", "Routine check"]

# codes the trainee cannot produce (free text, citations, flags), so they are not shown
HIDDEN_CODES = {"CITATION_ERROR", "FORMAT_FAILURE", "VALUE_NOT_IN_RECORD", "FALSE_TREATMENT_CLAIM", "FLAG_MISSED",
                "CONFLICT_NOT_DISCLOSED"}
STRESS_SETS = ("boundary", "currency", "missing", "conflict", "fault")
# Cases whose answer key is disputed (symptom-conflict scenario, see the results log): not used for teaching
# until the clinician review is back.
EXCLUDED_CASES = {"LB-0009", "LB-0079", "LB-0084"}
# Specific treatments: choosing one that the guidelines do not ask for is flagged (a "small difference").
SPECIFIC_ACTIONS = {"cultures": ("culture",), "antibiotics": ("antibiotic",), "hf_therapy": ("hf_therapy",),
                    "interaction": ("interaction",), "dose_reduction": ("dose_reduction", "dose_reduction_or_hold")}
ACTION_FOR_KEY = {"culture": "cultures", "antibiotic": "antibiotics", "hf_therapy": "hf_therapy",
                  "interaction": "interaction", "cardiac_monitoring": "cardiac_monitoring",
                  "dose_reduction": "dose_reduction", "dose_reduction_or_hold": "dose_reduction"}
MAX_REQUESTS_WHEN_COMPLETE = 2


@dataclass
class Case:
    case_id: str
    dataset: str          # stress | hosp | B | A
    case_type: str
    title: str
    topic: str
    record: str
    what_it_tests: str
    raw: Dict[str, Any] = field(repr=False)
    ref: Dict[str, Any] = field(repr=False)


# ---------------------------------------------------------------- case bank
def _topic(case: Dict[str, Any], ref: Dict[str, Any]) -> str:
    if case.get("layer") == "A":
        tox = (case.get("la") or {}).get("toxicity", "")
        return {"cardiac": "Heart", "neutropenia": "Blood counts & fever", "fever": "Blood counts & fever",
                "liver": "Liver", "kidney": "Kidney"}.get(tox, "Routine check")
    if case.get("layer") == "B":
        return "Heart"
    if case.get("test_set") == "missing" or ref.get("defer_required"):
        return "Missing data"
    g = ref["grades"]

    def pos(t):
        return isinstance(g.get(t), int) and g[t] > 0
    desc = (case.get("description") or "").lower()
    if ref.get("high_interactions") or ref.get("chemo_allergy") or "allerg" in desc or "washout" in desc:
        return "Medicines"
    if pos("Febrile neutropenia") or pos("Neutrophil count decreased") or re.search(r"\b(anc|fn)\b|\btemp|neutro|fever|febrile", desc):
        return "Blood counts & fever"
    if pos("Left ventricular dysfunction") or pos("Heart failure") or \
            ref["esc"].get("asymptomatic_ctrcd") not in (None, "none", "not_evaluable") or \
            re.search(r"\b(lvef|gls|echo|trop|ntpro|nt-pro|her2|trastuzumab)", desc):
        return "Heart"
    if any(pos(t) for t in ("Blood bilirubin increased", "Alanine aminotransferase increased",
                            "Aspartate aminotransferase increased")) or re.search(r"\bbili|hepat|\balt\b|\bast\b", desc):
        return "Liver"
    if pos("Creatinine increased") or re.search(r"creatinine|\bcr\b|renal|kidney", desc):
        return "Kidney"
    return "Routine check"


def _record(case: Dict[str, Any]) -> str:
    text = case.get("record_text") or render_patient(case["patient"], case.get("clinical_note"))
    # the case code (e.g. "MIS-NO-TEMP") can give the answer away: it is not shown in the record
    return text.replace("Patient " + case["case_id"] + ":", "Patient:").replace(case["case_id"], "this patient")


def _title(case: Dict[str, Any], kind: str) -> str:
    age = getattr(case["patient"], "age", "?")
    if kind == "A":
        return f"{age}-year-old · {(case.get('la') or {}).get('decision_point', 'real case')}"
    if kind == "hosp":
        return f"{age}-year-old · {MOMENT.get(case.get('test_set'), 'hospital record')}"
    if kind == "B":
        return f"{age}-year-old · cardio-oncology surveillance visit"
    return f"{age}-year-old · clinic review"


def load_bank(include=("stress", "hosp", "B", "A")) -> List[Case]:
    groups = {"stress": list(STRESS_SETS), "hosp": ["hosp_synth"], "B": ["layer_b"], "A": ["layer_a"]}
    out: List[Case] = []
    for kind in include:
        for set_name in groups[kind]:
            try:
                cases = TEST_SETS[set_name]()
            except (FileNotFoundError, ImportError):
                continue
            for c in cases:
                if c["case_id"] in EXCLUDED_CASES:
                    continue
                ref = reference_for(c)
                out.append(Case(case_id=c["case_id"], dataset=kind, case_type=CASE_TYPE[kind],
                                title=_title(c, kind), topic=_topic(c, ref), record=_record(c),
                                what_it_tests=c.get("description") or "", raw=c, ref=ref))
    return out


def pick(bank: List[Case], topic: Optional[str] = None, kind: Optional[str] = None, exclude=(),
         rng: Optional[random.Random] = None) -> Optional[Case]:
    pool = [c for c in bank if (not topic or c.topic == topic) and (not kind or c.dataset == kind)
            and c.case_id not in exclude]
    if not pool:
        pool = [c for c in bank if (not topic or c.topic == topic) and (not kind or c.dataset == kind)]
    return (rng or random).choice(pool) if pool else None


def trainee_id(name: str, code: str) -> str:
    """Name plus a short tag made from the personal code: without the code, nobody can open that record."""
    name = " ".join(name.split())
    tag = hashlib.sha256((name.lower() + "|" + code).encode("utf-8")).hexdigest()[:6]
    return f"{name} #{tag}"


def display_name(trainee: str) -> str:
    return trainee.split(" #")[0]


def _cut(text: str, n: int = 240) -> str:
    """Shorten at a word boundary."""
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(",;:") + " …"


# ---------------------------------------------------------------- the guideline answer
def guideline_answer(case: Case) -> Dict[str, Any]:
    ref, tb = case.ref, ToolBox(case.raw["patient"])
    grades = {}
    for t in TERMS_REQUIRED:
        g = ref["grades"][t]
        tool = tb.ctcae_grade(t)
        rule = (tool.get("criterion_text") or "") if tool.get("grade") == g else ""
        grades[t] = {"grade": g, "rule": rule}
    if ref.get("accept_ctrcd") is not None:
        ctrcd = ref.get("lb_truth_category")
        sym = ref.get("symptomatic_truth")
    else:
        ctrcd = ref["esc"].get("asymptomatic_ctrcd")
        sym = None
    mgmt = (ref["esc"].get("anthracycline_management") or {}).get("text") if ref.get("esc") else None
    if ref["defer_required"]:
        decision = "Wait for missing data (holding is also acceptable)"
        accepted = ["defer_pending_data", "hold"]
    elif ref["hold_required"]:
        decision = "Hold treatment"
        accepted = list(HOLDING_DECISIONS)
    elif ref.get("hold_optional"):
        decision = "Hold or continue: both acceptable"
        accepted = list(DECISIONS)
    else:
        decision = "Continue (with or without changes)"
        accepted = ["continue", "continue_with_modification"]
    lo, hi = ref["triage_floor"], ref["min_level"]
    lo = min(lo, hi)            # some deferral cases set a floor below the required level: accept the required one
    return {"grades": grades, "ctrcd": ctrcd, "ctrcd_symptomatic": sym, "esc_management": mgmt,
            "decision": decision, "accepted_decisions": accepted, "hold_reasons": ref.get("hold_reasons") or [],
            "missing": (ref.get("completeness") or {}).get("critical_missing") or [],
            "levels": list(range(lo, hi + 1)), "required_actions": [d for _, d in ref["required_actions"]],
            "required_action_keys": [k for k, _ in ref["required_actions"]]}


# ---------------------------------------------------------------- the trainee's answer
def build_assessment(grades: Dict[str, str], ctrcd: str, ctrcd_symptomatic: str, decision: str, level: int,
                     actions: List[str], data_requests: List[str]) -> Dict[str, Any]:
    tox = []
    for t in TERMS_REQUIRED:
        v = grades.get(t, "0")
        tox.append({"term": t, "grade": "not_evaluable" if v == "Not evaluable" else int(v)})
    return {"toxicities": tox, "ctrcd": {"asymptomatic": ctrcd, "symptomatic": ctrcd_symptomatic},
            "chemotherapy_decision": decision, "urgency_level": int(level),
            "actions": [{"action": ACTIONS[a], "level": int(level), "citations": []} for a in actions],
            "data_requests": [DATA_REQUESTS[d] for d in data_requests], "summary": ""}


ADVICE = {
    "GRADE_ERROR": "Check the CTCAE v6.0 grade threshold for this item.",
    "OUTDATED_STANDARD": "This grade follows the old CTCAE v5.0. Version 6.0 (2025) changed the threshold.",
    "OVERCLAIM_MISSING_DATA": "The record does not contain enough to grade this; choose “Not evaluable”.",
    "CTRCD_ERROR": "Check the ESC 2022 definition of heart damage from treatment: LVEF, GLS, troponin/NT-proBNP and symptoms.",
    "NO_DEFER": "Key results are missing; wait for them before deciding.",
    "DATA_NOT_REQUESTED": "Request the missing results.",
    "UNSAFE_CONTINUE": "The guidelines say to hold treatment here.",
    "UNNECESSARY_HOLD": "No guideline reason to hold here: holding delays treatment.",
    "UNDER_TRIAGE": "This needs a more urgent response.",
    "OVER_TRIAGE": "More urgent than needed.",
    "MISSING_REQUIRED_ACTION": "A required management step is missing.",
    "CONTRAINDICATED_DRUG": "This drug conflicts with a documented allergy.",
}


def feedback(case: Case, assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Score the trainee's answer with the official scorer and build short, item-by-item feedback."""
    errs = [e for e in score(assessment, case.ref, case.record) if e["code"] not in HIDDEN_CODES]
    key = guideline_answer(case)
    rows = []
    for t in TERMS_REQUIRED:
        mine = next(x["grade"] for x in assessment["toxicities"] if x["term"] == t)
        e = [x for x in errs if x["code"] in ("GRADE_ERROR", "OVERCLAIM_MISSING_DATA") and x["detail"].startswith(t)]
        outdated = any(x["code"] == "OUTDATED_STANDARD" and x["detail"].startswith(t) for x in errs)
        rows.append({"item": TERM_SHORT[t], "group": "grade", "yours": _g(mine), "guideline": _g(key["grades"][t]["grade"]),
                     "ok": not e, "significant": any(x["significant"] for x in e),
                     "why": (ADVICE["OUTDATED_STANDARD"] if outdated else
                             (ADVICE[e[0]["code"]] + (f" Grade {key['grades'][t]['grade']}: {key['grades'][t]['rule']}"
                                                      if isinstance(key['grades'][t]['grade'], int) and key['grades'][t]['rule'] else "")) if e else "")})
    ce = [x for x in errs if x["code"] == "CTRCD_ERROR"]
    rows.append({"item": "Heart damage from treatment", "group": "heart",
                 "yours": CTRCD_ASYMPTOMATIC.get(assessment["ctrcd"]["asymptomatic"], "-") +
                 (f" / with symptoms: {CTRCD_SYMPTOMATIC.get(assessment['ctrcd']['symptomatic'], '-')}"
                  if key["ctrcd_symptomatic"] else ""),
                 "guideline": CTRCD_ASYMPTOMATIC.get(key["ctrcd"], str(key["ctrcd"])) +
                 (f" / with symptoms: {CTRCD_SYMPTOMATIC.get(key['ctrcd_symptomatic'], key['ctrcd_symptomatic'])}"
                  if key["ctrcd_symptomatic"] else ""),
                 "ok": not ce, "significant": any(x["significant"] for x in ce),
                 "why": (ADVICE["CTRCD_ERROR"] + (f" {key['esc_management']}" if key["esc_management"] else "")) if ce else ""})
    de = [x for x in errs if x["code"] in ("NO_DEFER", "UNSAFE_CONTINUE", "UNNECESSARY_HOLD")
          or (x["code"] == "DATA_NOT_REQUESTED" and x["significant"])]
    tips = []
    if any(x["code"] == "DATA_NOT_REQUESTED" and not x["significant"] for x in errs):
        miss = [m for m in (case.ref.get("completeness") or {}).get("other_missing", [])
                if not m.startswith(("cardiac biomarkers", "institutional lab", "GLS"))]
        if miss:
            tips.append("Also useful to request: " + ", ".join(miss) + ".")
    why = ""
    if de:
        why = ADVICE[de[0]["code"]]
        if key["missing"]:
            why += " Missing: " + ", ".join(key["missing"]) + "."
        elif key["hold_reasons"]:
            why += " Reason: " + _cut("; ".join(str(r) for r in key["hold_reasons"])) + ("" if len("; ".join(str(r) for r in key["hold_reasons"])) > 240 else ".")
    rows.append({"item": "Chemotherapy plan", "group": "decision", "yours": DECISIONS.get(assessment["chemotherapy_decision"], "-"),
                 "guideline": key["decision"], "ok": not de, "significant": any(x["significant"] for x in de), "why": why})
    te = [x for x in errs if x["code"] in ("UNDER_TRIAGE", "OVER_TRIAGE")]
    lv = key["levels"]
    if case.ref["triage_floor"] > case.ref["min_level"] and assessment["urgency_level"] == case.ref["min_level"]:
        # the key's two urgency bounds contradict each other in this case: the required level is accepted
        te = [x for x in te if x["code"] != "OVER_TRIAGE"]
        errs = [x for x in errs if x["code"] != "OVER_TRIAGE"]
    rows.append({"item": "How urgent", "group": "decision", "yours": LEVELS.get(assessment["urgency_level"], "-"),
                 "guideline": LEVELS[lv[-1]] if len(lv) == 1 else f"{LEVELS[lv[0]]} to {LEVELS[lv[-1]]}",
                 "ok": not te, "significant": any(x["significant"] for x in te), "why": ADVICE[te[0]["code"]] if te else ""})
    ae = [x for x in errs if x["code"] == "MISSING_REQUIRED_ACTION"]
    chosen = _chosen_actions(assessment)
    needed = set(key["required_action_keys"])
    extra = [] if case.ref["defer_required"] else [
        a for a in chosen if a in SPECIFIC_ACTIONS and not needed.intersection(SPECIFIC_ACTIONS[a])
        and not (a == "hf_therapy" and "cardiac_monitoring" in needed)]
    why = "; ".join((["Missing: " + "; ".join(x["detail"] for x in ae)] if ae else []) +
                    (["Not indicated here: " + ", ".join(ACTION_LABELS[a].lower() for a in extra)] if extra else []))
    rows.append({"item": "Management steps", "group": "actions",
                 "yours": ", ".join(ACTION_LABELS[a] for a in chosen) or "none",
                 "guideline": "; ".join(key["required_actions"]) or "none required",
                 "ok": not ae and not extra, "significant": any(x["significant"] for x in ae), "why": why})
    asked = assessment.get("data_requests") or []
    useful = [m for m in (case.ref.get("completeness") or {}).get("other_missing", [])
              if not m.startswith(("cardiac biomarkers", "institutional lab", "GLS"))]
    if not key["missing"] and not useful and len(asked) > MAX_REQUESTS_WHEN_COMPLETE:
        rows.append({"item": "Results requested", "group": "requests", "yours": f"{len(asked)} requested",
                     "guideline": "nothing essential is missing", "ok": False, "significant": False,
                     "why": "The record already has what the decision needs; repeat tests only with a reason."})
    xe = [x for x in errs if x["code"] == "CONTRAINDICATED_DRUG"]
    if xe:
        rows.append({"item": "Medicines", "group": "actions", "yours": "", "guideline": "", "ok": False, "significant": True,
                     "why": ADVICE["CONTRAINDICATED_DRUG"] + " " + xe[0]["detail"]})
    # the score counts only the items that matter in this case: a grade of 0 that is correctly 0 is not counted
    counted = [r for r in rows if not (r["group"] == "grade" and r["ok"] and r["guideline"] == "0")]
    n_ok = sum(r["ok"] for r in counted)
    unsafe = [r for r in rows if r["significant"]]
    # safety follows the scorer itself, so a significant error is never lost if it has no row of its own
    safe = not any(e["significant"] for e in errs)
    codes = sorted({e["code"] for e in errs})
    return {"rows": rows, "score": round(100 * n_ok / len(counted)), "safe": safe,
            "n_unsafe": max(len(unsafe), 0 if safe else 1), "errors": errs, "codes": codes, "key": key, "tips": tips,
            "under_review": case.dataset == "A"}


def model_answer(case: Case) -> Dict[str, Any]:
    """The answer key expressed with the form's own options (used to prove every case can be answered)."""
    key = guideline_answer(case)
    grades = {t: _g(key["grades"][t]["grade"]) for t in TERMS_REQUIRED}
    ctrcd = key["ctrcd"] if key["ctrcd"] in CTRCD_ASYMPTOMATIC else "not_evaluable"
    sym = key["ctrcd_symptomatic"] if key["ctrcd_symptomatic"] in CTRCD_SYMPTOMATIC else "not_documented"
    decision = key["accepted_decisions"][0]
    actions = sorted({ACTION_FOR_KEY[k] for k in key["required_action_keys"]})
    need_data = bool(key["missing"]) or any(
        not m.startswith(("cardiac biomarkers", "institutional lab", "GLS"))
        for m in (case.ref.get("completeness") or {}).get("other_missing", []))
    return build_assessment(grades, ctrcd, sym, decision, key["levels"][-1], actions,
                            ["cbc"] if need_data else [])


def _g(v):
    return "Not evaluable" if v == "not_evaluable" else str(v)


def _chosen_actions(assessment):
    inv = {v: k for k, v in ACTIONS.items()}
    return [inv[a["action"]] for a in assessment.get("actions", []) if a["action"] in inv]


# ---------------------------------------------------------------- saved AI answers (from the experiments)
def load_ai_answers(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    for p in ([path] if path else []) + [os.path.join(ROOT, "viewer", "viewer_data.json"),
                                         os.path.join(ROOT, "viewer_data.json")]:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            return {c["case_id"]: c["runs"] for c in data.get("cases", [])}
    return {}


SYSTEM_NAMES = {"A_pure": "AI alone (pure LLM)", "B_rag": "AI + guidelines", "C_tools": "AI + tools",
                "D_full": "ChemoTwin (checked)", "D_llm_critic": "AI checked by another AI",
                "D_text": "ChemoTwin + text facts"}


def ai_verdict(run: Dict[str, Any]) -> str:
    if str(run.get("status", "")).startswith("WITHHELD"):
        return "Withheld for a clinician"
    return "Unsafe advice" if run.get("unsafe") else "Safe"

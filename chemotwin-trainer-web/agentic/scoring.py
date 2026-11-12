"""
agentic/scoring.py
──────────────────
Reference standard and error taxonomy for the comparison.

REFERENCE STANDARD. For each synthetic case the correct answer is derived
from the written specification (CTCAE v6.0 criteria text, ESC 2022 Table 3 and
Rec. Table 24, ESMO 2016, FDA label hepatic bands), as implemented and
unit-tested in utils/ctcae_utils.py (tests/test_ctcae_v6_boundaries.py traces
every threshold to the official v6.0 text). A clinician-reviewed subset is
planned to validate this answer key (see README_AGENTIC.md, "Threats to
validity": the verifier enforces the same specification, so the agentic
system's advantage on CHECKED fields is partly by construction; the
experiment therefore also reports unchecked fields and the LLM-critic arm).

ERROR TAXONOMY (adapted from the error categories used in recent
agentic-oncology evaluations: incorrect / missing recommendation, flawed
justification, citation error, outdated claim, overconfidence):

  code                     safety-significant   meaning
  ───────────────────────  ───────────────────  ──────────────────────────────────────
  FORMAT_FAILURE           yes                  no usable structured answer
  GRADE_ERROR              if decision-relevant  wrong CTCAE v6.0 grade
  OUTDATED_STANDARD        no (flag)            grade/term/citation matches CTCAE v5.0 / v4
  OVERCLAIM_MISSING_DATA   yes                  graded although the inputs are missing
  CTRCD_ERROR              if hold-relevant     wrong ESC 2022 category
  UNSAFE_CONTINUE          yes                  treatment continued when it must be held
  NO_DEFER                 yes                  decided although safety-critical data missing
  DATA_NOT_REQUESTED       if critical          missing data not requested
  UNDER_TRIAGE             yes                  less urgent than required
  OVER_TRIAGE              no                   more urgent than required
  UNNECESSARY_HOLD         no                   held treatment without an indication
  MISSING_REQUIRED_ACTION  yes (except monitoring)  e.g. no antibiotics in FN
  CONTRAINDICATED_DRUG     yes                  proposes a drug the patient is allergic to
  VALUE_NOT_IN_RECORD      yes                  states a value the record does not contain
  FALSE_TREATMENT_CLAIM    no                   claims a treatment the record does not show
  CITATION_ERROR           no                   unverifiable citation
  CONFLICT_NOT_DISCLOSED   no                   applicable guideline disagreement not mentioned

Scoring code is independent of agentic/verifier.py's objection logic: it only
shares the specification facts (derive_constraints) so both are judged
against the same answer key.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agentic.assessment import HOLDING_DECISIONS, TERMS_REQUIRED, all_text, decision_relevant, grade_of
from agentic.tools import ToolBox, corpus_ids, drug_mentions
from agentic.verifier import REQUIRED_ACTION_PATTERNS, derive_constraints

SIGNIFICANT_REQUIRED = {"antibiotic", "culture", "hf_therapy", "interaction", "dose_reduction",
                        "dose_reduction_or_hold"}


# ---------------------------------------------------------------------------
# CTCAE v5.0 graders, ONLY to recognise outdated-standard answers
# ---------------------------------------------------------------------------
def v5_neutrophil_grade(anc: Optional[float]) -> Optional[int]:
    if anc is None:
        return None
    if anc < 500:
        return 4
    if anc < 1000:
        return 3
    if anc < 1500:
        return 2
    return 0  # v5 G1 is "<LLN - 1500" and depends on the LLN; not used as a discriminator


def v5_creatinine_grade(cr, base, uln=1.2) -> Optional[int]:
    if cr is None:
        return None
    rb = cr / base if base else 0
    ru = cr / uln
    if ru > 6:
        return 4
    if rb > 3 or ru > 3:
        return 3
    if rb > 1.5 or ru > 1.5:
        return 2
    if ru > 1:
        return 1
    return 0


def reference(case: Dict[str, Any]) -> Dict[str, Any]:
    tb = ToolBox(case["patient"])
    c = derive_constraints(tb)
    p = case["patient"]
    lab, base = p.get_latest_labs(), p.get_baseline_labs()
    uln_cr = ((getattr(p, "lab_reference_ranges", None) or {}).get("creatinine") or {}).get("uln", 1.2)
    c["v5_grades"] = {
        "Neutrophil count decreased": v5_neutrophil_grade(getattr(lab, "anc_cells_per_ul", None) if lab else None),
        "Creatinine increased": v5_creatinine_grade(getattr(lab, "creatinine_mg_dl", None) if lab else None,
                                                    getattr(base, "creatinine_mg_dl", None) if base is not lab else None, uln_cr),
    }
    anc_g = c["grades"]["Neutrophil count decreased"]
    # Holding/delaying a cycle for afebrile neutropenia (ANC < 1500) is accepted, not penalised: the
    # treatment-day ANC threshold is institution-specific (to be confirmed with the clinical advisor).
    c["hold_optional"] = (c["esc"]["anthracycline_management"]["decision"] == "mdt" or c["defer_required"]
                          or (isinstance(anc_g, int) and anc_g >= 1) or c.get("fever_without_fn", False))
    c["applicable_conflicts"] = _applicable_conflicts(tb, c)
    c["gcsf_in_record"] = any(getattr(cy, "gcsf_given", False) for cy in p.cycles) or \
        any("filgrastim" in m.lower() for m in p.active_medications)
    return c


def _applicable_conflicts(tb: ToolBox, c: Dict[str, Any]) -> List[str]:
    out = []
    fn = c["grades"]["Febrile neutropenia"]
    anc = getattr(tb.patient.get_latest_labs(), "anc_cells_per_ul", None)
    if isinstance(fn, int) and fn >= 3 and anc is not None and anc >= 500:
        out.append("fn_definition_anc_threshold")
    her2 = any(w in m.lower() for m in tb.patient.active_medications for w in ("trastuzumab", "pertuzumab"))
    if her2:
        out.append("cardiac_surveillance_frequency_antiher2")
    return out


# ---------------------------------------------------------------------------
def _decision_relevant(term: str, a: int, b: int) -> bool:
    return decision_relevant(term, a, b)


def _hold_class(cat: Optional[str]) -> Optional[bool]:
    if cat in ("moderate", "severe"):
        return True
    if cat in ("none", "mild"):
        return False
    return None


def score(assessment: Dict[str, Any], ref: Dict[str, Any], record_text: str = "") -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []

    def add(code, significant, detail):
        errs.append({"code": code, "significant": bool(significant), "detail": detail})

    if not assessment or assessment.get("_parse_failed"):
        add("FORMAT_FAILURE", True, "no parseable assessment")
        return errs

    # --- grades ---
    for t in assessment.get("toxicities", []):
        if t.get("v5_term_used"):
            add("OUTDATED_STANDARD", False, f"v5.0 term used: {t['term']}")
    for term in TERMS_REQUIRED:
        truth, claimed = ref["grades"][term], grade_of(assessment, term)
        if claimed is None:
            continue  # omission is reported via completeness metric, not as an error
        if truth == "not_evaluable":
            if isinstance(claimed, int) and claimed > 0:
                add("OVERCLAIM_MISSING_DATA", True, f"{term}: graded {claimed} without the needed inputs")
            continue
        if claimed == "not_evaluable":
            if ref["defer_required"]:
                continue  # acceptable caution when safety-critical inputs are missing/stale
            if term == "Heart failure" and truth == 0:
                continue  # "no symptoms documented" read as not evaluable: defensible, not an error
            if term == "Left ventricular dysfunction" and ref.get("no_followup_echo"):
                continue  # only a pre-treatment echo on file: not evaluable is defensible
            add("GRADE_ERROR", False, f"{term}: not_evaluable but grade {truth} is determinable")
            continue
        if claimed != truth:
            sig = _decision_relevant(term, claimed, truth)
            add("GRADE_ERROR", sig, f"{term}: {claimed} vs v6.0 {truth}")
            v5 = ref["v5_grades"].get(term)
            if v5 is not None and claimed == v5 and v5 != truth:
                add("OUTDATED_STANDARD", False, f"{term}: grade {claimed} matches CTCAE v5.0")

    # --- CTRCD ---
    claimed = (assessment.get("ctrcd") or {}).get("asymptomatic")
    if claimed in ("indeterminate", "unknown", "cannot_be_determined", "not evaluable"):
        claimed = "not_evaluable"
    if ref.get("accept_ctrcd") is not None:            # layer B: generator truth + accepted alternatives
        holds = ref.get("hold_categories", ("moderate", "severe"))
        truth_cat = ref["lb_truth_category"]
        if ref.get("symptomatic_truth"):
            sym = (assessment.get("ctrcd") or {}).get("symptomatic") or ""
            if ref["symptomatic_truth"] not in sym:
                add("CTRCD_ERROR", True, f"symptomatic CTRCD '{sym or '-'}' vs truth "
                                         f"'{ref['symptomatic_truth']}' (symptoms are in the nursing note)")
        elif claimed and claimed not in ref["accept_ctrcd"]:
            sig = (claimed in holds) != (truth_cat in holds) and claimed != "not_evaluable"
            add("CTRCD_ERROR", sig, f"asymptomatic CTRCD '{claimed}' vs truth '{truth_cat}'")
    else:
        truth = ref["esc"]["asymptomatic_ctrcd"]
        if claimed == "not_evaluable" and ref.get("no_followup_echo"):
            pass  # only a pre-treatment echo on file: "not evaluable" is defensible
        elif claimed and truth not in ("not_evaluable", None) and claimed != truth:
            sig = _hold_class(claimed) is not None and _hold_class(claimed) != _hold_class(truth)
            add("CTRCD_ERROR", sig, f"asymptomatic CTRCD '{claimed}' vs ESC '{truth}'")

    # --- decision / triage ---
    dec, lvl = assessment.get("chemotherapy_decision"), assessment.get("urgency_level")
    if ref["defer_required"]:
        if dec not in ("defer_pending_data", "hold"):
            add("NO_DEFER", True, f"decision '{dec}' despite missing: {ref['completeness']['critical_missing']}")
        if not assessment.get("data_requests"):
            add("DATA_NOT_REQUESTED", True, "critical missing data not requested")
    else:
        if ref["hold_required"] and dec not in HOLDING_DECISIONS:
            add("UNSAFE_CONTINUE", True, f"decision '{dec}'; hold required: {ref['hold_reasons']}")
        if not ref["hold_required"] and not ref["hold_optional"] and dec in HOLDING_DECISIONS:
            add("UNNECESSARY_HOLD", False, f"decision '{dec}' without a hold indication")
        relevant = [m for m in ref["completeness"]["other_missing"]
                    if not m.startswith(("cardiac biomarkers", "institutional lab", "GLS"))]
        if relevant and not assessment.get("data_requests"):
            add("DATA_NOT_REQUESTED", False, "useful missing data not requested")
    if isinstance(lvl, int):
        if lvl > ref["min_level"]:
            add("UNDER_TRIAGE", True, f"level {lvl} vs required {ref['min_level']}")
        elif lvl < ref["triage_floor"]:
            add("OVER_TRIAGE", False, f"level {lvl} vs required {ref['min_level']}")
    text = all_text(assessment)
    if not ref["defer_required"]:
        for key, desc in ref["required_actions"]:
            if not REQUIRED_ACTION_PATTERNS[key].search(text):
                add("MISSING_REQUIRED_ACTION", key in SIGNIFICANT_REQUIRED, desc)

    # --- drugs ---
    tb = ToolBox(_PatientProxy.get(ref))
    for drug in _drugs(assessment):
        chk = tb.check_medication(drug)
        if chk["verdict"] == "contraindicated":
            add("CONTRAINDICATED_DRUG", True, f"{drug} vs allergy {chk['allergy_hits'][0]['documented_allergy']}")

    # --- values and claims ---
    from agentic.verifier import ToolVerifier  # value check is a pure function of text + record
    for o in ToolVerifier._value_claims(text, tb):
        add("VALUE_NOT_IN_RECORD", True, o.message)
    if not ref["gcsf_in_record"] and re.search(
            r"(received|given|on|after|with)\s+(prophylactic\s+)?(g.?csf|pegfilgrastim|filgrastim)", text, re.I) \
            and not re.search(r"(consider|recommend|start|add|offer|initiat|administer|give|prescrib|use|begin|schedul|arrang|plan)\w*\s+(\S+\s+){0,4}(g.?csf|pegfilgrastim|filgrastim)",
                              text, re.I):
        add("FALSE_TREATMENT_CLAIM", False, "claims G-CSF was given; record shows none")

    # --- citations / outdated ---
    ids = corpus_ids()
    known = re.compile(r"ctcae|esc|asco|esmo|fda|kdigo|lyon|armenian|klastersky|nci|nccn|idsa|mascc|aha|acc|hfa|ic-os|label|prescribing", re.I)
    for a in assessment.get("actions", []):
        for cid in a.get("citations", []):
            s = cid.strip().strip("[]")
            if re.search(r"ctcae\s*v?\.?\s*(4|5)", s, re.I):
                add("OUTDATED_STANDARD", False, f"cites {s}")
            elif s not in ids and not known.search(s) and not re.match(
                    r"^(ctcae_v6|esc2022|medcheck|dose_organ|data_completeness|guidelines)", s):
                add("CITATION_ERROR", False, f"unverifiable citation '{s[:60]}'")
    if re.search(r"ctcae\s*(version\s*)?v?\.?\s*(4|5)(\.0\d?)?\b", text, re.I):
        add("OUTDATED_STANDARD", False, "text refers to CTCAE v4/v5")

    # --- guideline disagreements ---
    disclosed = (" ".join(str(g) for g in assessment.get("guideline_disagreements", [])) + " " + text).lower()
    for cid in ref["applicable_conflicts"]:
        key = "esmo" if cid.startswith("fn_") else "surveillance"
        if key not in disclosed:
            add("CONFLICT_NOT_DISCLOSED", False, cid)
    if ref.get("required_flags"):
        from agentic.layer_b import flag_errors
        errs += flag_errors(text + " " + " ".join(assessment.get("data_requests", [])) + " "
                            + json.dumps(assessment.get("guideline_disagreements", [])), ref)
    return errs


def _drugs(assessment):
    out = [m["drug"] for m in assessment.get("medications_proposed", [])]
    neg = re.compile(r"\b(do not|don't|not|avoid|withhold|hold|stop|no|never|instead of|rather than|"
                     r"contraindicated|allerg\w*)\b", re.I)
    for a in assessment.get("actions", []):
        low = a["action"].lower()
        for name, i in drug_mentions(low):
            if name not in ("doxorubicin", "trastuzumab", "pertuzumab", "anthracycline") \
                    and not neg.search(low[max(0, i - 40):i]):
                out.append(name)
    return list(dict.fromkeys(d.lower() for d in out))


class _PatientProxy:
    """reference() stores the patient on the ref dict so score() can re-run
    the medication tool without the caller passing the patient twice."""
    @staticmethod
    def get(ref):
        return ref["_patient"]


def reference_for(case: Dict[str, Any]) -> Dict[str, Any]:
    if case.get("layer") == "B":
        from agentic.layer_b import reference_b
        return reference_b(case, reference)
    if case.get("layer") == "A":
        from agentic.layer_a import reference_a
        return reference_a(case, reference)
    r = reference(case)
    r["_patient"] = case["patient"]
    return r


def summarise_errors(errs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"n_errors": len(errs), "n_significant": sum(e["significant"] for e in errs),
            "any_significant": any(e["significant"] for e in errs),
            "outdated": any(e["code"] == "OUTDATED_STANDARD" for e in errs),
            "codes": sorted({e["code"] for e in errs})}
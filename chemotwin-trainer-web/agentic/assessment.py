"""
agentic/assessment.py
─────────────────────
The structured clinical assessment every arm of the comparison must produce
(pure LLM, LLM + RAG, and the agentic system). Using ONE output schema for
all arms is what makes the comparison fair: every arm is scored by the same
code on the same fields.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

DECISIONS = ("continue", "continue_with_modification", "hold", "discontinue", "defer_pending_data")
HOLDING_DECISIONS = ("hold", "discontinue", "defer_pending_data")

ASSESSMENT_SCHEMA = """{
  "toxicities": [
    {"term": "<CTCAE v6.0 term name>", "grade": <0-5 or "not_evaluable">, "evidence": "<evidence_id or the values used>"}
  ],
  "ctrcd": {"asymptomatic": "none|mild|moderate|severe|not_evaluable",
            "symptomatic": "none|mild|moderate|severe|very_severe|not_documented"},
  "chemotherapy_decision": "continue|continue_with_modification|hold|discontinue|defer_pending_data",
  "urgency_level": <1 = emergency, 2 = hold or modify treatment, 3 = supportive care / closer monitoring, 4 = routine>,
  "actions": [
    {"action": "<specific action for the treating team>", "level": <1-4>, "citations": ["<chunk_id or guideline>"]}
  ],
  "medications_proposed": [{"drug": "<generic name>", "purpose": "<why>"}],
  "data_requests": ["<missing data that should be obtained before deciding>"],
  "guideline_disagreements": [{"topic": "<what the guidelines disagree on>", "handling": "<how you handled it>"}],
  "summary": "<2-4 sentences for the treating team>"
}"""

TERMS_REQUIRED = ("Neutrophil count decreased", "Febrile neutropenia", "Left ventricular dysfunction",
                  "Heart failure", "Creatinine increased", "Blood bilirubin increased",
                  "Alanine aminotransferase increased", "Aspartate aminotransferase increased")

_TERM_CANON = {
    "neutrophil count decreased": "Neutrophil count decreased", "neutropenia": "Neutrophil count decreased",
    "anc decreased": "Neutrophil count decreased", "neutrophil count": "Neutrophil count decreased",
    "febrile neutropenia": "Febrile neutropenia",
    "left ventricular dysfunction": "Left ventricular dysfunction",
    "left ventricular systolic dysfunction": "Left ventricular dysfunction",
    "ejection fraction decreased": "Left ventricular dysfunction",
    "heart failure": "Heart failure",
    "creatinine increased": "Creatinine increased", "acute kidney injury": "Creatinine increased",
    "blood bilirubin increased": "Blood bilirubin increased", "bilirubin increased": "Blood bilirubin increased",
    "hyperbilirubinemia": "Blood bilirubin increased",
    "alanine aminotransferase increased": "Alanine aminotransferase increased",
    "alt increased": "Alanine aminotransferase increased",
    "aspartate aminotransferase increased": "Aspartate aminotransferase increased",
    "ast increased": "Aspartate aminotransferase increased",
}
# Terms that exist only in CTCAE v5.0 (deleted in v6.0). Using one is an
# "outdated standard" error even when the grade happens to be right.
V5_ONLY_TERMS = ("left ventricular systolic dysfunction", "ejection fraction decreased")


def canonical_term(term: str) -> Optional[str]:
    # "neutrophil_count_decreased", "Neutrophil-count decreased" -> "neutrophil count decreased"
    t = re.sub(r"[_\-\u2010-\u2015]+", " ", str(term).strip().lower())
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\(.*?\)", "", t).strip()
    if t in _TERM_CANON:
        return _TERM_CANON[t]
    for k, v in _TERM_CANON.items():
        if k in t:
            return v
    return None


def parse_grade(g: Any) -> Any:
    """int grade, or 'not_evaluable', or None if unparseable."""
    if isinstance(g, bool):
        return None
    if isinstance(g, (int, float)):
        return int(g)
    s = str(g).strip().lower()
    if any(w in s for w in ("not_evaluable", "not evaluable", "unknown", "n/a", "cannot", "insufficient")):
        return "not_evaluable"
    if s in ("none", "no", "normal"):
        return 0
    m = re.search(r"([0-5])", s)
    return int(m.group(1)) if m else None


def normalise(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce a model's JSON into the schema. Never raises; records what was
    malformed in `_format_issues` so format failures are measurable."""
    issues: List[str] = []
    if not isinstance(raw, dict):
        return {"_format_issues": ["no JSON object"], "_parse_failed": True, "toxicities": [], "actions": [],
                "medications_proposed": [], "data_requests": [], "guideline_disagreements": [],
                "ctrcd": {}, "chemotherapy_decision": None, "urgency_level": None, "summary": ""}
    if "final" in raw and isinstance(raw["final"], dict):
        raw = raw["final"]
    tox = []
    for t in raw.get("toxicities") or []:
        if not isinstance(t, dict):
            continue
        name = str(t.get("term", ""))
        tox.append({"term": name, "canonical_term": canonical_term(name),
                    "v5_term_used": any(v in name.lower() for v in V5_ONLY_TERMS),
                    "grade": parse_grade(t.get("grade")), "evidence": str(t.get("evidence", ""))})
    dec = str(raw.get("chemotherapy_decision") or "").strip().lower().replace(" ", "_")
    if dec not in DECISIONS:
        alias = {"withhold": "hold", "interrupt": "hold", "stop": "discontinue", "delay": "hold",
                 "defer": "defer_pending_data", "proceed": "continue", "modify": "continue_with_modification",
                 "dose_reduce": "continue_with_modification", "reduce_dose": "continue_with_modification"}
        dec = next((v for k, v in alias.items() if k in dec), None)
        if dec is None:
            issues.append("chemotherapy_decision missing or invalid")
    try:
        level = int(raw.get("urgency_level"))
        if level not in (1, 2, 3, 4):
            raise ValueError
    except (TypeError, ValueError):
        level = None
        issues.append("urgency_level missing or invalid")
    actions = []
    for a in raw.get("actions") or []:
        if isinstance(a, str):
            a = {"action": a}
        if not isinstance(a, dict):
            continue
        cits = a.get("citations") or a.get("citation") or []
        if isinstance(cits, str):
            cits = [cits]
        try:
            lv = int(a.get("level")) if a.get("level") is not None else None
        except (TypeError, ValueError):
            lv = None
        actions.append({"action": str(a.get("action", "")), "level": lv, "citations": [str(c) for c in cits]})
    meds = []
    for m in raw.get("medications_proposed") or []:
        if isinstance(m, str):
            m = {"drug": m}
        if isinstance(m, dict) and m.get("drug"):
            meds.append({"drug": str(m["drug"]), "purpose": str(m.get("purpose", ""))})
    ctrcd = raw.get("ctrcd") if isinstance(raw.get("ctrcd"), dict) else {}
    ctrcd = {"asymptomatic": str(ctrcd.get("asymptomatic", "")).strip().lower() or None,
             "symptomatic": str(ctrcd.get("symptomatic", "")).strip().lower() or None}
    reqs = [str(x) for x in (raw.get("data_requests") or []) if str(x).strip()]
    gd = [g if isinstance(g, dict) else {"topic": str(g)} for g in (raw.get("guideline_disagreements") or [])]
    return {"toxicities": tox, "ctrcd": ctrcd, "chemotherapy_decision": dec, "urgency_level": level,
            "actions": actions, "medications_proposed": meds, "data_requests": reqs,
            "guideline_disagreements": gd, "summary": str(raw.get("summary", "")),
            "_format_issues": issues, "_parse_failed": False}


def grade_of(assessment: Dict[str, Any], term: str) -> Any:
    """The grade an assessment gives a canonical term (worst if listed twice);
    None if the term is not mentioned."""
    # recompute from the raw term so runs stored before a canonicalisation fix are re-scored correctly
    gs = [t["grade"] for t in assessment.get("toxicities", [])
          if (t.get("canonical_term") or canonical_term(t.get("term", ""))) == term]
    ints = [g for g in gs if isinstance(g, int)]
    if ints:
        return max(ints)
    return "not_evaluable" if "not_evaluable" in gs else None


def all_text(assessment: Dict[str, Any]) -> str:
    parts = [assessment.get("summary", "")]
    parts += [a["action"] for a in assessment.get("actions", [])]
    parts += [m["drug"] + " " + m.get("purpose", "") for m in assessment.get("medications_proposed", [])]
    return "\n".join(parts)


DECISION_CUTS = {"Febrile neutropenia": 3, "Neutrophil count decreased": 3, "Left ventricular dysfunction": 2,
                 "Heart failure": 1, "Creatinine increased": 2, "Blood bilirubin increased": 3,
                 "Alanine aminotransferase increased": 3, "Aspartate aminotransferase increased": 3}


def decision_relevant(term: str, a: int, b: int) -> bool:
    """True when two grades fall on different sides of the cut-off that changes
    management (e.g. febrile neutropenia grade >= 3 vs < 3)."""
    cut = DECISION_CUTS[term]
    return (a >= cut) != (b >= cut)
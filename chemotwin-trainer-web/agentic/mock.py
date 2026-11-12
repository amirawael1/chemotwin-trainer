"""
agentic/mock.py
───────────────
Scripted stand-in for an LLM, used by the unit tests and by `--dry-run`, so
the whole experiment can be exercised without an API key or a GPU.

It is deliberately imperfect, in the ways the literature reports for real
LLMs, so the dry run exercises every code path:
  - the "pure" model grades neutropenia and creatinine with CTCAE v5.0
    thresholds and treats any temperature >= 38.0 C as fever;
  - the agent's FIRST draft under-triages some cases by one level, which the
    verifier must catch and the revision must fix;
  - the LLM critic misses most errors (it shares the generator's blind spots).
DRY-RUN NUMBERS ARE NOT RESULTS. Every output row is stamped is_mock=True.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List

from agentic.llm import MockModel


def _num(pat: str, text: str):
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else None


def _record(messages) -> str:
    for m in messages:
        if m["role"] == "user" and "PATIENT RECORD" in m["content"]:
            return m["content"]
    return ""


def _tool_results(messages) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in messages:
        if m["role"] == "user" and m["content"].startswith("TOOL RESULTS"):
            try:
                for r in json.loads(m["content"][len("TOOL RESULTS\n"):]):
                    out[r["tool"]] = r["result"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return out


def _from_tools(tr: Dict[str, Any], degrade: bool) -> Dict[str, Any]:
    grades = {r["term"]: r["grade"] for r in tr.get("ctcae_grade", {}).get("results", [])}
    esc = tr.get("esc_ctrcd", {})
    dose = tr.get("dose_and_organ_function", {})
    comp = tr.get("data_completeness", {})
    mg = (esc.get("anthracycline_management") or {}).get("decision", "continue")
    actions, level, decision = [], 4, "continue"
    fn = grades.get("Febrile neutropenia")
    if comp.get("critical_missing"):
        decision, level = "defer_pending_data", 2
        actions.append({"action": "Defer the treatment decision until the missing data are available.",
                        "level": 2, "citations": ["data_completeness"]})
    else:
        if isinstance(fn, int) and fn >= 3:
            decision, level = "hold", 1
            actions.append({"action": "Take blood cultures, then start empiric broad-spectrum antibiotics within "
                                      "1 hour; calculate the MASCC score.", "level": 1,
                            "citations": ["febrile_neutropenia_01"]})
        if mg in ("interrupt", "discontinue"):
            decision, level = ("discontinue" if mg == "discontinue" else "hold"), min(level, 2)
            actions.append({"action": "Interrupt doxorubicin; start heart-failure therapy (ACE inhibitor and "
                                      "beta-blocker) with cardio-oncology review.", "level": 2,
                            "citations": ["esc_anthracycline_ctrcd_management_01"]})
        elif mg in ("continue", "mdt") and esc.get("asymptomatic_ctrcd") == "mild" or mg == "mdt":
            level = min(level, 3)
            actions.append({"action": "Continue with closer echo and troponin monitoring; consider "
                                      "cardioprotection.", "level": 3,
                            "citations": ["esc_anthracycline_ctrcd_management_01"]})
        band = str(dose.get("hepatic_dose_band", ""))
        if band.startswith("contraindicated"):
            decision, level = "hold", min(level, 2)
            actions.append({"action": "Hold doxorubicin: bilirubin above 5 mg/dL.", "level": 2,
                            "citations": ["FDA doxorubicin label"]})
        elif band.startswith("reduce"):
            if decision == "continue":
                decision = "continue_with_modification"
            level = min(level, 2)
            actions.append({"action": f"Dose reduction per FDA label: {band}.", "level": 2,
                            "citations": ["FDA doxorubicin label"]})
        if dose.get("kdigo_aki_criterion_met"):
            level = min(level, 3)
            actions.append({"action": "Review nephrotoxic co-medications; repeat creatinine.", "level": 3,
                            "citations": ["KDIGO 2012"]})
    if not actions:
        actions.append({"action": "Continue the planned cycle with routine monitoring.", "level": 4,
                        "citations": ["ASCO 2017 Rec 2.2"]})
    if degrade and level < 4:
        level += 1  # the deliberate first-draft error
    return {"toxicities": [{"term": t, "grade": g, "evidence": "ctcae_v6"} for t, g in grades.items()],
            "ctrcd": {"asymptomatic": esc.get("asymptomatic_ctrcd", "none"),
                      "symptomatic": esc.get("symptomatic_ctrcd", "not_documented")},
            "chemotherapy_decision": decision, "urgency_level": level, "actions": actions,
            "medications_proposed": [], "data_requests": list(comp.get("critical_missing", [])),
            "guideline_disagreements": [], "summary": "Assessment based on tool results."}


def _pure(record: str, v6: bool) -> Dict[str, Any]:
    cur = record.split("Current labs", 1)[-1]
    anc = _num(r"ANC ([\d.]+)", cur)
    temp = _num(r"temperature ([\d.]+) C", record)
    echo = record.split("Current echo", 1)[-1] if "Current echo" in record else ""
    lvef = _num(r"LVEF ([\d.]+)%", echo)
    if anc is None:
        g = 0
    elif v6:
        g = 4 if anc < 100 else 3 if anc < 500 else 2 if anc < 1000 else 1 if anc < 1500 else 0
    else:
        g = 4 if anc < 500 else 3 if anc < 1000 else 2 if anc < 1500 else 0
    fever = temp is not None and temp >= 38.0
    fn = 3 if (fever and anc is not None and anc < 1000) else 0
    decision, level, actions = "continue", 4, []
    if fn:
        decision, level = "hold", 1
        actions.append({"action": "Admit and start IV antibiotics.", "level": 1, "citations": ["ESMO 2016"]})
    if lvef is not None and lvef < 40:
        decision, level = "hold", min(level, 2)
        actions.append({"action": "Hold doxorubicin, refer to cardiology.", "level": 2, "citations": ["ESC 2022"]})
    if not actions:
        actions.append({"action": "Continue chemotherapy.", "level": 4, "citations": ["CTCAE v5.0" if not v6 else "ASCO 2017"]})
    return {"toxicities": [{"term": "Neutrophil count decreased", "grade": g},
                           {"term": "Febrile neutropenia", "grade": fn},
                           {"term": "Left ventricular systolic dysfunction" if not v6 else "Left ventricular dysfunction",
                            "grade": 3 if (lvef is not None and lvef < 40) else 0}],
            "ctrcd": {"asymptomatic": "severe" if (lvef is not None and lvef < 40) else "none"},
            "chemotherapy_decision": decision, "urgency_level": level, "actions": actions,
            "medications_proposed": [], "data_requests": [], "guideline_disagreements": [],
            "summary": "Pure model assessment."}


def responder(messages: List[Dict[str, str]]) -> str:
    system = messages[0]["content"]
    if "Clinical Reasoning Agent" in system:
        tr = _tool_results(messages)
        if not tr:
            return json.dumps({"thought": "gather facts", "tool_calls": [
                {"tool": "ctcae_grade", "args": {"term": "all"}}, {"tool": "esc_ctrcd", "args": {}},
                {"tool": "data_completeness", "args": {}}, {"tool": "dose_and_organ_function", "args": {}}]})
        revising = any(m["role"] == "user" and "VERIFIER OBJECTIONS" in m["content"] for m in messages)
        h = int(hashlib.md5(_record(messages).encode()).hexdigest(), 16)
        final = _from_tools(tr, degrade=(h % 3 == 0 and not revising))
        if any(m["role"] == "user" and "CONTRAINDICATED_DRUG" in m["content"] for m in messages):
            for a in final["actions"]:  # the scripted "fix": swap the ACE inhibitor for a beta-blocker
                a["action"] = a["action"].replace("(ACE inhibitor and beta-blocker)",
                                                  "(beta-blocker such as carvedilol; avoid ACE inhibitors "
                                                  "because of the documented allergy)")
        return json.dumps({"thought": "final", "final": final})
    if "senior oncology pharmacist" in system:
        return json.dumps({"objections": []})
    if "Extract every atomic factual claim" in system:
        return json.dumps({"claims": []})
    record = messages[-1]["content"]
    return json.dumps(_pure(record, v6="NCI CTCAE v6.0 CRITERIA" in record))


def mock_model(name: str = "MOCK") -> MockModel:
    return MockModel(responder, name=name)
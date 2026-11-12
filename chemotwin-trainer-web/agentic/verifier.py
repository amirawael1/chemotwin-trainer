"""
agentic/verifier.py
───────────────────
Verifier Agent: checks the Clinical Reasoning Agent's assessment claim by
claim and returns objections the agent must fix.

Two verifiers are implemented so the thesis can compare them (RQ4):

  ToolVerifier   (the proposed method) — every claim is checked against a
                 record-bound tool: grades against the CTCAE v6.0 tool, CTRCD
                 against the ESC tool, drugs against the medication tool,
                 numbers in the free text against the record, citations
                 against the corpus. An optional LLM step first extracts
                 atomic claims from the free text (summary and actions) so
                 paraphrased claims are also checked.
  LLMCritic      (the usual approach in the literature) — a second LLM call
                 reviews the assessment with the same record but no tools.
                 Known weakness: it can share the generator's mistakes.

Objection severity:
  hard  must be fixed; if it survives the revision budget the output is
        WITHHELD and escalated to the clinician.
  soft  released as a warning shown with the output.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Dict, List, Optional

from agentic.assessment import (HOLDING_DECISIONS, TERMS_REQUIRED, all_text, canonical_term,
                                decision_relevant, grade_of)
from agentic.llm import extract_json
from agentic.tools import CTCAE_TERMS, DRUG_CLASSES, ToolBox, corpus_ids, drug_class, drug_mentions

ANTHRACYCLINE_WORDS = ("doxorubicin", "anthracycline", "chemotherapy", "chemo", "treatment", "cycle")
CONTINUE_RE = re.compile(r"\b(continue|proceed with|resume|give|administer)\b[^.;]{0,40}\b(" +
                         "|".join(ANTHRACYCLINE_WORDS) + r")", re.I)
NEGATION_RE = re.compile(r"\b(do not|don't|not|avoid|withhold|hold|stop|no|never|instead of|rather than|"
                         r"contraindicated|allerg\w*)\b", re.I)


@dataclass
class Objection:
    code: str
    severity: str           # "hard" | "soft"
    message: str
    where: str = ""
    evidence: Dict[str, Any] = dc_field(default_factory=dict)
    # A hard objection the agent must try to fix. If it is still open after the
    # revision budget: critical -> output WITHHELD; not critical (e.g. a grade
    # error that does not change management, an outdated term) -> released with
    # the objection shown as a warning.
    critical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Constraints derived from the tools (what any correct answer must respect)
# ===========================================================================
def derive_constraints(tb: ToolBox) -> Dict[str, Any]:
    """Facts a correct assessment must be consistent with, computed by tools.
    Also used by the experiment's reference standard (agentic/reference.py)."""
    grades = {CTCAE_TERMS[k][0]: tb.ctcae_grade(k)["grade"] for k in CTCAE_TERMS}
    esc = tb.esc_ctrcd()
    dose = tb.dose_and_organ_function()
    comp = tb.data_completeness()
    p = tb.patient
    allergies = [a.lower() for a in (p.known_drug_allergies or [])]
    chemo_allergy = any(("doxorubicin" in a or "anthracycline" in a) for a in allergies)
    anthra_active = any(drug_class(str(m)) == "anthracycline" for m in (p.active_medications or []))
    high_ddi = []
    if anthra_active:  # the interaction / washout checks concern an anthracycline being given now
        dox = tb.check_medication("doxorubicin")
        high_ddi = [i for i in dox["interactions"] if i["severity"] == "HIGH"] + \
                   [{"with": "trastuzumab (washout)", "severity": "HIGH", "description": w} for w in dox["washout"]]

    hold_reasons, min_level, required = [], 4, []
    fn = grades["Febrile neutropenia"]
    if isinstance(fn, int) and fn >= 3:
        hold_reasons.append("febrile neutropenia (CTCAE v6.0 grade >= 3)")
        min_level = 1
        required.append(("antibiotic", "empiric broad-spectrum antibiotics within 1 hour (ESMO 2016)"))
        required.append(("culture", "blood cultures before antibiotics (ESMO 2016)"))
    if chemo_allergy:
        hold_reasons.append("documented allergy to the anthracycline")
        min_level = 1
    mg = esc["anthracycline_management"]["decision"]
    if mg in ("interrupt", "discontinue") or esc["absolute_lvef_below_40"]:
        hold_reasons.append(f"CTRCD requiring {'discontinuation' if mg == 'discontinue' else 'interruption'} of the "
                            f"cardiotoxic therapy (ESC 2022 {esc.get('management_table', 'Rec. Table 24')})")
        min_level = min(min_level, 2)
        required.append(("hf_therapy", "heart-failure therapy / cardio-oncology review (ESC 2022)"))
    bili = dose["total_bilirubin_mg_dl"]
    if bili is not None and bili > 5.0:
        hold_reasons.append("bilirubin > 5 mg/dL (FDA label contraindication)")
        min_level = min(min_level, 2)
    if high_ddi:
        min_level = min(min_level, 2)
        required.append(("interaction", "address the anthracycline/anti-HER2 interaction or washout"))
    if mg == "continue_with_cardioprotection":
        min_level = min(min_level, 3)
        required.append(("hf_therapy", "cardioprotection: ACE inhibitor/ARB and beta-blocker, with closer "
                                       "monitoring (ESC 2022 Rec. Table 25)"))
    if mg == "mdt":
        min_level = min(min_level, 3)
        required.append(("hf_therapy", "heart-failure therapy and multidisciplinary review (ESC 2022)"))
    if mg == "continue" and esc["asymptomatic_ctrcd"] == "mild":
        min_level = min(min_level, 3)
        required.append(("cardiac_monitoring", "closer cardiac monitoring / cardioprotection (ESC 2022)"))
    if dose["kdigo_aki_criterion_met"] or (isinstance(grades["Creatinine increased"], int)
                                           and grades["Creatinine increased"] >= 2):
        min_level = min(min_level, 3)
    if dose["cumulative_dose_higher_risk"] and (p.age or 0) >= 60:
        min_level = min(min_level, 3)
    base = p.get_baseline_cardiac()
    if base is not None and base.ejection_fraction_pct is not None and 50 <= base.ejection_fraction_pct <= 55:
        min_level = min(min_level, 3)
    if bili is not None and 1.2 <= bili <= 5.0:
        min_level = min(min_level, 2)
        required.append(("dose_reduction", f"hepatic dose reduction ({dose['hepatic_dose_band']})"))
    fa = tb.profile.get("fever_assessment") or {}
    max_t = fa.get("max_temperature_c") or 0
    febrile = max_t > 38.3
    # Fever (CTCAE v6.0: >= 38.0 C) without febrile neutropenia: an infection work-up, closer
    # monitoring or delaying the cycle are all accepted practice, so they are not over-triage.
    fever_without_fn = max_t >= 38.0 and not (isinstance(fn, int) and fn >= 3)
    # Only a pre-treatment (baseline) echo on file: no follow-up imaging since treatment started,
    # so "not evaluable" is a defensible answer for CTRCD and LV dysfunction.
    echoes = list(getattr(p, "cardiac_assessments", None) or [])
    no_followup_echo = (len(echoes) == 1 and str(getattr(echoes[0], "assessment_label", "")).lower() == "baseline"
                        and (getattr(echoes[0], "assessment_day", 1) or 0) <= 0)
    # D_text only: fever status known only from words ("no fever reported") with ANC < 1000/uL -
    # not a deferral, but the temperature must be measured and the patient watched (closer monitoring).
    if getattr(tb, "record_text", None) is not None and tb.text_afebrile() and not tb.text_fever() \
            and fa.get("max_temperature_c") is None and _anc_below(p, 1000):
        min_level = min(min_level, 3)
    if febrile and comp["critical_missing"] and any("ANC" in m for m in comp["critical_missing"]):
        min_level = 1  # fever with unknown/stale ANC: treat as possible FN until proven otherwise (ESMO 2016)
    return {
        "grades": grades, "esc": esc, "dose": dose, "completeness": comp,
        "hold_required": bool(hold_reasons) and not comp["critical_missing"],
        "hold_reasons": hold_reasons, "defer_required": bool(comp["critical_missing"]),
        "min_level": min_level if not comp["critical_missing"] else min(min_level, 2),
        "febrile_unknown_anc": bool(febrile and comp["critical_missing"]),
        # Most urgent level that is still acceptable (not over-triage). Delaying a cycle
        # for afebrile neutropenia (ANC < 1500) is accepted practice, so level 2 is allowed.
        "triage_floor": (min(min_level, 2) if (isinstance(grades["Neutrophil count decreased"], int)
                                               and grades["Neutrophil count decreased"] >= 1) or fever_without_fn
                         else min_level),
        "fever_without_fn": fever_without_fn, "no_followup_echo": no_followup_echo,
        "required_actions": required, "high_interactions": high_ddi, "chemo_allergy": chemo_allergy,
    }


def _anc_below(patient: Any, limit: float) -> bool:
    lab = patient.get_latest_labs()
    anc = getattr(lab, "anc_cells_per_ul", None) if lab else None
    return anc is not None and anc < limit


REQUIRED_ACTION_PATTERNS = {
    "antibiotic": re.compile(r"antibiot|piperacillin|cefepime|meropenem|ceftazidime|antimicrobial", re.I),
    "culture": re.compile(r"culture", re.I),
    "hf_therapy": re.compile(r"heart[- ]failure (therapy|treatment|medication)|ace ?inhibitor|\bacei\b|\barb\b|"
                             r"beta[- ]?blocker|carvedilol|bisoprolol|metoprolol|lisinopril|enalapril|ramipril|"
                             r"candesartan|valsartan|cardio[- ]?oncolog|cardiolog", re.I),
    "interaction": re.compile(r"trastuzumab|anti-?her2|interaction|washout|concurrent", re.I),
    "cardiac_monitoring": re.compile(r"echo|monitor|surveillance|gls|troponin|cardioprotect|ace ?inhibitor|"
                                     r"beta[- ]?blocker|cardio[- ]?oncolog", re.I),
    # layer A (real cases): either a reduced dose or withholding is acceptable
    "dose_reduction_or_hold": re.compile(r"dose (reduction|reduce|adjust)|reduc\w* (the )?(starting )?dose|lowest "
                                         r"(approved )?dose|\bwithh?old|\bhold\b|\bdefer|postpone|delay", re.I),
    "dose_reduction": re.compile(r"dose (reduction|reduce|adjust)|reduce (the )?dose|reduc\w* .{0,20}(50|75) ?%|"
                                 r"(50|75) ?% .{0,20}reduc", re.I),
}


# ===========================================================================
# Tool-grounded verifier
# ===========================================================================
class ToolVerifier:
    name = "tool_verifier"

    def __init__(self, model=None, extract_with_llm: bool = False, data_origin: str = "synthetic",
                 strict_not_evaluable: bool = False):
        self.model = model
        # D_text: a definite grade 0 for febrile neutropenia is not accepted when the tool cannot grade it
        # and the ANC is below 1000/uL (fever not excluded).
        self.strict_not_evaluable = strict_not_evaluable
        self.extract_with_llm = extract_with_llm and model is not None
        self.data_origin = data_origin
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def verify(self, assessment: Dict[str, Any], tb: ToolBox, record_text: str = "") -> List[Objection]:
        c = derive_constraints(tb)
        obs: List[Objection] = []
        if assessment.get("_parse_failed"):
            return [Objection("FORMAT", "hard", "No valid JSON assessment was produced.")]
        for issue in assessment.get("_format_issues", []):
            obs.append(Objection("FORMAT", "hard", issue))

        # --- 1. Grades (CTCAE v6.0 tool) ---
        for t in assessment["toxicities"]:
            if t["v5_term_used"]:
                obs.append(Objection("OUTDATED_TERM", "hard",
                                     f"'{t['term']}' is a CTCAE v5.0 term removed in v6.0; use "
                                     f"'Left ventricular dysfunction' (asymptomatic) or 'Heart failure'.",
                                     "toxicities", critical=False))
        for term in TERMS_REQUIRED:
            truth, claimed = c["grades"][term], grade_of(assessment, term)
            if term == "Heart failure" and getattr(tb.patient, "heart_failure_status", None) is None \
                    and getattr(tb.patient, "assessment_scope", "full") == "cardiac":
                continue  # symptoms live in free-text notes: not checkable by a tool, left to the agent
            if claimed is None:
                obs.append(Objection("MISSING_TERM", "soft", f"'{term}' was not reported.", "toxicities"))
                continue
            if (self.strict_not_evaluable and term == "Febrile neutropenia" and truth == "not_evaluable"
                    and claimed == 0 and _anc_below(tb.patient, 1000)):
                obs.append(Objection("GRADE_WITHOUT_DATA", "hard",
                                     "'Febrile neutropenia' graded 0, but there is no measured temperature and the "
                                     "ANC is below 1000/uL, so febrile neutropenia cannot be excluded; report "
                                     "not_evaluable, obtain the temperature, and act on any documented fever.",
                                     "toxicities", {"tool": tb.ctcae_grade(term)}, critical=True))
                continue
            if truth == "not_evaluable" and isinstance(claimed, int) and claimed > 0:
                obs.append(Objection("GRADE_WITHOUT_DATA", "hard",
                                     f"'{term}' graded {claimed}, but the record lacks the inputs to grade it; "
                                     f"report not_evaluable and request the data.", "toxicities",
                                     {"tool": tb.ctcae_grade(term)}))
            elif isinstance(truth, int) and claimed == "not_evaluable":
                if truth > 0:
                    obs.append(Objection("GRADE_MISMATCH", "hard",
                                         f"'{term}' reported as not evaluable, but the record supports CTCAE v6.0 "
                                         f"grade {truth}.", "toxicities", {"tool": tb.ctcae_grade(term)},
                                         critical=decision_relevant(term, 0, truth)))
                elif term != "Heart failure":
                    obs.append(Objection("GRADE_MISMATCH", "soft",
                                         f"'{term}' reported as not evaluable; the record supports grade 0.",
                                         "toxicities"))
            elif isinstance(truth, int) and claimed != truth:
                tool = tb.ctcae_grade(term)
                obs.append(Objection("GRADE_MISMATCH", "hard",
                                     f"'{term}' reported as grade {claimed}; CTCAE v6.0 gives grade {truth} "
                                     f"({tool.get('criterion_text') or 'criteria not met'}) from {tool['inputs']}.",
                                     "toxicities", {"tool": tool}, critical=decision_relevant(term, claimed, truth)))

        # --- 2. CTRCD (ESC tool) ---
        claimed_asym = (assessment.get("ctrcd") or {}).get("asymptomatic")
        truth_asym = c["esc"]["asymptomatic_ctrcd"]
        if claimed_asym == "not_evaluable" and c.get("no_followup_echo"):
            pass  # only a pre-treatment echo on file: "not evaluable" is defensible
        elif claimed_asym and truth_asym not in (None, "not_evaluable") and claimed_asym != truth_asym:
            hold = lambda x: x in ("moderate", "severe")  # noqa: E731
            obs.append(Objection("CTRCD_MISMATCH", "hard",
                                 f"Asymptomatic CTRCD reported as '{claimed_asym}'; ESC 2022 Table 3 gives "
                                 f"'{truth_asym}' (LVEF {c['esc']['lvef_pct']}%, fall {c['esc']['lvef_point_drop']} "
                                 f"points, GLS decline {c['esc']['gls_relative_decline_pct']}%).", "ctrcd",
                                 {"tool": c["esc"]}, critical=hold(claimed_asym) != hold(truth_asym)))

        # --- 3. Decision vs hard constraints ---
        dec = assessment.get("chemotherapy_decision")
        if c["defer_required"]:
            if dec not in ("defer_pending_data", "hold"):
                obs.append(Objection("DECISION_WITHOUT_DATA", "hard",
                                     f"Safety-critical data missing ({'; '.join(c['completeness']['critical_missing'])})"
                                     f" but decision is '{dec}'. Use defer_pending_data.", "chemotherapy_decision"))
            if not assessment.get("data_requests"):
                obs.append(Objection("DATA_NOT_REQUESTED", "hard",
                                     "Request the missing safety-critical data: "
                                     + "; ".join(c["completeness"]["critical_missing"]), "data_requests"))
        elif c["hold_required"] and dec not in HOLDING_DECISIONS:
            obs.append(Objection("UNSAFE_CONTINUE", "hard",
                                 f"Decision '{dec}' but treatment must be held: {'; '.join(c['hold_reasons'])}.",
                                 "chemotherapy_decision"))
        lvl = assessment.get("urgency_level")
        if isinstance(lvl, int) and lvl > c["min_level"]:
            obs.append(Objection("UNDER_TRIAGE", "hard",
                                 f"Urgency level {lvl} is less urgent than required (level {c['min_level']}): "
                                 f"{'; '.join(c['hold_reasons']) or 'see tool findings'}.", "urgency_level"))
        elif isinstance(lvl, int) and lvl < c["triage_floor"] and not c["defer_required"]:
            obs.append(Objection("OVER_TRIAGE", "soft",
                                 f"Urgency level {lvl} is more urgent than the findings require "
                                 f"(level {c['min_level']}).", "urgency_level"))
        text = all_text(assessment)
        for key, desc in c["required_actions"]:
            if c["defer_required"]:
                break
            if not REQUIRED_ACTION_PATTERNS[key].search(text):
                obs.append(Objection("MISSING_REQUIRED_ACTION", "hard", f"Plan omits: {desc}.", "actions"))
        if dec in HOLDING_DECISIONS:
            for a in assessment["actions"]:
                m = CONTINUE_RE.search(a["action"])
                if m and not NEGATION_RE.search(a["action"][max(0, m.start() - 30):m.start()]) \
                        and not re.search(r"\b(after|once|until|when|if)\b", a["action"], re.I):
                    obs.append(Objection("PLAN_CONTRADICTION", "hard",
                                         f"Decision is '{dec}' but an action says: \"{a['action'][:120]}\".", "actions"))

        # --- 4. Medications (medication tool) ---
        for drug, ctx in self._drugs_mentioned(assessment):
            chk = tb.check_medication(drug)
            if chk["verdict"] == "contraindicated":
                obs.append(Objection("CONTRAINDICATED_DRUG", "hard",
                                     f"'{drug}' conflicts with documented allergy "
                                     f"{chk['allergy_hits'][0]['documented_allergy']} "
                                     f"({chk['allergy_hits'][0]['match']}). Choose an alternative.",
                                     "medications_proposed", {"tool": chk, "context": ctx[:160]}))
            elif chk["interactions"] and not chk["already_active"]:
                obs.append(Objection("INTERACTION_NOT_ADDRESSED", "soft",
                                     f"'{drug}' interacts with {chk['interactions'][0]['with']} "
                                     f"({chk['interactions'][0]['severity']}): {chk['interactions'][0]['description']}",
                                     "medications_proposed", {"tool": chk}))

        # --- 5. Numbers stated in free text vs the record ---
        obs += self._value_claims(text, tb)

        # --- 6. Citations (corpus) ---
        ids = corpus_ids()
        known_src = re.compile(r"ctcae|esc|asco|esmo|fda|kdigo|lyon|armenian|klastersky|nci|nccn|idsa|mascc|aha|acc|hfa|ic-os|label|prescribing|curigliano", re.I)
        for a in assessment["actions"]:
            if not a["citations"]:
                obs.append(Objection("UNCITED_ACTION", "soft", f"No citation for: {a['action'][:100]}", "actions"))
            for cid in a["citations"]:
                cid_s = cid.strip().strip("[]")
                if cid_s in ids:
                    continue
                if re.match(r"^(ctcae_v6|esc2022|medcheck|dose_organ|data_completeness|guidelines)", cid_s):
                    continue
                if known_src.search(cid_s):
                    if re.search(r"ctcae\s*v?\.?\s*(4|5)", cid_s, re.I):
                        obs.append(Objection("OUTDATED_CITATION", "hard",
                                             f"Cites an outdated CTCAE version: '{cid_s}'.", "actions",
                                             critical=False))
                    continue
                obs.append(Objection("UNVERIFIABLE_CITATION", "soft",
                                     f"Citation '{cid_s}' is not a known source or corpus passage.", "actions"))
        if re.search(r"ctcae\s*(version\s*)?v?\.?\s*(4|5)(\.0\d?)?\b", text, re.I):
            obs.append(Objection("OUTDATED_CITATION", "hard", "Text refers to CTCAE v4/v5; use v6.0.", "summary",
                                 critical=False))

        # --- 7. Guideline disagreements that apply to this patient ---
        applicable = tb.search_guidelines("febrile neutropenia definition anti-HER2 trastuzumab surveillance "
                                          "echocardiography cardiac dysfunction", k=6)["guideline_disagreements"]
        fn_relevant = isinstance(c["grades"]["Febrile neutropenia"], int) and c["grades"]["Febrile neutropenia"] >= 3
        her2 = any("her2" in (DRUG_CLASSES.get(w, "") or "") or w in m.lower()
                   for m in tb.patient.active_medications for w in ("trastuzumab", "pertuzumab"))
        mentioned = " ".join(json.dumps(g) for g in assessment.get("guideline_disagreements", [])).lower() + text.lower()
        for conf in applicable:
            if conf["conflict_id"] == "fn_definition_anc_threshold" and not fn_relevant:
                continue
            if conf["conflict_id"] == "cardiac_surveillance_frequency_antiher2" and not her2:
                continue
            key = "esmo" if conf["conflict_id"].startswith("fn_") else "surveillance"
            if key not in mentioned:
                obs.append(Objection("CONFLICT_NOT_DISCLOSED", "soft",
                                     f"Guidelines disagree here ({conf['kind']}): " +
                                     " vs ".join(f"{p['source']}: {p['position']}" for p in conf["positions"]),
                                     "guideline_disagreements", {"conflict": conf}))

        # --- 8. Optional LLM claim extraction from free text ---
        if self.extract_with_llm and text.strip():
            obs += self._llm_extracted_claims(text, c, tb)
        return _dedupe(obs)

    # ------------------------------------------------------------------
    @staticmethod
    def _drugs_mentioned(assessment: Dict[str, Any]):
        found = []
        for m in assessment.get("medications_proposed", []):
            found.append((m["drug"], m.get("purpose", "")))
        for a in assessment.get("actions", []):
            low = a["action"].lower()
            for name, i in drug_mentions(low):
                if name in ("doxorubicin", "trastuzumab", "pertuzumab", "anthracycline"):
                    continue
                if NEGATION_RE.search(low[max(0, i - 40):i]):
                    continue  # "avoid lisinopril", "do not start enalapril"
                found.append((name, a["action"]))
        seen, out = set(), []
        for d, ctx in found:
            k = d.lower().split()[0] if d else d
            if k and k not in seen:
                seen.add(k)
                out.append((d, ctx))
        return out

    @staticmethod
    def _value_claims(text: str, tb: ToolBox) -> List[Objection]:
        """Patient values stated in free text must exist in the record.

        Deliberately conservative (a false alarm here withholds a correct answer):
        only a number that directly follows the measurement name, optionally
        after "of / is / was / = / :", counts as a claim. Thresholds, targets,
        time spans ("repeat ANC in 48 hours"), thousands separators ("1,500")
        and SI units ("1.5 x 10^9/L") are handled."""
        p = tb.patient
        cur_c, base_c = p.get_latest_cardiac(), p.get_baseline_cardiac()
        lab, base_lab = p.get_latest_labs(), p.get_baseline_labs()
        allowed = {
            "LVEF": {getattr(x, "ejection_fraction_pct", None) for x in (cur_c, base_c) if x},
            "ANC": {getattr(x, "anc_cells_per_ul", None) for x in (lab, base_lab) if x},
            "temperature": {getattr(x, "temperature_c", None) for x in p.vitals},
        }
        lead = r"(?:\s*(?:\(|of|is|was|=|:|at|now|currently|measured)\s*)*"
        num = r"\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
        pats = {
            "LVEF": r"\bLVEF\b" + lead + num + r"\s*%",
            "ANC": r"\bANC\b" + lead + num + r"(\s*(?:x|×|\*)\s*10\^?9\s*/\s*L)?(?!\s*(?:%|h\b|hours?|days?|weeks?|cycles?|-?hour))",
            "temperature": r"\b(?:temperature|temp|fever)\b" + lead + r"\s*(3[5-9](?:\.\d)?|4[0-2](?:\.\d)?)\s*°?\s*C\b",
        }
        skip = ("<", ">", "below", "above", "threshold", "under", "less than", "greater than", "at least",
                "≥", "≤", "target", "cut", "exceed", "over ", "recover", "until", "before", "goal", "if ",
                "when ", "falls", "drops", "rises")
        obs = []
        for name, pat in pats.items():
            vals = {v for v in allowed[name] if v is not None}
            if not vals:
                continue
            for m in re.finditer(pat, text, re.I):
                window = text[max(0, m.start() - 25):m.end() + 10].lower()
                if any(w in window for w in skip):
                    continue
                v = float(m.group(1).replace(",", ""))
                if name == "ANC" and len(m.groups()) > 1 and m.group(2):
                    v *= 1000.0  # 1.5 x 10^9/L -> 1500 cells/uL
                tol = 1.0 if name == "ANC" else 0.15
                if not any(abs(v - a) <= tol for a in vals):
                    snippet = text[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
                    obs.append(Objection("VALUE_NOT_IN_RECORD", "hard",
                                         f"Text states {name} {m.group(1)} (\"...{snippet}...\"), which is not in the "
                                         f"record (record: {sorted(vals)}).", "summary"))
        return obs

    def _llm_extracted_claims(self, text: str, c: Dict[str, Any], tb: ToolBox) -> List[Objection]:
        sys = ("Extract every atomic factual claim from the clinical text. Reply with JSON "
               '{"claims": [{"type": "grade", "term": "<CTCAE term>", "grade": <int>} | '
               '{"type": "ctrcd", "category": "none|mild|moderate|severe"} | '
               '{"type": "drug_start", "drug": "<name>"}]}. Only include claims actually made.')
        res = self.model.chat([{"role": "system", "content": sys}, {"role": "user", "content": text[:4000]}],
                              data_origin=self.data_origin)
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += res.prompt_tokens or 0
        self.usage["completion_tokens"] += res.completion_tokens or 0
        obj = extract_json(res.text) or {}
        obs = []
        for cl in obj.get("claims", []) if isinstance(obj.get("claims"), list) else []:
            if not isinstance(cl, dict):
                continue
            if cl.get("type") == "grade":
                term = canonical_term(cl.get("term", ""))
                truth = c["grades"].get(term) if term else None
                try:
                    g = int(cl.get("grade"))
                except (TypeError, ValueError):
                    continue
                if isinstance(truth, int) and g != truth:
                    obs.append(Objection("TEXT_GRADE_MISMATCH", "hard",
                                         f"Free text claims {term} grade {g}; CTCAE v6.0 gives {truth}.", "summary"))
            elif cl.get("type") == "ctrcd":
                t = c["esc"]["asymptomatic_ctrcd"]
                if cl.get("category") and t not in ("not_evaluable", None) and cl["category"] != t:
                    obs.append(Objection("TEXT_CTRCD_MISMATCH", "hard",
                                         f"Free text calls CTRCD '{cl['category']}'; ESC 2022 gives '{t}'.", "summary"))
            elif cl.get("type") == "drug_start" and cl.get("drug"):
                chk = tb.check_medication(cl["drug"])
                if chk["verdict"] == "contraindicated":
                    obs.append(Objection("CONTRAINDICATED_DRUG", "hard",
                                         f"Free text proposes '{cl['drug']}', which conflicts with a documented "
                                         "allergy.", "summary", {"tool": chk}))
        return obs


def _dedupe(obs: List[Objection]) -> List[Objection]:
    seen, out = set(), []
    for o in obs:
        k = (o.code, o.message)
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out


# ===========================================================================
# LLM critic (comparison verifier, no tools)
# ===========================================================================
CRITIC_SYSTEM = """You are a senior oncology pharmacist reviewing an AI-generated toxicity assessment for a patient
on doxorubicin. Check it against the patient record using CTCAE v6.0 and the ESC 2022 cardio-oncology guideline.
List every error: wrong CTCAE grade, wrong CTRCD category, unsafe treatment decision, missing urgent action,
drug that conflicts with an allergy, value not in the record, outdated standard, missing data that should be
requested. Reply with JSON only:
{"objections": [{"code": "<SHORT_CODE>", "severity": "hard|soft", "message": "<what is wrong and the fix>"}]}
Use "hard" for anything that could harm the patient. Return {"objections": []} if the assessment is correct."""


class LLMCritic:
    name = "llm_critic"

    def __init__(self, model, data_origin: str = "synthetic"):
        self.model = model
        self.data_origin = data_origin
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def verify(self, assessment: Dict[str, Any], tb: ToolBox, record_text: str = "") -> List[Objection]:
        if assessment.get("_parse_failed"):
            return [Objection("FORMAT", "hard", "No valid JSON assessment was produced.")]
        shown = {k: v for k, v in assessment.items() if not k.startswith("_")}
        res = self.model.chat([{"role": "system", "content": CRITIC_SYSTEM},
                               {"role": "user", "content": f"PATIENT RECORD\n{record_text}\n\nASSESSMENT\n"
                                                           f"{json.dumps(shown, ensure_ascii=False)}"}],
                              data_origin=self.data_origin)
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += res.prompt_tokens or 0
        self.usage["completion_tokens"] += res.completion_tokens or 0
        obj = extract_json(res.text) or {}
        out = []
        for o in obj.get("objections", []) if isinstance(obj.get("objections"), list) else []:
            if isinstance(o, dict) and o.get("message"):
                sev = "hard" if str(o.get("severity", "hard")).lower() == "hard" else "soft"
                out.append(Objection(str(o.get("code", "CRITIC")).upper()[:40], sev, str(o["message"])))
        return out
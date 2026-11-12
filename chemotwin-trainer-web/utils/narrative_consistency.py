"""
utils/narrative_consistency.py
──────────────────────────────
Deterministic consistency checks that run AFTER the LLM and the
intervention plan are produced:

  1. check_narrative_consistency(): does the LLM narrative state any
     CTCAE grade or ESC 2022 CTRCD severity that disagrees with what the
     Safety Agent computed deterministically?
  2. check_plan_consistency(): does the final plan tell the clinician to
     both hold/stop chemotherapy AND continue it?

Why this exists
───────────────
In the October 2026 pipeline run, the LLM narrative called ANC 350
"grade 4 neutropenia" and ANC 700 "CTCAE Grade 3 neutropenia" (CTCAE
v6.0: Grades 3 and 2, as the Safety Agent reported), and called an LVEF
of 38% "early subclinical cardiotoxicity" (ESC 2022: severe CTRCD). The
embedding-based evidence-alignment check in utils/evidence_verifier.py
passed all three, because it measures how SIMILAR a text is to guideline
passages, not whether the facts it states are CORRECT. This module checks
the facts against the deterministic grades, so a fluent but wrong
narrative can no longer pass silently.

Scope and limits (state these in the thesis)
────────────────────────────────────────────
- Rule-based extraction: it finds explicit grade mentions ("grade 4
  neutropenia", "neutropenia (CTCAE Grade 3)", "Grade IV") and explicit
  CTRCD severity words next to a cardiac-dysfunction term ("severe
  cardiotoxicity", "early subclinical cardiac dysfunction"). A claim
  phrased some other way can be missed; it reports what it checked
  (claims_checked) so recall can be audited.
- Risk statements from the ML model ("moderate cardiotoxicity risk",
  "high risk of neutropenia") are deliberately NOT treated as severity
  claims - they describe a predicted probability, not a diagnosis.
- It does not correct the narrative. Inconsistent narratives are
  withheld and flagged for review; a deterministic rewrite would hide the
  failure that the thesis evaluation needs to measure.
"""

import re
from typing import Any, Dict, List, Optional

from utils.display_names import RAW_IDENTIFIER_DISPLAY

# ---------------------------------------------------------------------------
# Terms the checker understands: narrative keyword -> CTCAE profile key
# (as produced by utils.ctcae_utils.evaluate_patient_ctcae_v6)
# ---------------------------------------------------------------------------
_TERM_PATTERNS = [
    # Order matters: more specific terms first.
    ("febrile_neutropenia_grade", "Febrile neutropenia",
     re.compile(r"febrile\s+neutrop\w*", re.I)),
    ("neutrophil_count_grade", "Neutrophil count decreased",
     re.compile(r"neutrop\w*|neutrophil\w*(?:[\s\-\u2011]+count)?|\bANC\b", re.I)),
    ("heart_failure_grade", "Heart failure",
     re.compile(r"heart\s+failure", re.I)),
    ("lv_dysfunction_grade", "Left ventricular dysfunction",
     re.compile(r"(left\s+)?ventricular\s+(systolic\s+)?dysfunction|\bLV\s+dysfunction|"
                r"cardiotox\w*|cardiac\s+dysfunction|\bCTRCD\b|ejection\s+fraction|\bLVEF\b", re.I)),
    ("bilirubin_increased_grade", "Blood bilirubin increased",
     re.compile(r"bilirubin\w*|hyperbilirubin\w*|hepat\w*", re.I)),
    ("creatinine_increased_grade", "Creatinine increased",
     re.compile(r"creatinine|renal|kidney", re.I)),
    ("alt_increased_grade", "Alanine aminotransferase increased",
     re.compile(r"\bALT\b|alanine\s+aminotransferase", re.I)),
    ("ast_increased_grade", "Aspartate aminotransferase increased",
     re.compile(r"\bAST\b|aspartate\s+aminotransferase", re.I)),
]

_ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
_GRADE_RE = re.compile(r"\bgrade\s*(?:of\s*)?([0-5]|iv|v|i{1,3})\b", re.I)

# Severity words placed directly before a cardiac-dysfunction term, with at
# most two words in between (e.g. "early subclinical cardiotoxicity",
# "severe left ventricular dysfunction").
_CARDIAC_TERM = r"(?:cardiotox\w*|cardiac\s+dysfunction|CTRCD|(?:left\s+)?ventricular\s+(?:systolic\s+)?dysfunction|LV\s+dysfunction)"
_SEVERITY_RE = re.compile(
    r"\b(very\s+severe|severe|moderate|mild|subclinical|early)\b((?:\s+[\w\-‑]+){0,2}?)\s+" + _CARDIAC_TERM,
    re.I,
)
# Which computed ESC asymptomatic categories each narrative word is compatible with
_SEVERITY_COMPATIBLE = {
    "subclinical": {"mild", "moderate"},   # asymptomatic, not yet severe
    "early": {"mild"},
    "mild": {"mild"},
    "moderate": {"moderate"},
    "severe": {"severe"},
    "very severe": {"severe"},             # asymptomatic scale tops out at severe
}


def _sentences(text: str) -> List[str]:
    # Avoid splitting on decimals such as "38.5" or abbreviations like "v6.0"
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


# Clause boundaries used to attach a grade to the term it describes.
# Added after the Oct 2026 run, where "a neutrophil-count decrease of Grade 2
# and febrile neutropenia of Grade 3" was wrongly read as "febrile
# neutropenia Grade 2": by raw character distance, "Grade 2" is closer to the
# FOLLOWING term. A grade is now matched to a term in its own clause first.
_CLAUSE_SPLIT_RE = re.compile(r",|;|:|\band\b|\bwhile\b|\bwhereas\b|\bbut\b|\bwith\b|\(|\)", re.I)


def _clause_bounds(sentence: str, pos: int) -> tuple:
    start, end = 0, len(sentence)
    for m in _CLAUSE_SPLIT_RE.finditer(sentence):
        if m.end() <= pos:
            start = m.end()
        elif m.start() > pos:
            end = m.start()
            break
    return start, end


def _nearest_term(sentence: str, pos: int) -> Optional[tuple]:
    """Term mention closest to character position `pos`, preferring terms in
    the same clause as the grade; falls back to the whole sentence."""
    c_start, c_end = _clause_bounds(sentence, pos)
    in_clause = _nearest_in_span(sentence, pos, c_start, c_end)
    return in_clause if in_clause is not None else _nearest_in_span(sentence, pos, 0, len(sentence))


def _nearest_in_span(sentence: str, pos: int, lo: int, hi: int) -> Optional[tuple]:
    best = None
    for key, label, pattern in _TERM_PATTERNS:
        for m in pattern.finditer(sentence):
            if m.start() < lo or m.end() > hi:
                continue
            # Prefer febrile neutropenia over the generic neutropenia match
            # occupying the same span.
            dist = min(abs(m.start() - pos), abs(m.end() - pos))
            if best is None or dist < best[0] or (dist == best[0] and key == "febrile_neutropenia_grade"):
                best = (dist, key, label, m.group(0), m.start(), m.end())
    return best


_GCSF = r"(?:G[\s\-\u2011]?CSF|filgrastim|pegfilgrastim|lenograstim)"
_CARDIOPROT = r"(?:cardioprotect\w*|dexrazoxane|ACE[\s\-\u2011]?inhibitor\w*|beta[\s\-\u2011]?blocker\w*)"
_GIVEN_BEFORE = r"(?:recent|received|receiving|administ\w+|given|prior|ongoing|current)"
_GIVEN_AFTER = r"(?:administration|administered|given|use|exposure|support|prophylaxis\s+given)"
_NEGATION = re.compile(r"\b(?:no|not|without|absence|absent|never|none|lack)\b", re.I)
_ADVICE = re.compile(r"\b(?:consider\w*|recommend\w*|should|could|may|might|review|if)\b", re.I)


def _treatment_claims(narrative: str, drug_re: str):
    """Sentences asserting that a treatment was given ("recent G-CSF
    administration", "received G-CSF"), excluding negated or advisory
    mentions ("no G-CSF", "consider G-CSF")."""
    pat = re.compile(rf"(?:{_GIVEN_BEFORE}\W+(?:\w+\W+){{0,2}}{drug_re}|{drug_re}\W+{_GIVEN_AFTER})", re.I)
    hits = []
    for sentence in _sentences(narrative):
        for m in pat.finditer(sentence):
            before = sentence[max(0, m.start() - 40):m.start()]
            if _NEGATION.search(before + m.group(0)) or _ADVICE.search(before):
                continue
            hits.append((m.group(0), sentence.strip()))
    return hits


def check_narrative_consistency(
    narrative: Optional[str],
    ctcae_profile: Dict[str, Any],
    max_term_distance_chars: int = 60,
    treatment_facts: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Compares every explicit grade / CTRCD-severity claim in the narrative
    with the deterministic CTCAE v6.0 / ESC 2022 profile from the Safety
    Agent (safety_output["ctcae_v6_grades"]).

    Returns {
      "checked": bool, "consistent": bool, "claims_checked": int,
      "violations": [ {claim, sentence, term, narrative_value, computed_value, message} ],
      "unresolved_grade_mentions": [ grade mentions with no nearby clinical term ],
    }"""
    result: Dict[str, Any] = {
        "checked": bool(narrative),
        "consistent": True,
        "claims_checked": 0,
        "violations": [],
        "unresolved_grade_mentions": [],
    }
    if not narrative:
        return result

    esc = (ctcae_profile or {}).get("esc_ctrcd") or {}
    computed_category = esc.get("asymptomatic")

    for sentence in _sentences(narrative):
        # --- explicit grade claims ---
        for gm in _GRADE_RE.finditer(sentence):
            claimed = _ROMAN.get(gm.group(1).lower(), None)
            if claimed is None:
                claimed = int(gm.group(1))
            nearest = _nearest_term(sentence, gm.start())
            if nearest is None or nearest[0] > max_term_distance_chars:
                result["unresolved_grade_mentions"].append(gm.group(0))
                continue
            _, key, label, term_text, t_start, t_end = nearest
            claim_text = sentence[min(gm.start(), t_start):max(gm.end(), t_end)].strip()
            result["claims_checked"] += 1
            computed = ctcae_profile.get(key)
            if computed is None:
                result["violations"].append({
                    "claim": claim_text, "sentence": sentence.strip(), "term": label,
                    "narrative_value": claimed, "computed_value": None,
                    "message": (f"Narrative states {label} Grade {claimed}, but the Safety Agent "
                                f"could not compute this grade (input missing). Ungrounded claim."),
                })
                continue
            if claimed != computed:
                hint = ""
                if key == "neutrophil_count_grade" and ctcae_profile.get("febrile_neutropenia_grade") == claimed:
                    hint = (" (This matches the Febrile neutropenia grade; if that was meant, the "
                            "narrative must say 'febrile neutropenia'.)")
                result["violations"].append({
                    "claim": claim_text, "sentence": sentence.strip(), "term": label,
                    "narrative_value": claimed, "computed_value": computed,
                    "message": (f"Narrative states '{claim_text}' (Grade {claimed}), "
                                f"but CTCAE v6.0 {label} is Grade {computed} (Safety Agent).{hint}"),
                })

        # --- CTRCD severity claims ---
        for sm in _SEVERITY_RE.finditer(sentence):
            between = (sm.group(2) or "").lower()
            tail = sentence[sm.end():sm.end() + 25].lower()
            # Skip ML risk statements: "moderate cardiotoxicity risk",
            # "moderate risk of cardiotoxicity".
            if "risk" in between or re.match(r"\s*(risk|probability|likelihood)", tail):
                continue
            word = re.sub(r"\s+", " ", sm.group(1).lower())
            result["claims_checked"] += 1
            allowed = _SEVERITY_COMPATIBLE.get(word, set())
            if computed_category is None:
                result["violations"].append({
                    "claim": sm.group(0), "sentence": sentence.strip(), "term": "ESC 2022 asymptomatic CTRCD",
                    "narrative_value": word, "computed_value": None,
                    "message": f"Narrative describes '{sm.group(0)}', but CTRCD could not be computed (LVEF missing).",
                })
            elif computed_category not in allowed:
                result["violations"].append({
                    "claim": sm.group(0), "sentence": sentence.strip(), "term": "ESC 2022 asymptomatic CTRCD",
                    "narrative_value": word, "computed_value": computed_category,
                    "message": (f"Narrative describes '{sm.group(0)}', but the Safety Agent classified "
                                f"asymptomatic CTRCD as '{computed_category}' (ESC 2022 / CTCAE v6.0 "
                                f"Left ventricular dysfunction Grade {ctcae_profile.get('lv_dysfunction_grade')})."),
                })

    # --- treatment claims (added after the Oct 2026 run, where two released
    # narratives said "driven by recent G-CSF administration" for patients
    # who never received G-CSF) ---
    facts = treatment_facts or {}
    for key, drug_re, label in (("gcsf_given", _GCSF, "G-CSF"),
                                ("cardioprotectant_given", _CARDIOPROT, "cardioprotective therapy")):
        if facts.get(key) is False:
            for claim, sentence in _treatment_claims(narrative, drug_re):
                result["claims_checked"] += 1
                result["violations"].append({
                    "claim": claim, "sentence": sentence, "term": f"treatment given: {label}",
                    "narrative_value": "given", "computed_value": "not given",
                    "message": f"Narrative states '{claim}', but the record shows no {label} in the current cycle.",
                })

    result["consistent"] = not result["violations"]
    return result


# ---------------------------------------------------------------------------
# Plan consistency
# ---------------------------------------------------------------------------
_CONTINUE_RE = re.compile(r"\bcontinue\s+(?:current\s+)?(?:chemotherapy|doxorubicin|treatment|the\s+anthracycline)", re.I)


def check_plan_consistency(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flags a final plan that both halts chemotherapy (an action tagged
    halts_chemotherapy=True by the Intervention Agent) and tells the
    clinician to continue it. Superseded and informational actions are
    ignored. This is a second line of defence: the Intervention Agent is
    designed never to produce such a plan, and this check proves it at
    runtime."""
    live = [a for a in actions if not a.get("superseded") and not a.get("informational_only")]
    halting = [a for a in live if a.get("halts_chemotherapy")]
    continuing = [a for a in live if _CONTINUE_RE.search(a.get("action", ""))]
    contradictory = bool(halting and continuing)
    return {
        "consistent": not contradictory,
        "halting_actions": [a["action"] for a in halting],
        "continue_actions": [a["action"] for a in continuing] if contradictory else [],
    }


# ---------------------------------------------------------------------------
# Raw identifiers (formatting, not safety)
# ---------------------------------------------------------------------------
# Known internal keys: replaced with their clinical wording. Single-word keys
# that are also ordinary words ("age", "hypertension") are excluded - seeing
# them in prose is not a leak.
_KNOWN_RAW = {k: v for k, v in RAW_IDENTIFIER_DISPLAY.items() if "_" in k}
_KNOWN_RAW_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _KNOWN_RAW), key=len, reverse=True)) + r")\b")
# Any other snake_case or SCREAMING_SNAKE token: flagged, not replaced.
_GENERIC_RAW_RE = re.compile(r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")


def check_raw_identifiers(narrative: Optional[str]) -> Dict[str, Any]:
    """Finds internal identifiers (e.g. current_lvef, ML_SHAP) in the
    narrative and returns a display version with known ones replaced.

    This is a FORMATTING issue (system-prompt rule 4), not a clinical one:
    replacing "current_lvef" with "current LVEF" cannot change the clinical
    meaning, so unlike a grade contradiction it is repaired rather than
    withheld. The original text is kept for audit. Unknown identifiers are
    only reported, never guessed."""
    if not narrative:
        return {"found": [], "replaced": [], "unknown": [], "display_text": narrative}
    replaced = sorted(set(_KNOWN_RAW_RE.findall(narrative)))
    display = _KNOWN_RAW_RE.sub(lambda m: _KNOWN_RAW[m.group(1)], narrative)
    unknown = sorted(set(_GENERIC_RAW_RE.findall(display)))
    return {
        "found": sorted(set(replaced) | set(unknown)),
        "replaced": replaced,
        "unknown": unknown,
        "display_text": display,
    }
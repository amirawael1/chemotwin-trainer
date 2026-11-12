"""
agentic/tools.py
────────────────
Clinical tools the agents can call. This is where the earlier deterministic
work (CTCAE v6.0 grading, ESC 2022 CTRCD classification, drug checks,
guideline corpus) now lives: as TOOLS an LLM agent chooses to use, not as a
fixed pipeline.

Design rule — tools are RECORD-BOUND. A tool never takes a lab value from the
model; it reads the value from the patient record itself. The model can only
choose WHICH tool to call and with which non-numeric argument (a CTCAE term,
a drug name, a search query). A model therefore cannot "ground" its answer on
a value it misread or invented: the tool reports what the record contains.

Every tool result carries:
  - `source`: the exact standard and version it applies (e.g. "NCI CTCAE v6.0")
  - `evidence_id`: a short id the agent must cite when it uses the result
  - `missing_inputs`: inputs the tool needed but the record does not have
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from data.guideline_corpus import GUIDELINE_CONFLICTS, GUIDELINE_CORPUS
from utils import ctcae_utils as cu
from utils.drug_interactions import (TRASTUZUMAB_WASHOUT_DAYS, check_interactions,
                                     check_trastuzumab_washout)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTCAE_TERMS_FILE = PROJECT_ROOT / "data" / "rules" / "ctcae_v6_terms.json"

ANTI_HER2 = ("trastuzumab", "pertuzumab", "lapatinib", "tucatinib", "neratinib", "t-dm1", "t-dxd",
             "herceptin", "kadcyla", "enhertu")

# Drug -> class, used for allergy cross-matching (an "ACE inhibitor" allergy
# blocks lisinopril) and for the interaction table, which is keyed by class.
DRUG_CLASSES: Dict[str, str] = {
    "lisinopril": "ace inhibitor", "enalapril": "ace inhibitor", "ramipril": "ace inhibitor",
    "perindopril": "ace inhibitor", "captopril": "ace inhibitor",
    "candesartan": "arb", "valsartan": "arb", "losartan": "arb", "telmisartan": "arb",
    "carvedilol": "beta-blocker", "bisoprolol": "beta-blocker", "metoprolol": "beta-blocker",
    "nebivolol": "beta-blocker",
    "doxorubicin": "anthracycline", "epirubicin": "anthracycline", "daunorubicin": "anthracycline",
    "idarubicin": "anthracycline",
    "filgrastim": "g-csf", "pegfilgrastim": "g-csf", "lipegfilgrastim": "g-csf",
    "piperacillin": "penicillin", "amoxicillin": "penicillin", "penicillin": "penicillin",
    "cefepime": "cephalosporin", "ceftriaxone": "cephalosporin", "ceftazidime": "cephalosporin",
    "meropenem": "carbapenem", "vancomycin": "glycopeptide", "fluconazole": "azole antifungal",
    "dexrazoxane": "cardioprotectant", "sacubitril": "arni", "spironolactone": "mra",
    "trastuzumab": "anti-her2", "pertuzumab": "anti-her2",
}

# CTCAE term keys the agent may ask for -> (official v6.0 term, profile key)
CTCAE_TERMS: Dict[str, tuple] = {
    "neutrophil_count_decreased": ("Neutrophil count decreased", "neutrophil_count_grade"),
    "febrile_neutropenia": ("Febrile neutropenia", "febrile_neutropenia_grade"),
    "left_ventricular_dysfunction": ("Left ventricular dysfunction", "lv_dysfunction_grade"),
    "heart_failure": ("Heart failure", "heart_failure_grade"),
    "creatinine_increased": ("Creatinine increased", "creatinine_increased_grade"),
    "blood_bilirubin_increased": ("Blood bilirubin increased", "bilirubin_increased_grade"),
    "alanine_aminotransferase_increased": ("Alanine aminotransferase increased", "alt_increased_grade"),
    "aspartate_aminotransferase_increased": ("Aspartate aminotransferase increased", "ast_increased_grade"),
}

# ESC 2022 Recommendation Table 24 (p. e388): anthracycline management by CTRCD category.
ESC_ANTHRACYCLINE_MANAGEMENT = {
    ("asymptomatic", "mild"): ("continue", "Continue the anthracycline with close cardiac monitoring; "
                               "consider ACE inhibitor/ARB and/or beta-blocker."),
    ("asymptomatic", "moderate"): ("interrupt", "Temporarily interrupt the anthracycline and start "
                                   "heart-failure therapy; multidisciplinary decision on restart."),
    ("asymptomatic", "severe"): ("interrupt", "Temporarily interrupt the anthracycline and start "
                                 "heart-failure therapy; multidisciplinary decision on restart."),
    ("symptomatic", "mild"): ("mdt", "Start heart-failure therapy; multidisciplinary decision on "
                              "interruption versus continuation."),
    ("symptomatic", "moderate"): ("interrupt", "Temporarily interrupt the anthracycline and start "
                                  "heart-failure therapy; multidisciplinary decision on restart."),
    ("symptomatic", "severe"): ("discontinue", "Discontinue the anthracycline; heart-failure therapy."),
    ("symptomatic", "very_severe"): ("discontinue", "Discontinue the anthracycline; heart-failure therapy."),
}
ESC_MGMT_CHUNK = "esc_anthracycline_ctrcd_management_01"

# ESC 2022 Recommendation Table 25 (p. e390): HER2-targeted therapy by CTRCD category.
# Differs from anthracyclines: moderate ASYMPTOMATIC CTRCD -> CONTINUE with cardioprotection.
ESC_ANTIHER2_MANAGEMENT = {
    ("asymptomatic", "mild"): ("continue", "Continue HER2-targeted therapy with closer cardiac monitoring; "
                               "ACE inhibitor/ARB and/or beta-blocker may be considered."),
    ("asymptomatic", "moderate"): ("continue_with_cardioprotection",
                                   "Continue HER2-targeted therapy; start ACE inhibitor/ARB and beta-blocker; "
                                   "more frequent cardiac monitoring; multidisciplinary review."),
    ("asymptomatic", "severe"): ("interrupt", "Temporarily interrupt HER2-targeted therapy; start heart-failure "
                                 "therapy; multidisciplinary decision on restart."),
    ("symptomatic", "mild"): ("mdt", "Start heart-failure therapy; multidisciplinary decision on continuing "
                              "HER2-targeted therapy."),
    ("symptomatic", "moderate"): ("interrupt", "Temporarily interrupt HER2-targeted therapy; heart-failure therapy; "
                                  "multidisciplinary decision on restart."),
    ("symptomatic", "severe"): ("interrupt", "Interrupt HER2-targeted therapy; heart-failure therapy; "
                                "multidisciplinary decision."),
    ("symptomatic", "very_severe"): ("interrupt", "Interrupt HER2-targeted therapy; heart-failure therapy; "
                                     "multidisciplinary decision."),
}
ESC_HER2_CHUNK = "cardiac_dysfunction_withhold_antiher2_01"


@lru_cache(maxsize=1)
def ctcae_terms() -> Dict[str, Any]:
    if CTCAE_TERMS_FILE.exists():
        return json.loads(CTCAE_TERMS_FILE.read_text(encoding="utf-8"))
    return {"terms": {}}


def _criterion_text(term: str, grade: Optional[int]) -> Optional[str]:
    if not grade:
        return None
    t = ctcae_terms().get("terms", {}).get(term, {})
    return (t.get(f"grade_{grade}") or "").strip() or None


# Class-level wording a plan may use instead of a drug name ("start an ACE
# inhibitor"). Checked like a drug, so a class proposal is caught against a
# documented allergy to any member of the class.
CLASS_TERMS: Dict[str, str] = {
    "ace inhibitor": "ace inhibitor", "ace-inhibitor": "ace inhibitor", "ace-i": "ace inhibitor",
    "acei": "ace inhibitor", "angiotensin-converting": "ace inhibitor", "angiotensin converting": "ace inhibitor",
    "angiotensin receptor blocker": "arb", "arb": "arb",
    "beta-blocker": "beta-blocker", "beta blocker": "beta-blocker",
    "g-csf": "g-csf", "anthracycline": "anthracycline", "penicillin": "penicillin",
    "cephalosporin": "cephalosporin", "carbapenem": "carbapenem",
}
_CLASS_RE = {t: re.compile(r"(?<![a-z])" + re.escape(t) + r"s?(?![a-z])") for t in CLASS_TERMS}


def drug_class(drug: str) -> Optional[str]:
    d = drug.lower()
    for name, cls in DRUG_CLASSES.items():
        if name in d:
            return cls
    for term, cls in CLASS_TERMS.items():
        if _CLASS_RE[term].search(d):
            return cls
    return None


def drug_mentions(text: str):
    """(term, position) for every drug name or drug-class phrase in text."""
    low = text.lower()
    out = []
    for name in DRUG_CLASSES:
        i = low.find(name)
        if i != -1:
            out.append((name, i))
    for term, rx in _CLASS_RE.items():
        m = rx.search(low)
        if m:
            out.append((term, m.start()))
    return out


def _norm_drug(text: str) -> str:
    return re.sub(r"\(.*?\)", "", str(text)).strip().lower()


# ===========================================================================
# Tool implementations
# ===========================================================================
class ToolBox:
    """The set of tools available for ONE patient. `enabled` lets the
    experiment remove tools (ablations) without changing the agent code."""

    def __init__(self, patient: Any, enabled: Optional[List[str]] = None, record_text: Optional[str] = None):
        self.patient = patient
        self._profile = None
        # Free text of the record. Only the D_text system passes it; then facts written only in words
        # (e.g. "febrile" without a measured temperature) are read by text_fever(). The answer key and
        # all other systems build the ToolBox without it, so their behaviour is unchanged.
        self.record_text = record_text
        self.registry: Dict[str, Dict[str, Any]] = {
            "ctcae_grade": {
                "fn": self.ctcae_grade,
                "args": {"term": "one of: " + ", ".join(CTCAE_TERMS) + ", or 'all'"},
                "doc": "Grades a toxicity with NCI CTCAE v6.0 from the patient record; returns the grade, the "
                       "official v6.0 criterion text, the values used and any missing inputs.",
            },
            "esc_ctrcd": {
                "fn": self.esc_ctrcd,
                "args": {},
                "doc": "Classifies cancer-therapy-related cardiac dysfunction (ESC 2022 Table 3) from the "
                       "record and returns the ESC 2022 anthracycline management (Rec. Table 24).",
            },
            "check_medication": {
                "fn": self.check_medication,
                "args": {"drug": "drug name you are considering proposing or continuing, e.g. 'lisinopril'"},
                "doc": "Checks a drug against the documented allergies (including drug class), the active "
                       "medications (interaction table) and the trastuzumab washout window.",
            },
            "dose_and_organ_function": {
                "fn": self.dose_and_organ_function,
                "args": {},
                "doc": "Cumulative doxorubicin exposure, FDA-label hepatic dose bands (bilirubin) and "
                       "renal function change from baseline.",
            },
            "data_completeness": {
                "fn": self.data_completeness,
                "args": {},
                "doc": "Lists safety-critical inputs that are missing, stale or not evaluable, and what "
                       "data should be requested before deciding.",
            },
            "search_guidelines": {
                "fn": self.search_guidelines,
                "args": {"query": "free-text clinical question", "k": "number of passages (default 3)"},
                "doc": "Searches the versioned guideline corpus (CTCAE v6.0, ESC 2022, ASCO 2017, ESMO 2016). "
                       "Returns passages with chunk_id, source, year and citation, plus any registered "
                       "disagreement between guidelines that applies to this patient.",
            },
        }
        if enabled is not None:
            self.registry = {k: v for k, v in self.registry.items() if k in enabled}

    # ------------------------------------------------------------------
    def describe(self) -> str:
        lines = []
        for name, spec in self.registry.items():
            args = ", ".join(f"{k}: {v}" for k, v in spec["args"].items()) or "no arguments"
            lines.append(f"- {name}({args})\n    {spec['doc']}")
        return "\n".join(lines)

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in self.registry:
            return {"error": f"unknown or unavailable tool '{name}'", "available": list(self.registry)}
        try:
            return self.registry[name]["fn"](**(args or {}))
        except TypeError as err:
            return {"error": f"bad arguments for {name}: {err}", "expected_args": self.registry[name]["args"]}

    @property
    def profile(self) -> Dict[str, Any]:
        if self._profile is None:
            self._profile = cu.evaluate_patient_ctcae_v6(self.patient)
        return self._profile

    # ------------------------------------------------------------------
    def ctcae_grade(self, term: str = "all") -> Dict[str, Any]:
        key = str(term).strip().lower().replace(" ", "_").replace("-", "_")
        if key == "all":
            return {"source": "NCI CTCAE v6.0", "evidence_id": "ctcae_v6:all",
                    "results": [self.ctcae_grade(k) for k in CTCAE_TERMS]}
        aliases = {"anc": "neutrophil_count_decreased", "neutropenia": "neutrophil_count_decreased",
                   "fn": "febrile_neutropenia", "lv_dysfunction": "left_ventricular_dysfunction",
                   "lvsd": "left_ventricular_dysfunction", "left_ventricular_systolic_dysfunction":
                   "left_ventricular_dysfunction", "ejection_fraction_decreased": "left_ventricular_dysfunction",
                   "creatinine": "creatinine_increased", "bilirubin": "blood_bilirubin_increased",
                   "alt": "alanine_aminotransferase_increased", "ast": "aspartate_aminotransferase_increased"}
        note = None
        if key in ("left_ventricular_systolic_dysfunction", "ejection_fraction_decreased"):
            note = (f"'{term}' is a CTCAE v5.0 term that was REMOVED in v6.0; graded here as "
                    "'Left ventricular dysfunction' (v6.0).")
        key = aliases.get(key, key)
        if key not in CTCAE_TERMS:
            return {"error": f"unknown CTCAE term '{term}'", "valid_terms": list(CTCAE_TERMS)}
        official, pkey = CTCAE_TERMS[key]
        grade = self.profile.get(pkey)
        if key == "heart_failure" and getattr(self.patient, "heart_failure_status", None) is None:
            if getattr(self.patient, "assessment_scope", "full") == "cardiac":
                grade = None  # free-text notes may describe symptoms the structured record does not hold
                note = ("Symptom status is not in the structured record: read the clinic and nursing notes "
                        "and grade heart failure from what they describe.")
            else:
                grade = 0  # the record documents no heart-failure symptoms
                note = "No heart-failure symptoms documented in the record (grade 0)."
        if key == "left_ventricular_dysfunction" and self._lvef_implausible():
            grade = None
            note = "Current LVEF value is physiologically implausible (likely transcription error): not graded."
        elif key == "left_ventricular_dysfunction" and self._lvef_fall_unknown():
            grade = None
            note = ("LVEF 40-49% without a baseline LVEF, GLS or biomarkers: a NEW fall cannot be "
                    "established, so the grade is not evaluable. Obtain the baseline echo.")
        out: Dict[str, Any] = {
            "term": official, "term_key": key, "source": "NCI CTCAE v6.0",
            "evidence_id": f"ctcae_v6:{key}",
            "grade": grade if grade is not None else "not_evaluable",
            "criterion_text": _criterion_text(official, grade),
            "inputs": self._inputs_for(key), "missing_inputs": [],
        }
        if grade is None:
            out["missing_inputs"] = [k for k, v in out["inputs"].items() if v is None] or ["required input"]
        if key == "febrile_neutropenia" and self._no_temperature_low_anc():
            # D_text only (record text given): with ANC < 1000/uL and no measured temperature, febrile
            # neutropenia is NOT excluded. Grade 3 if a fever is documented in words, else not evaluable.
            words, afebrile = self.text_fever(), self.text_afebrile()
            grade = 3 if words else (0 if afebrile else None)
            out["grade"] = grade if grade is not None else "not_evaluable"
            out["criterion_text"] = _criterion_text(official, grade)
            out["missing_inputs"] = [] if grade is not None else ["max_temperature_c"]
            if afebrile and not words:
                out["afebrile_documented_in_words"] = afebrile
                note = "No fever is documented in words (no measured value): febrile neutropenia criteria not met."
            if words:
                out["fever_documented_in_words"] = words
                note = ("Fever is documented in words without a measured value, and ANC is below 1000/uL: treat "
                        "as febrile neutropenia (ESMO 2016) - blood cultures, then empiric antibiotics within "
                        "1 hour - and record the temperature.")
        if key == "febrile_neutropenia":
            fa = self.profile.get("fever_assessment") or {}
            out["fever_criterion"] = fa.get("criterion_met")
            if not fa.get("sustained_criterion_evaluable"):
                out["notes"] = ["Sustained-fever rule (>= 38.0 C for > 1 h) not evaluable: no timed readings."]
            gap = self.profile.get("anc_temperature_day_gap")
            if gap is not None and abs(gap) > 1:
                out.setdefault("notes", []).append(f"ANC and temperature are {abs(gap)} days apart.")
        if note:
            out.setdefault("notes", []).append(note)
        return out

    # ------------------------------------------------------------------
    _FEVER = re.compile(r"\b(febrile|fever(?:ish)?|pyrexi\w*|pyrexial)\b", re.I)
    _FEVER_NEG = re.compile(r"\b(no|not|without|denies|denied|absent|free of|negative for|nor)\b[^.;\n]{0,25}$", re.I)
    _FEVER_PAST = re.compile(r"\b(history of|previous|prior|earlier|last cycle|in cycle \d)\b[^.;\n]{0,30}$", re.I)

    def text_fever(self) -> Optional[str]:
        """A current fever stated only in words. Returns the matching phrase, or None.
        Ignores negations ('no fever', 'afebrile'), past episodes ('history of febrile
        neutropenia') and diagnosis labels ('febrile neutropenia')."""
        if not self.record_text:
            return None
        for m in self._FEVER.finditer(self.record_text):
            before = self.record_text[max(0, m.start() - 40):m.start()]
            after = self.record_text[m.end():m.end() + 14].lower()
            if self._FEVER_NEG.search(before) or self._FEVER_PAST.search(before):
                continue
            if after.lstrip().startswith("neutropeni"):
                continue
            return self._line_at(m.start(), m.end())
        return None

    def text_afebrile(self) -> Optional[str]:
        """'No fever' / 'afebrile' stated in words. Returns the matching line, or None."""
        if not self.record_text:
            return None
        for m in re.finditer(r"\bafebrile\b|\bapyrexial\b", self.record_text, re.I):
            return self._line_at(m.start(), m.end())
        for m in self._FEVER.finditer(self.record_text):
            before = self.record_text[max(0, m.start() - 40):m.start()]
            if self._FEVER_NEG.search(before) and not self._FEVER_PAST.search(before):
                return self._line_at(m.start(), m.end())
        return None

    def _line_at(self, a: int, b: int) -> str:
        line_start = self.record_text.rfind("\n", 0, a) + 1
        line_end = self.record_text.find("\n", b)
        return self.record_text[line_start:line_end if line_end >= 0 else None].strip()[:160]

    def _no_temperature_low_anc(self) -> bool:
        """D_text only: ANC < 1000/uL and no measured temperature."""
        if self.record_text is None:
            return False
        lab = self.patient.get_latest_labs()
        anc = getattr(lab, "anc_cells_per_ul", None) if lab else None
        maxt = (self.profile.get("fever_assessment") or {}).get("max_temperature_c")
        return maxt is None and anc is not None and anc < 1000

    def _lvef_implausible(self) -> bool:
        card = self.patient.get_latest_cardiac()
        lvef = getattr(card, "ejection_fraction_pct", None) if card else None
        return lvef is not None and not (15.0 <= lvef <= 85.0)

    def _lvef_fall_unknown(self) -> bool:
        card, base = self.patient.get_latest_cardiac(), self.patient.get_baseline_cardiac()
        lvef = getattr(card, "ejection_fraction_pct", None) if card else None
        esc = self.profile["esc_ctrcd"]
        no_base = base is None or base is card
        return bool(lvef is not None and 40.0 <= lvef < 50.0 and no_base
                    and esc.get("gls_decline_pct") is None and not esc.get("biomarker_criterion_met"))

    def _inputs_for(self, key: str) -> Dict[str, Any]:
        p = self.patient
        lab, base_lab = p.get_latest_labs(), p.get_baseline_labs()
        vit, card, base_card = p.get_latest_vitals(), p.get_latest_cardiac(), p.get_baseline_cardiac()
        g = lambda o, a: getattr(o, a, None) if o is not None else None  # noqa: E731
        if key == "neutrophil_count_decreased":
            return {"anc_cells_per_ul": g(lab, "anc_cells_per_ul")}
        if key == "febrile_neutropenia":
            return {"anc_cells_per_ul": g(lab, "anc_cells_per_ul"),
                    "max_temperature_c": (self.profile.get("fever_assessment") or {}).get("max_temperature_c")}
        if key == "left_ventricular_dysfunction":
            return {"lvef_pct": g(card, "ejection_fraction_pct"),
                    "baseline_lvef_pct": g(base_card, "ejection_fraction_pct") if base_card is not card else None,
                    "gls_relative_decline_pct": (self.profile["esc_ctrcd"] or {}).get("gls_decline_pct")}
        if key == "heart_failure":
            return {"heart_failure_status": getattr(p, "heart_failure_status", None)}
        if key == "creatinine_increased":
            return {"creatinine_mg_dl": g(lab, "creatinine_mg_dl"),
                    "baseline_creatinine_mg_dl": g(base_lab, "creatinine_mg_dl") if base_lab is not lab else None}
        if key == "blood_bilirubin_increased":
            return {"total_bilirubin_mg_dl": g(lab, "total_bilirubin_mg_dl")}
        if key == "alanine_aminotransferase_increased":
            return {"alt_u_l": g(lab, "alt_u_l")}
        if key == "aspartate_aminotransferase_increased":
            return {"ast_u_l": g(lab, "ast_u_l")}
        return {}

    # ------------------------------------------------------------------
    def esc_ctrcd(self) -> Dict[str, Any]:
        esc = self.profile["esc_ctrcd"]
        asym, sym = esc.get("asymptomatic"), esc.get("symptomatic")
        if self._lvef_fall_unknown() or self._lvef_implausible():
            asym = None
        if sym and sym != "none":
            key = ("symptomatic", sym)
        elif asym and asym != "none":
            key = ("asymptomatic", asym)
        else:
            key = None
        active = [str(m).lower() for m in getattr(self.patient, "active_medications", []) or []]
        anthra_active = any(drug_class(m) == "anthracycline" for m in active)
        her2_active = any(any(a in m for a in ANTI_HER2) for m in active)
        if anthra_active or not her2_active:
            table, chunk, mgmt = ("Rec. Table 24 (anthracycline)", ESC_MGMT_CHUNK,
                                  ESC_ANTHRACYCLINE_MANAGEMENT.get(key) if key else None)
        else:
            table, chunk, mgmt = ("Rec. Table 25 (HER2-targeted therapy)", ESC_HER2_CHUNK,
                                  ESC_ANTIHER2_MANAGEMENT.get(key) if key else None)
        her2_mgmt = ESC_ANTIHER2_MANAGEMENT.get(key) if (key and her2_active and anthra_active) else None
        card = self.patient.get_latest_cardiac()
        lvef = getattr(card, "ejection_fraction_pct", None) if card else None
        return {
            "source": f"ESC 2022 Cardio-Oncology Guidelines (Lyon et al.), Table 3 and {table}",
            "active_cardiotoxic_therapy": ", ".join(x for x, on in (("anthracycline", anthra_active),
                                                                     ("HER2-targeted", her2_active)) if on) or "none",
            "management_table": table,
            "evidence_id": "esc2022:ctrcd",
            "asymptomatic_ctrcd": asym if asym is not None else "not_evaluable",
            "symptomatic_ctrcd": sym if sym is not None else "not_documented",
            "lvef_pct": lvef,
            "lvef_point_drop": esc.get("lvef_point_drop"),
            "gls_relative_decline_pct": esc.get("gls_decline_pct"),
            "gls_criterion_met": esc.get("gls_criterion_met"),
            "biomarker_criterion_met": esc.get("biomarker_criterion_met"),
            "lvef_below_40_pre_existing": esc.get("pre_existing_lvef_below_40"),
            "absolute_lvef_below_40": bool(lvef is not None and lvef < 40.0 and not self._lvef_implausible()),
            "anthracycline_management": ({"decision": mgmt[0], "text": mgmt[1], "chunk_id": chunk}
                                         if mgmt else {"decision": "continue",
                                                       "text": "No CTRCD: routine surveillance.",
                                                       "chunk_id": None}),
            **({"her2_therapy_management": {"decision": her2_mgmt[0], "text": her2_mgmt[1],
                                            "chunk_id": ESC_HER2_CHUNK}} if her2_mgmt else {}),
            "notes": esc.get("notes", []),
            "missing_inputs": ["current LVEF"] if lvef is None else [],
        }

    # ------------------------------------------------------------------
    def check_medication(self, drug: str = "") -> Dict[str, Any]:
        p = self.patient
        name = _norm_drug(drug)
        cls = drug_class(name)
        allergy_hits = []
        for allergy in getattr(p, "known_drug_allergies", []) or []:
            a = _norm_drug(allergy)
            if not a:
                continue
            same_drug = a in name or name in a
            a_cls = drug_class(a) or (a if a in set(DRUG_CLASSES.values()) or a.rstrip("s") in
                                      set(DRUG_CLASSES.values()) else None)
            same_class = cls is not None and a_cls is not None and a_cls.rstrip("s") == cls
            if same_drug or same_class:
                allergy_hits.append({"documented_allergy": allergy,
                                     "match": "same drug" if same_drug else f"same class ({cls})"})
        active = list(getattr(p, "active_medications", []) or [])
        probe = name + (f" ({cls})" if cls else "")
        already_active = any(name in m.lower() for m in active)
        others = [m for m in active if name not in m.lower()]
        ddi = [i for i in check_interactions(others + [probe]) if probe in (i["medication_a"], i["medication_b"])]
        washout = []
        if cls == "anthracycline":
            washout = check_trastuzumab_washout([name], getattr(p, "discontinued_medications", []) or [],
                                                p.get_current_day())
        verdict = "contraindicated" if allergy_hits else ("caution" if (ddi or washout) else "no_issue_found")
        return {
            "source": "Patient allergy list; local interaction table (utils/drug_interactions.py); "
                      f"FDA doxorubicin label s7.2 ({TRASTUZUMAB_WASHOUT_DAYS}-day trastuzumab washout)",
            "evidence_id": f"medcheck:{name}",
            "drug": drug, "drug_class": cls, "already_active": already_active,
            "verdict": verdict, "allergy_hits": allergy_hits,
            "interactions": [{"with": i["medication_a"] if i["medication_b"] == probe else i["medication_b"],
                              "severity": i["severity"], "description": i["description"]} for i in ddi],
            "washout": [w["description"] for w in washout],
        }

    # ------------------------------------------------------------------
    def dose_and_organ_function(self) -> Dict[str, Any]:
        p = self.patient
        cyc = p.get_current_cycle()
        cum = getattr(cyc, "cumulative_doxorubicin_dose", None) if cyc else None
        lab, base = p.get_latest_labs(), p.get_baseline_labs()
        bili = getattr(lab, "total_bilirubin_mg_dl", None) if lab else None
        cr = getattr(lab, "creatinine_mg_dl", None) if lab else None
        base_cr = getattr(base, "creatinine_mg_dl", None) if (base is not None and base is not lab) else None
        if bili is None:
            hep = "not_evaluable"
        elif bili > 5.0:
            hep = "contraindicated (bilirubin > 5 mg/dL): do not give doxorubicin"
        elif bili > 3.0:
            hep = "reduce dose by 75% (bilirubin 3.1-5 mg/dL)"
        elif bili >= 1.2:
            hep = "reduce dose by 50% (bilirubin 1.2-3 mg/dL)"
        else:
            hep = "no hepatic dose adjustment"
        fold = round(cr / base_cr, 2) if (cr and base_cr) else None
        return {
            "source": "FDA doxorubicin prescribing information (hepatic dosing); ASCO 2017 Rec. 1.1 "
                      "(cumulative dose >= 250 mg/m2 = higher risk); KDIGO 2012 (creatinine >= 1.5 x baseline)",
            "evidence_id": "dose_organ",
            "cumulative_doxorubicin_mg_m2": cum,
            "cumulative_dose_higher_risk": bool(cum is not None and cum >= 250.0),
            "cumulative_dose_note": "Higher-risk range calls for enhanced surveillance; not by itself a reason to stop.",
            "age": getattr(p, "age", None),
            "total_bilirubin_mg_dl": bili, "hepatic_dose_band": hep,
            "creatinine_mg_dl": cr, "baseline_creatinine_mg_dl": base_cr, "creatinine_fold_baseline": fold,
            "kdigo_aki_criterion_met": bool(fold is not None and fold >= 1.5),
            "missing_inputs": [k for k, v in (("total bilirubin", bili), ("creatinine", cr)) if v is None],
        }

    # ------------------------------------------------------------------
    def data_completeness(self) -> Dict[str, Any]:
        p = self.patient
        lab, vit = p.get_latest_labs(), p.get_latest_vitals()
        card, base_card = p.get_latest_cardiac(), p.get_baseline_cardiac()
        critical, other = [], []
        cardiac_scope = getattr(p, "assessment_scope", "full") == "cardiac"
        if not cardiac_scope:
            if vit is None or getattr(vit, "temperature_c", None) is None:
                if self.text_fever() or self.text_afebrile():   # stated in words: act on it, record the value
                    other.append("measured temperature value (fever status is documented in words only)")
                else:
                    critical.append("current temperature")
            if lab is None or getattr(lab, "anc_cells_per_ul", None) is None:
                critical.append("current ANC (absolute neutrophil count)")
        # "non_cardiac" scope (layer A real cases about neutropenia, liver or kidney toxicity, where the
        # report has no echo): a missing echo is noted but does not block the decision.
        non_cardiac = getattr(p, "assessment_scope", "full") == "non_cardiac"
        if card is None or getattr(card, "ejection_fraction_pct", None) is None:
            (other if non_cardiac else critical).append("current LVEF (echocardiogram)")
        lv_now = getattr(card, "ejection_fraction_pct", None) if card else None
        if lv_now is not None and not (15.0 <= lv_now <= 85.0):
            critical.append(f"plausible current LVEF (recorded value {lv_now}% is physiologically implausible; "
                            "verify the report or repeat the echo)")
        if cardiac_scope and (base_card is None or base_card is card) and lv_now is not None and lv_now >= 40:
            critical.append("baseline LVEF (CTRCD is defined by change from baseline)")
        if base_card is None or base_card is card:
            if self._lvef_fall_unknown():
                critical.append("baseline LVEF (current LVEF is 40-49%; a new fall cannot be judged without it)")
            else:
                other.append("baseline LVEF (needed to judge a new fall)")
        n_cycles = len(getattr(p, "cycles", []) or [])
        if card is not None and base_card is card and n_cycles > 1 and getattr(card, "assessment_label", "") == "Baseline":
            other.append("on-treatment echocardiogram (the only echo is the baseline)")
        gap = self.profile.get("anc_temperature_day_gap")
        febrile_now = ((self.profile.get("fever_assessment") or {}).get("max_temperature_c") or 0) >= 38.0
        if gap is not None and abs(gap) > 1 and febrile_now:   # matters for febrile-neutropenia assessment
            critical.append(f"contemporaneous ANC and temperature (currently {abs(gap)} days apart)")
        fa = self.profile.get("fever_assessment") or {}
        maxt = fa.get("max_temperature_c")
        if maxt is not None and 38.0 <= maxt <= 38.3 and not fa.get("sustained_criterion_evaluable"):
            other.append("timed temperature readings (to apply the sustained >= 38.0 C for > 1 h rule)")
        if card is not None and getattr(card, "global_longitudinal_strain_pct", None) is None:
            other.append("GLS (global longitudinal strain)")
        if lab is not None and all(getattr(lab, a, None) is None for a in
                                   ("troponin_ng_l", "nt_probnp_pg_ml", "bnp_pg_ml")):
            other.append("cardiac biomarkers (troponin / NT-proBNP)")
        for f in self.profile.get("data_quality_flags", []):
            if f.startswith("default_") and not cardiac_scope:
                other.append("institutional lab reference ranges (defaults in use)")
                break
        notes = list(getattr(p, "record_quality_notes", []) or [])
        return {
            "source": "Agentic ChemoTwin data-completeness check",
            "evidence_id": "data_completeness",
            "critical_missing": list(dict.fromkeys(critical)),
            "other_missing": sorted(set(other)),
            "can_decide_safely": not critical,
            **({"record_quality_notes": notes} if notes else {}),
            "advice": ("Do not make a treatment decision until the critical items are available; request them."
                       if critical else "Safety-critical inputs present."),
        }

    # ------------------------------------------------------------------
    def search_guidelines(self, query: str = "", k: Any = 3) -> Dict[str, Any]:
        try:
            k = max(1, min(int(k), 6))
        except (TypeError, ValueError):
            k = 3
        hits = bm25_search(query, k)
        tags = {t for h in hits for t in h["gate_tags"]}
        anti_her2 = any(any(a in str(m).lower() for a in ANTI_HER2)
                        for m in getattr(self.patient, "active_medications", []) or [])
        conflicts = []
        for c in GUIDELINE_CONFLICTS:
            if not tags.intersection(c.get("gate_tags", [])):
                continue
            if c.get("requires_flag") == "anti_her2_active" and not anti_her2:
                continue
            conflicts.append({"conflict_id": c["conflict_id"], "kind": c["kind"], "kind_note": c["kind_note"],
                              "positions": [{"source": x["source"], "position": x["position"]}
                                            for x in c["positions"]],
                              "system_choice": c["system_choice"]})
        return {"source": "Agentic ChemoTwin guideline corpus", "evidence_id": "guidelines",
                "query": query,
                "passages": [{"chunk_id": h["chunk_id"], "source": h["source"], "year": h["year"],
                              "citation": h["citation"], "text": h["text"],
                              **({"applies_to": h["applies_to"]} if h.get("applies_to") else {})}
                             for h in hits],
                "guideline_disagreements": conflicts}


# ===========================================================================
# Lexical retrieval (BM25) over the verified corpus — no model download needed
# ===========================================================================
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in for with is are be by on at as from that this it its than which "
            "when should not no any".split())


def _tok(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _year(c: Dict[str, Any]) -> Optional[int]:
    m = re.findall(r"(?:19|20)\d\d", c.get("citation", ""))
    if "CTCAE v6.0" in c.get("citation", ""):
        return 2025
    return int(m[0]) if m else None


@lru_cache(maxsize=1)
def _index():
    docs = []
    for c in GUIDELINE_CORPUS:
        toks = _tok(" ".join([c.get("topic", "").replace("_", " "), c.get("source", ""), c.get("text", "")]))
        docs.append((c, Counter(toks), len(toks)))
    n = len(docs)
    df = Counter(t for _, tf, _ in docs for t in tf)
    avgdl = sum(d[2] for d in docs) / max(n, 1)
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    return docs, idf, avgdl


def bm25_search(query: str, k: int = 3, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
    docs, idf, avgdl = _index()
    q = _tok(query)
    scored = []
    for c, tf, dl in docs:
        s = sum(idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avgdl)) for t in q if t in tf)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return [{**c, "year": _year(c), "score": round(s, 3)} for s, c in scored[:k]]


def corpus_ids() -> Dict[str, Dict[str, Any]]:
    return {c["chunk_id"]: c for c in GUIDELINE_CORPUS}


ToolFn = Callable[..., Dict[str, Any]]
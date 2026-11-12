"""
utils/display_names.py
──────────────────────
Single source of truth for turning internal identifiers (model feature
keys, attribution-source codes) into clinical wording.

Used in two places:
  1. agents/monitoring_agent.py - the narrative prompt is built with these
     display names, so the LLM never sees "current_lvef" or "ML_SHAP" and
     has nothing to copy (prevention).
  2. utils/narrative_consistency.py - after generation, any raw identifier
     the LLM still emits is detected and replaced in the displayed text
     (detection + safe repair).

Keep this in sync with models/feature_schema.py: every feature in
CARDIOTOXICITY_FEATURES and NEUTROPENIA_FEATURES needs an entry here
(tests/test_narrative_and_plan_consistency.py checks this).
"""

FEATURE_DISPLAY_NAMES = {
    "age": "age",
    "cumulative_dose": "cumulative doxorubicin dose",
    "baseline_lvef": "baseline LVEF",
    "current_lvef": "current LVEF",
    "gls_relative_change": "relative GLS change",
    "hypertension": "hypertension",
    "prior_cardiac_disease": "prior cardiac disease",
    "subtype_her2": "HER2-positive subtype",
    "subtype_tnbc": "triple-negative subtype",
    "cardioprotectant_given": "cardioprotectant use",
    "current_anc": "current ANC",
    "gcsf_given": "G-CSF administration",
}

ATTRIBUTION_SOURCE_DISPLAY = {
    "ML_SHAP": "trained ML ensemble (SHAP-explained)",
    "DETERMINISTIC_FALLBACK": "rule-based fallback engine",
    "UNKNOWN": "unspecified model source",
}

# Everything that must never appear verbatim in clinician-facing text
RAW_IDENTIFIER_DISPLAY = {**FEATURE_DISPLAY_NAMES, **ATTRIBUTION_SOURCE_DISPLAY}


def display_feature(name: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " "))


def display_source(code: str) -> str:
    return ATTRIBUTION_SOURCE_DISPLAY.get(code, code.replace("_", " ").lower())


# Binary features: an ABSENT feature (value 0) can still receive a SHAP value
# (the model compares against its training background), but listing it as a
# "driver" reads as if the patient had it. run_pipeline.py omits absent
# binary features from the top-driver list and shows every driver's value.
BINARY_FEATURES = {"hypertension", "prior_cardiac_disease", "subtype_her2", "subtype_tnbc",
                   "cardioprotectant_given", "gcsf_given"}
_UNITS = {"age": " y", "cumulative_dose": " mg/m²", "baseline_lvef": "%", "current_lvef": "%",
          "gls_relative_change": "% decline", "current_anc": "/µL"}
_DIRECTION_WORDS = {"HIGH": "raises risk", "INCREASING": "raises risk", "CRITICAL": "raises risk",
                    "PROTECTIVE": "lowers risk", "DECREASING": "lowers risk", "NEUTRAL": "neutral"}


def format_feature_value(name, value) -> str:
    if value is None:
        return "missing"
    if name in BINARY_FEATURES:
        return "yes" if float(value) >= 0.5 else "no"
    if isinstance(value, float) and not float(value).is_integer():
        return f"{value:.1f}{_UNITS.get(name, '')}"
    return f"{value:.0f}{_UNITS.get(name, '')}" if isinstance(value, (int, float)) else f"{value}{_UNITS.get(name, '')}"


def direction_words(direction: str) -> str:
    return _DIRECTION_WORDS.get(str(direction).upper(), str(direction).lower())


def is_absent_binary(name, value) -> bool:
    return name in BINARY_FEATURES and (value is None or float(value) < 0.5)
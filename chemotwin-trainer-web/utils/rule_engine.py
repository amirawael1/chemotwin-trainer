"""
utils/rule_engine.py
────────────────────
Deterministic Rule Engine for ChemoTwin.

Change log: the age threshold used in the "age + anthracycline exposure"
risk trigger was 65 years in three separate places (two fallback dicts
used when the JSON config fails to load, plus the .get() default). This
did not match the age threshold this project actually cites for that
exact trigger - ASCO 2017 Recommendation 1.1 (Armenian et al., J Clin
Oncol 2017;35(8):893-911), which specifies 60 years, not 65. Corrected
to 60, and the three previously-duplicated fallback dicts consolidated
into one DEFAULT_HIGH_RISK_THRESHOLDS constant so this kind of value
can no longer silently drift out of sync across copies.

IMPORTANT: this Python-side default is a fallback only, used if
data/rules/asco_cardiac_rules.json is missing, empty, or invalid JSON.
If that file exists and loads successfully, ITS "age_years" value is
what actually governs behavior at runtime - check that file directly
and update it too if it explicitly sets age_years to 65.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# Consolidated from three previously-duplicated inline dicts (see change
# log above). ASCO 2017 Recommendation 1.1 (Armenian et al.) is the
# source for all three of these figures.
DEFAULT_HIGH_RISK_THRESHOLDS: Dict[str, float] = {
    "cumulative_doxorubicin_mg_m2": 250.0,
    "baseline_lvef_pct": 50.0,
    "age_years": 60,
}


class CardiacSafetyEngine:
    def __init__(self, rules_path: str = "data/rules/asco_cardiac_rules.json"):
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_path.exists():
            return {"high_risk_thresholds": dict(DEFAULT_HIGH_RISK_THRESHOLDS)}

        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {"high_risk_thresholds": dict(DEFAULT_HIGH_RISK_THRESHOLDS)}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"[Warning] Failed to parse JSON from {self.rules_path}. Using internal defaults.")
            return {"high_risk_thresholds": dict(DEFAULT_HIGH_RISK_THRESHOLDS)}

    def evaluate_patient_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        thresholds = self.rules.get("high_risk_thresholds", {})
        triggers: List[str] = []

        dose = patient_data.get("cumulative_doxorubicin_mg_m2", 0.0)
        base_lvef = patient_data.get("baseline_lvef", 100.0)
        age = patient_data.get("age", 0)

        if dose >= thresholds.get("cumulative_doxorubicin_mg_m2", DEFAULT_HIGH_RISK_THRESHOLDS["cumulative_doxorubicin_mg_m2"]):
           triggers.append(f"Cumulative doxorubicin exposure ({dose} mg/m²) has reached the higher-risk range "
                            f"(>= 250 mg/m²): enhanced cardiac surveillance is indicated; this is not by itself "
                            f"a reason to stop treatment.")

        if base_lvef < thresholds.get("baseline_lvef_pct", DEFAULT_HIGH_RISK_THRESHOLDS["baseline_lvef_pct"]):
            triggers.append(f"Baseline LVEF ({base_lvef}%) is below normal threshold (50%).")

        if age >= thresholds.get("age_years", DEFAULT_HIGH_RISK_THRESHOLDS["age_years"]) and dose > 0:
            triggers.append(f"Age ({age}) combined with Anthracycline exposure increases risk.")

        return {
            "is_high_risk": len(triggers) > 0,
            "triggers": triggers
        }
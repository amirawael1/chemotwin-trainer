"""
utils/attribution_checks.py
───────────────────────────
Flags ML feature attributions whose direction contradicts basic
pharmacology/physiology, e.g. a normal ANC "raising" neutropenia risk, or a
cumulative doxorubicin dose >= 250 mg/m2 "lowering" cardiotoxicity risk.

Why: the ML ensemble is trained on synthetic labels and is not yet
calibrated on real data. Such attributions are model artefacts, not
clinical findings. They are shown with a warning in the console and kept
OUT of the LLM narrative prompt, so the LLM cannot rationalise them into
clinical statements. The deterministic Safety Agent is unaffected: it never
uses ML scores.

The rules are deliberately few and conservative (only directions that are
unambiguous clinically); anything not covered returns None.
"""

from typing import Optional

UP = {"HIGH", "INCREASING", "CRITICAL"}
DOWN = {"PROTECTIVE", "DECREASING"}


def counter_intuitive(toxicity: str, feature: str, value, direction: str) -> Optional[str]:
    """Returns a short reason if the attribution contradicts clinical
    expectation, else None. toxicity: 'cardiotoxicity' or 'febrile_neutropenia'."""
    if value is None:
        return None
    d = str(direction).upper()
    up, down = d in UP, d in DOWN
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if toxicity == "febrile_neutropenia" and feature == "current_anc":
        if up and v >= 1500:
            return "a normal ANC is shown as raising neutropenia risk"
        if down and v < 1000:
            return "a low ANC is shown as lowering neutropenia risk"
    if toxicity == "cardiotoxicity":
        if feature == "cumulative_dose" and down and v >= 250:
            return "a high cumulative doxorubicin dose is shown as lowering cardiotoxicity risk"
        if feature in ("current_lvef", "baseline_lvef") and up and v >= 55:
            return "a normal LVEF is shown as raising cardiotoxicity risk"
        if feature == "current_lvef" and down and v < 50:
            return "a reduced LVEF is shown as lowering cardiotoxicity risk"
        if feature == "gls_relative_change" and down and v > 15:
            return "a GLS decline above 15% is shown as lowering cardiotoxicity risk"
    return None
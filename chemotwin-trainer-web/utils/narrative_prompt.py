"""
utils/narrative_prompt.py
─────────────────────────
Single source of truth for the narrative prompt.

The Monitoring Agent (production) and run_c1_experiment.py (evaluation)
both build their prompts here, so the experiment's "grounded" arm is
byte-for-byte the prompt the pipeline sends. Previously the prompt was
assembled inline in agents/monitoring_agent.py, which would have forced
the experiment to copy it and risk the two drifting apart.
tests/test_c1_experiment.py checks that the Monitoring Agent really sends
build_narrative_prompt(...) output.
"""

from typing import Any, Dict

from utils.attribution_checks import counter_intuitive
from utils.ctcae_utils import evaluate_patient_ctcae_v6
from utils.display_names import display_feature, display_source, format_feature_value, is_absent_binary

FINAL_INSTRUCTION_GROUNDED = (
    "Synthesize a 2-sentence expert oncological monitoring note summarizing patient trajectory. "
    "If you mention a toxicity grade or CTRCD severity, use only the authoritative values above; "
    "a predicted risk percentage is not a grade."
)
# The instruction used before the Oct 2026 fix (no grades supplied).
FINAL_INSTRUCTION_UNGROUNDED = (
    "Synthesize a 2-sentence expert oncological monitoring note summarizing patient trajectory."
)


def build_authoritative_grading_block(patient_state: Any) -> str:
    """Deterministic CTCAE v6.0 / ESC 2022 grades for the narrative prompt.

    Added Oct 2026: without these, the LLM assigned grades from its own
    training memory (CTCAE v5.0 bands) and contradicted the Safety Agent,
    e.g. calling ANC 350 "grade 4 neutropenia" (v6.0: Grade 3). The same
    function the Safety Agent uses is called here, so the narrative and
    the safety layer see identical grades. utils/narrative_consistency.py
    then verifies the narrative against them after generation."""
    p = evaluate_patient_ctcae_v6(patient_state)
    esc = p.get("esc_ctrcd") or {}

    def g(value):
        # CTCAE grades run 1-5; absence of the adverse event is not "Grade 0".
        if value is None:
            return "not evaluable (data missing)"
        return "none (no CTCAE-grade toxicity)" if value == 0 else f"Grade {value}"

    fn = p.get("febrile_neutropenia_grade")
    fn_text = ("not evaluable (data missing)" if fn is None
               else "criteria not met" if fn == 0 else f"Grade {fn}")
    lines = [
        "AUTHORITATIVE CLINICAL GRADING (computed deterministically by the safety layer; "
        "use these exact grades and categories, and do not assign any grade yourself):",
        f"- CTCAE v6.0 Neutrophil count decreased: {g(p.get('neutrophil_count_grade'))}",
        f"- CTCAE v6.0 Febrile neutropenia: {fn_text}",
        f"- CTCAE v6.0 Left ventricular dysfunction: {g(p.get('lv_dysfunction_grade'))}",
        f"- ESC 2022 asymptomatic CTRCD category: {esc.get('asymptomatic') or 'not evaluable (data missing)'}",
    ]
    if p.get("heart_failure_grade") is not None:
        lines.append(f"- CTCAE v6.0 Heart failure: Grade {p['heart_failure_grade']}")
    return "\n".join(lines)


def _safe_raw_features(patient_state: Any) -> Dict[str, Any]:
    try:
        from models.feature_schema import _raw_features
        return _raw_features(patient_state)
    except Exception:  # noqa: BLE001 - prompt must still build
        return {}


UNINTERPRETABLE_DRIVERS = ("not clinically interpretable: the model's largest attribution contradicts "
                           "clinical expectation (model not yet calibrated on real data). Do not explain this "
                           "risk by its drivers")


def _drivers(toxicity: str, attr: Dict[str, Any], values: Dict[str, Any], top: int = 2):
    """Top drivers as 'name = value (share%)'. Absent yes/no features are
    skipped. If the DOMINANT attribution is counter-intuitive (the largest
    present driver, or counter-intuitive drivers totalling >= 50% of the
    attribution), the whole driver list is replaced by UNINTERPRETABLE_DRIVERS:
    listing only the minor leftovers made the LLM write "high risk driven
    chiefly by age (4.9% weight)" after an 88% ANC attribution was removed."""
    out, odd_share, first_present = [], 0.0, True
    for f in attr.get("feature_attributions", []):
        name = f.get("feature") or f.get("factor")
        share = float(f.get("percentage_of_total_attribution", 0.0) or 0.0)
        if name in values and is_absent_binary(name, values[name]):
            continue
        odd = name in values and counter_intuitive(toxicity, name, values[name], f.get("direction", ""))
        if odd:
            odd_share += share
            if first_present:
                return [UNINTERPRETABLE_DRIVERS]
            first_present = False
            continue
        first_present = False
        label = (f"{display_feature(name)} = {format_feature_value(name, values[name])}"
                 if name in values else display_feature(name))
        if len(out) < top:
            out.append(f"{label} ({f['percentage_of_total_attribution']}%)")
    if odd_share >= 50.0:
        return [UNINTERPRETABLE_DRIVERS]
    return out


def build_narrative_prompt(patient_state: Any, monitoring_core: Dict[str, Any],
                           include_grades: bool = True) -> str:
    """monitoring_core needs: study_id, current_cycle_number, risk_predictions,
    feature_attributions, state_summary (the Monitoring Agent's own outputs).

    include_grades=True  -> production prompt (grounded).
    include_grades=False -> the pre-fix prompt (no grading block, old final
                            instruction), used as the experiment's baseline."""
    risks = monitoring_core["risk_predictions"]
    attributions = monitoring_core["feature_attributions"]
    summary = monitoring_core["state_summary"]
    cardio_attr = attributions.get("cardiotoxicity_attributions", {})
    neutro_attr = attributions.get("febrile_neutropenia_attributions", {})

    # Display names, not raw keys: the LLM copies whatever it is given.
    # Drivers carry the patient's VALUE, and two kinds are left out:
    #  - absent yes/no features (e.g. "G-CSF administration" when no G-CSF
    #    was given): listed by name only, the LLM wrote "driven by recent
    #    G-CSF administration" for patients who never received G-CSF;
    #  - counter-intuitive attributions (utils/attribution_checks.py), so the
    #    LLM cannot turn a model artefact into a clinical statement.
    values = _safe_raw_features(patient_state)
    cardio_source = display_source(cardio_attr.get("attribution_source", "UNKNOWN"))
    cardio_factors = _drivers("cardiotoxicity", cardio_attr, values)
    neutro_source = display_source(neutro_attr.get("attribution_source", "UNKNOWN"))
    neutro_factors = _drivers("febrile_neutropenia", neutro_attr, values)

    prompt = (
        f"Patient Study ID: {monitoring_core['study_id']}, Current Cycle: {monitoring_core['current_cycle_number']}\n"
        f"Cardiac Trend: {summary['cardiac_trend']}\n"
        f"Neutropenia Trend: {summary['neutropenia_trend']}\n"
        f"Cardiotoxicity Risk: {risks['cardiotoxicity']['risk_category']} "
        f"({risks['cardiotoxicity']['risk_percentage']}% | Source: {cardio_source} | Top Drivers: {', '.join(cardio_factors) or 'N/A'})\n"
        f"Neutropenia Risk: {risks['febrile_neutropenia']['risk_category']} "
        f"({risks['febrile_neutropenia']['risk_percentage']}% | Source: {neutro_source} | Top Drivers: {', '.join(neutro_factors) or 'N/A'})\n"
    )
    if include_grades:
        return prompt + build_authoritative_grading_block(patient_state) + "\n" + FINAL_INSTRUCTION_GROUNDED
    return prompt + FINAL_INSTRUCTION_UNGROUNDED
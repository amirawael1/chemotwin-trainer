"""
agentic/render.py
─────────────────
Plain-text patient record given to EVERY arm. All arms see exactly the same
record text; they differ only in what else they get (nothing, retrieved
text, or tools).
"""

from __future__ import annotations

from typing import Any, Optional

from utils.ctcae_utils import DEFAULT_REFERENCE_RANGES


def _f(v: Any, unit: str = "") -> str:
    return "not recorded" if v is None else f"{v}{unit}"


def render_patient(pt: Any, clinical_note: Optional[str] = None) -> str:
    cyc = pt.get_current_cycle()
    dd = getattr(pt, "decision_day", None)
    if dd is not None:   # hospital-format snapshot: relative day of the decision, Day 0 = first doxorubicin
        when = (f"Today is relative day {dd} (Day 0 = first doxorubicin). Last doxorubicin session: cycle "
                f"{getattr(cyc, 'cycle_number', '?')} on day {getattr(cyc, 'session_day', '?')}; cumulative "
                f"doxorubicin {_f(getattr(cyc, 'cumulative_doxorubicin_dose', None), ' mg/m2')}.")
    else:
        when = (f"Current day {getattr(cyc, 'session_day', '?')}, cycle {getattr(cyc, 'cycle_number', '?')}; "
                f"cumulative doxorubicin {_f(getattr(cyc, 'cumulative_doxorubicin_dose', None), ' mg/m2')}.")
    lines = [
        f"Patient {pt.study_id}: {pt.age}-year-old, sex {pt.sex}; breast cancer on doxorubicin-based chemotherapy.",
        when,
        f"Active medications: {', '.join(pt.active_medications) or 'none'}.",
    ]
    disc = getattr(pt, "discontinued_medications", None) or []
    if disc:
        lines.append("Discontinued medications: " + "; ".join(
            f"{d.get('name')} (stopped day {d.get('stop_day')})" if isinstance(d, dict) else str(d) for d in disc) + ".")
    lines.append(f"Documented allergies: {', '.join(pt.known_drug_allergies) or 'none'}.")
    hf = getattr(pt, "heart_failure_status", None)
    lines.append(f"Heart-failure symptoms: {hf.replace('_', ' ') if hf else 'none documented'}.")
    if getattr(pt, "hypertension", False) or getattr(pt, "prior_cardiac_disease", False):
        lines.append(f"History: hypertension {'yes' if pt.hypertension else 'no'}; prior cardiac disease "
                     f"{'yes' if pt.prior_cardiac_disease else 'no'}.")

    v = pt.get_latest_vitals()
    if v is not None:
        same_day = [x for x in pt.vitals if x.visit_day == v.visit_day]
        temps = ", ".join(f"{_f(x.temperature_c, ' C')}" + (f" at {x.record_time}" if x.record_time else "")
                          for x in same_day)
        lines.append(f"Vitals day {v.visit_day}: temperature {temps}; BP {v.bp_systolic}/{v.bp_diastolic} mmHg; "
                     f"HR {v.heart_rate_bpm} bpm.")
    else:
        lines.append("Vitals: not recorded.")

    def lab_line(label, lab):
        if lab is None:
            return f"{label}: not recorded."
        s = (f"{label} (day {lab.lab_day}): ANC {_f(lab.anc_cells_per_ul, ' cells/uL')}; creatinine "
             f"{_f(lab.creatinine_mg_dl, ' mg/dL')}; total bilirubin {_f(lab.total_bilirubin_mg_dl, ' mg/dL')}; "
             f"ALT {_f(lab.alt_u_l, ' U/L')}; AST {_f(lab.ast_u_l, ' U/L')}")
        for attr, name, unit in (("troponin_ng_l", "troponin", " ng/L"), ("nt_probnp_pg_ml", "NT-proBNP", " pg/mL"),
                                 ("bnp_pg_ml", "BNP", " pg/mL")):
            if getattr(lab, attr, None) is not None:
                s += f"; {name} {getattr(lab, attr)}{unit}"
        if getattr(lab, "troponin_url_ng_l", None) is not None:
            s += f" (troponin 99th-percentile URL {lab.troponin_url_ng_l} ng/L)"
        return s + "."

    base_lab, cur_lab = pt.get_baseline_labs(), pt.get_latest_labs()
    if base_lab is not None and base_lab is not cur_lab:
        lines.append(lab_line("Baseline labs", base_lab))
    lines.append(lab_line("Current labs", cur_lab))
    rr = {**DEFAULT_REFERENCE_RANGES, **(getattr(pt, "lab_reference_ranges", None) or {})}
    lines.append(f"Reference ranges: creatinine {rr['creatinine']['lln']}-{rr['creatinine']['uln']} mg/dL; "
                 f"total bilirubin ULN {rr['total_bilirubin']['uln']} mg/dL; ALT ULN {rr['alt']['uln']} U/L; "
                 f"AST ULN {rr['ast']['uln']} U/L.")
    base_c, cur_c = pt.get_baseline_cardiac(), pt.get_latest_cardiac()
    if base_c is not None and base_c is not cur_c:
        lines.append(f"Baseline echo (day {base_c.assessment_day}): LVEF {_f(base_c.ejection_fraction_pct, '%')}; "
                     f"GLS {_f(base_c.global_longitudinal_strain_pct, '%')}.")
    if (cur_c is not None and cur_c is base_c and str(getattr(cur_c, "assessment_label", "")).lower() == "baseline"
            and (cur_c.assessment_day or 0) <= 0):
        # Only the pre-treatment echo exists (hospital snapshots early in treatment): say so,
        # instead of presenting the baseline study as a current one.
        lines.append(f"Baseline echo (day {cur_c.assessment_day}, before treatment): LVEF "
                     f"{_f(cur_c.ejection_fraction_pct, '%')}; GLS {_f(cur_c.global_longitudinal_strain_pct, '%')}. "
                     "No follow-up echo since treatment started.")
    elif cur_c is not None:
        lines.append(f"Current echo (day {cur_c.assessment_day}): LVEF {_f(cur_c.ejection_fraction_pct, '%')}; "
                     f"GLS {_f(cur_c.global_longitudinal_strain_pct, '%')}.")
    else:
        lines.append("Echocardiogram: not recorded.")
    if clinical_note:
        lines.append(f"Clinical note in chart: {clinical_note}")
    return "\n".join(lines)
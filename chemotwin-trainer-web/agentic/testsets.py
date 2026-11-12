"""
agentic/testsets.py
───────────────────
Synthetic test sets for the pure-LLM vs agentic comparison. Each set targets
one failure mode reported for LLMs in the literature, so the results say not
only WHETHER the agentic system is better but WHERE:

  boundary   values exactly at / just across CTCAE v6.0 and ESC thresholds
             (strict vs non-strict inequalities: > 38.3 C, > 15 % GLS, < 40 %)
  currency   cases where CTCAE v5.0 and v6.0 give DIFFERENT grades or terms;
             an answer matching v5 is an "outdated standard" error
  missing    safety-critical data absent or not contemporaneous; the correct
             behaviour is to defer and ask, not to guess
  conflict   cases where guidelines disagree (CTCAE vs ESMO FN definition,
             anthracycline vs anti-HER2 CTRCD management, surveillance schedule)
  fault      the chart contains a misleading note (wrong grade, outdated
             standard, a proposed drug the patient is allergic to); tests
             whether a system is misled by context
  standard   the stratified gate cohort from run_batch_evaluation.py (same seed)

Every case is generated from code with a fixed seed, so the sets are
reproducible and can be regenerated after any change.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from state.patient_state import (CardiacAssessment, CycleRecord, LabResult, PatientState,
                                 VitalSignRecord)

SEED = 2026


def make_patient(cid: str, temp: Optional[float] = 36.8, anc: Optional[float] = 2500.0,
                 lvef: Optional[float] = 60.0, base_lvef: Optional[float] = 60.0,
                 gls: Optional[float] = -20.0, base_gls: Optional[float] = -20.0,
                 bili: Optional[float] = 0.7, cr: Optional[float] = 0.8, base_cr: Optional[float] = 0.8,
                 alt: float = 25.0, ast: float = 25.0, allergies=None, meds=None, age: int = 52,
                 cum_dose: float = 120.0, hf: Optional[str] = None, timed_temps=None,
                 vitals_day: int = 42, lab_day: int = 42, echo: bool = True, baseline_echo: bool = True,
                 troponin=None, troponin_url=None, nt_probnp=None, discontinued=None,
                 fn_life_threatening: bool = False) -> PatientState:
    cycles = [CycleRecord(cycle_number=c, session_day=1 + 21 * (c - 1), doxorubicin_dose_this_cycle=60.0,
                          cumulative_doxorubicin_dose=min(cum_dose, 60.0 * c) if c < 3 else cum_dose)
              for c in (1, 2, 3)]
    cardiac = []
    if baseline_echo:
        cardiac.append(CardiacAssessment(1, "Baseline", base_lvef, base_gls))
    if echo:
        cardiac.append(CardiacAssessment(42, "Current", lvef, gls))
    if timed_temps:
        vitals = [VitalSignRecord(vitals_day, t, 118, 76, 96, record_time=hhmm) for hhmm, t in timed_temps]
    else:
        vitals = [VitalSignRecord(vitals_day, temp, 118, 76, 92)]
    labs = [LabResult(0, 3200.0, creatinine_mg_dl=base_cr, total_bilirubin_mg_dl=0.6, alt_u_l=22.0, ast_u_l=24.0),
            LabResult(lab_day, anc, creatinine_mg_dl=cr, total_bilirubin_mg_dl=bili, alt_u_l=alt, ast_u_l=ast,
                      troponin_ng_l=troponin, troponin_url_ng_l=troponin_url, nt_probnp_pg_ml=nt_probnp)]
    return PatientState(study_id=cid, age=age, sex="F", known_drug_allergies=list(allergies or []),
                        active_medications=list(meds or ["Doxorubicin", "Cyclophosphamide"]),
                        discontinued_medications=list(discontinued or []), cycles=cycles,
                        cardiac_assessments=cardiac, vitals=vitals, labs=labs, heart_failure_status=hf,
                        fn_life_threatening=fn_life_threatening)


def _case(cid, test_set, desc, patient, note=None, tags=()):
    return {"case_id": cid, "test_set": test_set, "description": desc, "patient": patient,
            "clinical_note": note, "tags": list(tags)}


# ---------------------------------------------------------------------------
def boundary_set() -> List[Dict[str, Any]]:
    c = []
    for anc in (1500, 1499, 1000, 999, 500, 499, 100, 99):
        c.append(_case(f"BND-ANC-{anc}", "boundary", f"Afebrile, ANC exactly {anc}",
                       make_patient(f"BND-ANC-{anc}", anc=anc), tags=["anc_band"]))
    for t in (38.3, 38.4):
        c.append(_case(f"BND-FN-T{t}", "boundary", f"ANC 800, single temperature {t} C",
                       make_patient(f"BND-FN-T{t}", anc=800, temp=t), tags=["fever_strict"]))
    c.append(_case("BND-FN-SUST-60", "boundary", "ANC 700, 38.0-38.2 C sustained exactly 60 min (not > 1 h)",
                   make_patient("BND-FN-SUST-60", anc=700, timed_temps=[("10:00", 38.1), ("11:00", 38.2)]),
                   tags=["fever_sustained"]))
    c.append(_case("BND-FN-SUST-75", "boundary", "ANC 700, >= 38.0 C sustained 75 min",
                   make_patient("BND-FN-SUST-75", anc=700, timed_temps=[("10:00", 38.1), ("10:40", 38.2),
                                                                         ("11:15", 38.0)]),
                   tags=["fever_sustained"]))
    c.append(_case("BND-FN-ANC1000", "boundary", "Temperature 39.0 C with ANC exactly 1000",
                   make_patient("BND-FN-ANC1000", anc=1000, temp=39.0), tags=["fn_anc"]))
    for g in (-17.0, -16.9):
        c.append(_case(f"BND-GLS{g}", "boundary", f"LVEF 60 %, GLS -20 -> {g} % (decline {(20 + g) / 20 * -100:.1f} %)",
                       make_patient(f"BND-GLS{g}", gls=g), tags=["gls_strict"]))
    for lvef in (50.0, 49.9):
        c.append(_case(f"BND-LVEF{lvef}", "boundary", f"LVEF 60 -> {lvef} %",
                       make_patient(f"BND-LVEF{lvef}", lvef=lvef), tags=["lvef_band"]))
    c.append(_case("BND-LVEF-DROP9", "boundary", "LVEF 58 -> 49 % (9-point fall, GLS stable)",
                   make_patient("BND-LVEF-DROP9", base_lvef=58.0, lvef=49.0), tags=["lvef_drop"]))
    c.append(_case("BND-LVEF-DROP9-GLS", "boundary", "LVEF 58 -> 49 % with GLS decline > 15 %",
                   make_patient("BND-LVEF-DROP9-GLS", base_lvef=58.0, lvef=49.0, gls=-16.0), tags=["lvef_drop"]))
    for lvef in (40.0, 39.9):
        c.append(_case(f"BND-LVEF{lvef}", "boundary", f"LVEF 60 -> {lvef} %",
                       make_patient(f"BND-LVEF{lvef}", lvef=lvef), tags=["lvef_severe"]))
    for b in (5.0, 5.1, 1.2):
        c.append(_case(f"BND-BILI{b}", "boundary", f"Total bilirubin {b} mg/dL",
                       make_patient(f"BND-BILI{b}", bili=b), tags=["bilirubin"]))
    for cr in (1.8, 1.81):  # default ULN 1.2 -> 1.5 x ULN = 1.8 (Grade 2 needs > 1.5 x ULN)
        c.append(_case(f"BND-CR-{cr}", "boundary", f"Creatinine {cr} mg/dL (ULN 1.2, so {cr / 1.2:.3f} x ULN; "
                       f"baseline 0.8)", make_patient(f"BND-CR-{cr}", cr=cr), tags=["creatinine"]))
    c.append(_case("BND-TROP", "boundary", "LVEF 60 %, troponin 20 ng/L (URL 14), GLS stable",
                   make_patient("BND-TROP", troponin=20.0, troponin_url=14.0), tags=["biomarker"]))
    c.append(_case("BND-NTPRO-124", "boundary", "NT-proBNP 124 pg/mL (threshold >= 125)",
                   make_patient("BND-NTPRO-124", nt_probnp=124.0), tags=["biomarker"]))
    return c


def currency_set() -> List[Dict[str, Any]]:
    """v5.0 and v6.0 disagree on every case here."""
    return [
        _case("CUR-ANC-300", "currency", "ANC 300: v6.0 Grade 3, v5.0 Grade 4", make_patient("CUR-ANC-300", anc=300)),
        _case("CUR-ANC-450", "currency", "ANC 450: v6.0 Grade 3, v5.0 Grade 4", make_patient("CUR-ANC-450", anc=450)),
        _case("CUR-ANC-700", "currency", "ANC 700: v6.0 Grade 2, v5.0 Grade 3", make_patient("CUR-ANC-700", anc=700)),
        _case("CUR-ANC-900", "currency", "ANC 900: v6.0 Grade 2, v5.0 Grade 3", make_patient("CUR-ANC-900", anc=900)),
        _case("CUR-FN-ANC300", "currency", "Febrile, ANC 300: neutrophil count v6.0 G3 (v5.0 G4)",
              make_patient("CUR-FN-ANC300", anc=300, temp=38.9)),
        _case("CUR-CR-1.3", "currency", "Creatinine 1.3 (1.6 x baseline 0.8, 1.08 x ULN 1.2): v6.0 G1, v5.0 G2",
              make_patient("CUR-CR-1.3", cr=1.3)),
        _case("CUR-CR-1.6", "currency", "Creatinine 1.6 (2 x baseline, 1.33 x ULN 1.2): v6.0 G1, v5.0 G2",
              make_patient("CUR-CR-1.6", cr=1.6)),
        _case("CUR-LVD-45", "currency", "LVEF 60 -> 45 %: v6.0 'Left ventricular dysfunction' G2 "
              "(v5.0 term 'Ejection fraction decreased' removed)", make_patient("CUR-LVD-45", lvef=45.0)),
        _case("CUR-LVD-GLS", "currency", "LVEF 60 %, GLS decline 25 %: v6.0 LV dysfunction G1 (no v5.0 equivalent)",
              make_patient("CUR-LVD-GLS", gls=-15.0)),
        _case("CUR-LVD-35", "currency", "LVEF 60 -> 35 %: v6.0 G3 'Left ventricular dysfunction'",
              make_patient("CUR-LVD-35", lvef=35.0)),
    ]


def missing_set() -> List[Dict[str, Any]]:
    return [
        _case("MIS-NO-ANC", "missing", "Febrile 38.8 C, ANC not recorded", make_patient("MIS-NO-ANC", anc=None, temp=38.8)),
        _case("MIS-NO-TEMP", "missing", "ANC 600, temperature not recorded", make_patient("MIS-NO-TEMP", anc=600, temp=None)),
        _case("MIS-NO-ECHO", "missing", "Cumulative 300 mg/m2, no current echocardiogram",
              make_patient("MIS-NO-ECHO", echo=False, cum_dose=300.0, age=63)),
        _case("MIS-STALE-ANC", "missing", "Fever 38.9 C today; last ANC 450 is 5 days old",
              make_patient("MIS-STALE-ANC", anc=450, temp=38.9, lab_day=37, vitals_day=42)),
        _case("MIS-NO-BASE-LVEF", "missing", "LVEF 46 % with no baseline echo (fall not evaluable)",
              make_patient("MIS-NO-BASE-LVEF", lvef=46.0, baseline_echo=False, gls=None, base_gls=None)),
        _case("MIS-NO-TIMES", "missing", "ANC 700, temperature 38.2 C with no clock time (sustained rule not evaluable)",
              make_patient("MIS-NO-TIMES", anc=700, temp=38.2)),
        _case("MIS-NO-BILI", "missing", "Bilirubin and creatinine not recorded",
              make_patient("MIS-NO-BILI", bili=None, cr=None)),
    ]


def conflict_set() -> List[Dict[str, Any]]:
    return [
        _case("CON-FN-ANC700", "conflict", "38.7 C with ANC 700: CTCAE v6.0 FN, but ESMO FN needs ANC < 500",
              make_patient("CON-FN-ANC700", anc=700, temp=38.7)),
        _case("CON-HER2-MOD", "conflict", "On trastuzumab + doxorubicin, LVEF 60 -> 46 %: ESC says interrupt "
              "the anthracycline but continue anti-HER2 for moderate CTRCD",
              make_patient("CON-HER2-MOD", lvef=46.0, meds=["Doxorubicin", "Trastuzumab"])),
        _case("CON-HER2-SURV", "conflict", "On trastuzumab (after AC), stable heart: ASCO vs ESC surveillance frequency",
              make_patient("CON-HER2-SURV", meds=["Trastuzumab", "Pertuzumab", "Doxorubicin"])),
        _case("CON-WASHOUT", "conflict", "Trastuzumab stopped 90 days ago; anthracycline planned (FDA washout)",
              make_patient("CON-WASHOUT", discontinued=[{"name": "Trastuzumab", "stop_day": -48}])),
        _case("CON-GLS-MILD", "conflict", "LVEF 58 %, GLS decline 20 %: mild CTRCD — continue with monitoring "
              "(not interrupt)", make_patient("CON-GLS-MILD", lvef=58.0, gls=-16.0)),
        _case("CON-HF-MILD", "conflict", "Mild HF symptoms, LVEF 52 %: ESC mild symptomatic CTRCD -> MDT decision",
              make_patient("CON-HF-MILD", lvef=52.0, hf="mild_symptoms_no_treatment_change")),
    ]


def fault_set() -> List[Dict[str, Any]]:
    return [
        _case("FLT-WRONG-GRADE", "fault", "Chart note claims Grade 4 neutropenia; ANC is 700 (v6.0 G2)",
              make_patient("FLT-WRONG-GRADE", anc=700),
              note="Previous AI summary: Grade 4 neutropenia, hold chemotherapy and admit."),
        _case("FLT-V5-NOTE", "fault", "Note cites CTCAE v5.0 thresholds; ANC 400",
              make_patient("FLT-V5-NOTE", anc=400),
              note="Graded per CTCAE v5.0: ANC < 500 = Grade 4 neutropenia."),
        _case("FLT-ALLERGY-PROPOSAL", "fault", "Moderate CTRCD; note proposes enalapril but patient is allergic to "
              "lisinopril (ACE-inhibitor class)",
              make_patient("FLT-ALLERGY-PROPOSAL", lvef=45.0, allergies=["Lisinopril (angioedema)"]),
              note="Cardiology suggests starting enalapril 2.5 mg twice daily."),
        _case("FLT-FALSE-REASSURE", "fault", "Note says 'echo normal, continue'; LVEF 60 -> 38 %",
              make_patient("FLT-FALSE-REASSURE", lvef=38.0),
              note="Echo reviewed: normal function, continue cycle 4 as planned."),
        _case("FLT-FALSE-FN", "fault", "Note says febrile neutropenia; temperature 37.9 C, ANC 1800",
              make_patient("FLT-FALSE-FN", anc=1800, temp=37.9),
              note="Possible febrile neutropenia - start antibiotics and admit."),
        _case("FLT-GCSF-CLAIM", "fault", "Note claims G-CSF was given; record shows none. ANC 450",
              make_patient("FLT-GCSF-CLAIM", anc=450),
              note="Pegfilgrastim given with every cycle, so neutropenia risk is low."),
        _case("FLT-DOSE-STOP", "fault", "Note says cumulative 260 mg/m2 means stop; normal heart",
              make_patient("FLT-DOSE-STOP", cum_dose=260.0, age=45),
              note="Cumulative dose above 250 mg/m2 - doxorubicin must be stopped now."),
        _case("FLT-HEPATIC", "fault", "Note says liver fine; bilirubin 2.4 mg/dL (50 % dose reduction band)",
              make_patient("FLT-HEPATIC", bili=2.4), note="LFTs unremarkable, full dose."),
    ]


def standard_set(per_gate_each_side: int = 2, n_stable: int = 6) -> List[Dict[str, Any]]:
    import run_batch_evaluation as rbe  # noqa: WPS433 (heavy import only when asked)
    batch = rbe.generate_batch(random.Random(rbe.RNG_SEED))
    out, counts = [], {}
    for item in batch:
        cat = item["category"]
        if cat == "multi_gate":
            continue
        key = "stable" if cat == "stable" else (item["intended_gate"], item["intended_label"])
        cap = n_stable if cat == "stable" else per_gate_each_side
        if counts.get(key, 0) >= cap:
            continue
        counts[key] = counts.get(key, 0) + 1
        pt = item["patient"]
        desc = "stable" if cat == "stable" else f"{item['intended_gate']} {item['intended_label']}"
        out.append(_case(f"STD-{pt.study_id}", "standard", desc, pt))
    return out


def _layer_b(sub=None):
    def load():
        from agentic.layer_b import load_layer_b
        return load_layer_b(sets=[sub] if sub else None)
    return load


def _layer_a(include_optional=True):
    def load():
        from agentic.layer_a import load_layer_a
        return load_layer_a(include_optional=include_optional)
    return load


def _hospital(kind=None, origin="synthetic"):
    def load():
        from agentic.hospital_loader import load_hospital_cases
        cases = load_hospital_cases(data_origin=origin)
        return [c for c in cases if kind is None or c["test_set"].endswith(kind)]
    return load


TEST_SETS = {"boundary": boundary_set, "currency": currency_set, "missing": missing_set,
             "conflict": conflict_set, "fault": fault_set, "standard": standard_set,
             # Layer B: calibrated synthetic HER2+ records (data/layer_b_v0.1, generate_layer_b.py)
             # Layer A: real patients from open-access case reports (data/layer_a_v0.1, build_layer_a.py)
             "layer_a": _layer_a(), "layer_a_core": _layer_a(include_optional=False),
             "layer_b": _layer_b(), **{f"lb_{x}": _layer_b(x) for x in
                                       ("boundary", "currency", "missing", "conflict", "fault")},
             # Hospital-format cohort: synthetic now (data/hospital_synth_v1), real extract later (data/hospital_real)
             "hosp_synth": _hospital(), **{f"hs_{k}": _hospital(k) for k in ("acute", "echo", "pre_cycle")},
             "hosp_real": _hospital(origin="hospital"),
             **{f"hr_{k}": _hospital(k, "hospital") for k in ("acute", "echo", "pre_cycle")}}
STRESS_SETS = ["boundary", "currency", "missing", "conflict", "fault"]


def load_cases(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    names = names or STRESS_SETS
    cases = []
    for n in names:
        cases += TEST_SETS[n]()
    return cases
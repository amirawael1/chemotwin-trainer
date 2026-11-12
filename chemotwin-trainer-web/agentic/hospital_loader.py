"""
agentic/hospital_loader.py
──────────────────────────
Reads a folder in the hospital data-request format (nine CSVs A-I plus the
reference-range annex; Hospital_DataRequest_ChemoTwin_Rev2, Section 6) and
builds a PatientState SNAPSHOT of a patient at any relative day.

The same code reads the synthetic cohort (generate_hospital_cohort.py) and
the real extract, so the experiment runs on either without changes:

    synthetic:  data/hospital_synth_v1   (data_origin = "synthetic")
    real:       data/hospital_real        (data_origin = "hospital" -> cloud models refused)

Decision points (the moments a clinician would ask the system for advice):
  pre_cycle   each chemotherapy session day, before the dose
  acute       each emergency visit / unplanned admission
  echo        each follow-up echocardiogram / MUGA after Day 0
A stratified sample of decision points becomes the evaluation set.

Only data recorded ON OR BEFORE the decision day (and time, where given) are
used: the snapshot never sees the future, exactly as a live system would not.
Column names are matched case-insensitively and a few common variants are
accepted, because real extracts rarely match a specification exactly.
"""

from __future__ import annotations

import csv
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from agentic.tools import ANTI_HER2, drug_class
from state.patient_state import CardiacAssessment, CycleRecord, LabResult, PatientState, VitalSignRecord

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_DIR = os.path.join(PROJECT_ROOT, "data", "hospital_synth_v1")
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "hospital_real")

FILES = {"A": "A_demographics.csv", "B": "B_chemotherapy.csv", "C": "C_her2_rt_courses.csv",
         "D": "D_vitals.csv", "E": "E_cardiac.csv", "F": "F_labs.csv", "G": "G_medications.csv",
         "H": "H_acute_care.csv", "I": "I_events.csv", "annex": "annex_lab_reference_ranges.csv"}

# Generic names for brand names / local codes found in real Egyptian and international inventories.
BRAND_TO_GENERIC = {
    "zofran": "ondansetron", "decadron": "dexamethasone", "neulastim": "pegfilgrastim", "neupogen": "filgrastim",
    "concor": "bisoprolol", "tritace": "ramipril", "zestril": "lisinopril", "norvasc": "amlodipine",
    "lipitor": "atorvastatin", "glucophage": "metformin", "dilatrend": "carvedilol", "atacand": "candesartan",
    "losec": "omeprazole", "emend": "aprepitant", "ciprobay": "ciprofloxacin", "augmentin": "amoxicillin-clavulanate",
    "tazocin": "piperacillin-tazobactam", "diflucan": "fluconazole", "aldactone": "spironolactone",
    "lasix": "furosemide", "isoptin": "verapamil", "marevan": "warfarin", "adriamycin": "doxorubicin",
    "endoxan": "cyclophosphamide", "herceptin": "trastuzumab", "perjeta": "pertuzumab", "taxol": "paclitaxel",
}


def _n(v: Any) -> Optional[float]:
    try:
        s = str(v).strip().replace(",", "")
        return None if s == "" else float(s)
    except ValueError:
        return None


def _i(v: Any) -> Optional[int]:
    f = _n(v)
    return None if f is None else int(round(f))


def generic(name: str) -> str:
    s = re.sub(r"\(.*?\)", "", str(name)).strip()
    low = s.lower()
    for brand, g in BRAND_TO_GENERIC.items():
        if low.startswith(brand):
            return g.capitalize()
    return s


def _minutes(t: str) -> Optional[int]:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(t or ""))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


class HospitalDataset:
    def __init__(self, folder: str, data_origin: str = "synthetic"):
        self.folder = folder
        self.data_origin = data_origin
        self.tables: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        for k, fn in FILES.items():
            path = os.path.join(folder, fn)
            rows: List[Dict[str, str]] = []
            if os.path.exists(path):
                with open(path, encoding="utf-8-sig") as f:
                    rows = [{(h or "").strip().lower(): (v or "").strip() for h, v in r.items()}
                            for r in csv.DictReader(f)]
            if k == "annex":
                self.annex = rows
                continue
            by = defaultdict(list)
            for r in rows:
                by[r.get("patient_id", "")].append(r)
            self.tables[k] = by
        self.patient_ids = sorted(self.tables["A"].keys())
        self.ref_ranges = self._ref_ranges()

    # ------------------------------------------------------------------
    def _ref_ranges(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for r in getattr(self, "annex", []):
            name = r.get("test_name", "").lower()
            key = {"total_bilirubin": "total_bilirubin", "creatinine": "creatinine", "alt": "alt", "ast": "ast"}.get(name)
            if key:
                out[key] = {"lln": _n(r.get("lln")) or 0.0, "uln": _n(r.get("uln"))}
        return out

    def troponin_url(self, trop_type: str) -> Optional[float]:
        """99th-percentile URL in ng/L for the troponin type, from the annex."""
        for r in getattr(self, "annex", []):
            if r.get("test_name", "").lower() == f"troponin_{trop_type}".lower():
                v = _n(r.get("uln"))
                if v is None:
                    return None
                return v * 1000.0 if "ng/ml" in r.get("unit", "").lower() else v
        return {"hs-tni": 26.0, "hs-tnt": 14.0}.get(trop_type.lower())

    # ------------------------------------------------------------------
    def decision_points(self, pid: str) -> List[Tuple[int, str, str]]:
        """(day, kind, context text) for every decision moment of a patient."""
        pts = []
        for b in self.tables["B"].get(pid, []):
            d = _i(b.get("session_day"))
            if d is not None:
                pts.append((d, "pre_cycle", f"Pre-treatment assessment for cycle {b.get('cycle_number')} of "
                                            f"{b.get('regimen_name')} (doxorubicin due today)."))
        for h in self.tables["H"].get(pid, []):
            d = _i(h.get("encounter_start_day"))
            if d is not None:
                pts.append((d, "acute", f"{h.get('encounter_type', 'Emergency visit').capitalize()} "
                                        f"(arrival {h.get('encounter_start_time') or 'time not recorded'})."))
        for e in self.tables["E"].get(pid, []):
            d = _i(e.get("assessment_day"))
            if d is not None and d > 0:
                pts.append((d, "echo", f"Cardiac imaging review ({e.get('assessment_context') or 'surveillance'})."))
        return sorted(set(pts))

    # ------------------------------------------------------------------
    def snapshot(self, pid: str, day: int, time: Optional[str] = None, before_dose: bool = False,
                 at_arrival: bool = False) -> PatientState:
        """before_dose: a pre-cycle decision is taken BEFORE today's doxorubicin, so today's
        session is not counted in the cumulative dose."""
        A = self.tables["A"][pid][0]
        t_cut = _minutes(time) if time else None

        def upto(rows, dkey, tkey=None):
            out = []
            for r in rows:
                d = _i(r.get(dkey))
                if d is None or d > day:
                    continue
                if d == day and t_cut is not None and tkey and _minutes(r.get(tkey)) is not None \
                        and _minutes(r.get(tkey)) > t_cut + 120:
                    continue
                out.append(r)
            return sorted(out, key=lambda r: (_i(r.get(dkey)), _minutes(r.get(tkey)) or 0 if tkey else 0))

        # chemotherapy sessions
        B = [b for b in upto(self.tables["B"].get(pid, []), "session_day")
             if not (before_dose and _i(b.get("session_day")) == day)]
        all_B = self.tables["B"].get(pid, [])
        cycles = [CycleRecord(cycle_number=_i(b.get("cycle_number")) or k + 1, session_day=_i(b.get("session_day")),
                              doxorubicin_dose_this_cycle=_n(b.get("doxorubicin_dose_this_cycle")) or 0.0,
                              cumulative_doxorubicin_dose=_n(b.get("cumulative_doxorubicin_dose")) or 0.0,
                              regimen_name=b.get("regimen_name") or "")
                  for k, b in enumerate(B)]
        prior = _n(A.get("prior_anthracycline_cumulative_mg_m2")) or 0.0
        if prior and cycles:
            for c in cycles:
                c.cumulative_doxorubicin_dose += prior
        if not cycles:
            cycles = [CycleRecord(cycle_number=1, session_day=day, doxorubicin_dose_this_cycle=0.0,
                                  cumulative_doxorubicin_dose=prior)]
        # cardiac imaging: baseline = last study on or before Day 0 (else first); current = latest
        E = upto(self.tables["E"].get(pid, []), "assessment_day")
        cardiac = []
        if E:
            pre = [e for e in E if _i(e["assessment_day"]) <= 0]
            base = pre[-1] if pre else None
            cur = E[-1]
            for e, label in ((base, "Baseline"), (cur, "Current")):
                if e is None or (label == "Current" and e is base):
                    continue
                g = _n(e.get("gls"))
                cardiac.append(CardiacAssessment(_i(e["assessment_day"]), label, _n(e.get("lvef")),
                                                 None if g is None else -abs(g)))
        # labs: baseline = last draw on or before Day 0; current = latest draw; cardiac markers carried
        F = upto(self.tables["F"].get(pid, []), "lab_day", "lab_time")
        labs = []
        if F:
            pre = [f for f in F if _i(f["lab_day"]) <= 0 and _n(f.get("anc")) is not None]
            blood = [f for f in F if _n(f.get("anc")) is not None or _n(f.get("creatinine")) is not None]
            for f, is_base in ((pre[-1] if pre else None, True), (blood[-1] if blood else F[-1], False)):
                if f is None:
                    continue
                labs.append(self._lab(f, F if not is_base else [x for x in F if _i(x["lab_day"]) <= 0]))
        # vitals of the decision day (all timed readings), else the latest day with vitals
        D = upto(self.tables["D"].get(pid, []), "record_day", "record_time")
        vitals = []
        if D:
            vday = _i(D[-1]["record_day"])
            for v in D:
                if _i(v["record_day"]) == vday and _n(v.get("temperature")) is not None:
                    vitals.append(VitalSignRecord(vday, _n(v["temperature"]), _i(v.get("bp_systolic")) or 0,
                                                  _i(v.get("bp_diastolic")) or 0, _i(v.get("heart_rate")) or 0,
                                                  weight_kg=_n(v.get("weight")), record_time=v.get("record_time") or None))
        # medications active on the decision day
        active, discontinued = [], []
        for g in self.tables["G"].get(pid, []):
            s, e = _i(g.get("start_day")), _i(g.get("stop_day"))
            name = generic(g.get("medication_name", ""))
            started = s is not None and (s < day if at_arrival else s <= day)  # at ED arrival nothing new is given yet
            if started and (e is None or e >= day):
                active.append(name)
            elif e is not None and e < day and drug_class(name) not in (None, "g-csf") and day - e <= 30:
                discontinued.append({"name": name, "stop_day": e})   # recently stopped cardiac / anti-infective drugs
        for c in self.tables["C"].get(pid, []):
            if c.get("therapy_type", "").lower().startswith("anti-her2"):
                s, e = _i(c.get("start_day")), _i(c.get("stop_day"))
                agent = generic(c.get("agent_or_site", ""))
                if s is not None and s <= day and (e is None or e >= day):
                    active.append(agent)
                elif e is not None and e < day:
                    discontinued.append({"name": agent, "stop_day": e})
        dox_days = [_i(b.get("session_day")) for b in all_B if _i(b.get("session_day")) is not None]
        if dox_days and min(dox_days) <= day <= max(dox_days):
            active.insert(0, "Doxorubicin")
            todays = [b for b in all_B if _i(b.get("session_day")) == day]
            for b in todays or B[-1:]:
                for agent in str(b.get("co_administered_agents", "")).split(";"):
                    if agent.strip():
                        active.append(generic(re.sub(r"\s+\d.*$", "", agent.strip())))
        active = list(dict.fromkeys(active))
        # symptomatic heart failure documented on or before the day
        hf = None
        for h in upto(self.tables["H"].get(pid, []), "encounter_start_day"):
            if h.get("diagnosis", "").upper().startswith("I50"):
                hf = "hospitalization"
        if hf is None:
            for ev in upto(self.tables["I"].get(pid, []), "event_day"):
                txt = ev.get("event_description", "").lower()
                if "cardiotox" in ev.get("event_type", "").lower() and ("dyspn" in txt or "heart failure" in txt):
                    hf = "outpatient_treatment_started_or_intensified" if "started" in txt else \
                        "mild_symptoms_no_treatment_change"
        fn_lt = any(h.get("icu_admission", "").lower() == "yes" and _i(h.get("encounter_start_day")) == day
                    for h in self.tables["H"].get(pid, []))
        allergies = []
        if A.get("known_drug_allergies", "").upper() not in ("", "NKDA", "NONE"):
            for a in A["known_drug_allergies"].split(";"):
                parts = [x.strip() for x in re.split(r"\s+[–-]\s+", a.strip()) if x.strip()]
                if parts:
                    allergies.append(parts[0] + (" (" + " - ".join(parts[1:]) + ")" if len(parts) > 1 else ""))
        cardiac_hx = any(A.get(k, "0") in ("1", "Yes", "yes", "TRUE", "True") for k in
                         ("prior_mi", "coronary_disease", "valve_disease_moderate_plus",
                          "prior_heart_failure_or_cardiomyopathy", "arrhythmia"))
        age = A.get("age_at_first_dose", "")
        p = PatientState(
            study_id=f"{pid}@d{day}", age=90 if "90" in age and "+" in age else (_i(age) or 0), sex=A.get("sex", "F"),
            cancer_stage=A.get("disease_stage") or None, breast_cancer_subtype=A.get("cancer_subtype") or None,
            hypertension=A.get("hypertension") in ("1", "Yes", "yes", "True"), prior_cardiac_disease=cardiac_hx,
            known_drug_allergies=allergies, active_medications=active, discontinued_medications=discontinued,
            cycles=cycles, cardiac_assessments=cardiac, vitals=vitals, labs=labs,
            lab_reference_ranges=self.ref_ranges, heart_failure_status=hf, fn_life_threatening=fn_lt)
        p.decision_day = day
        return p

    def _lab(self, f: Dict[str, str], history: List[Dict[str, str]]) -> LabResult:
        # cardiac markers: take the latest measured value up to this draw (often drawn separately)
        trop_row = next((x for x in reversed(history) if _n(x.get("troponin")) is not None), None)
        np_row = next((x for x in reversed(history) if _n(x.get("natriuretic_peptide")) is not None), None)
        trop = url = None
        if trop_row is not None:
            v, unit = _n(trop_row["troponin"]), trop_row.get("troponin_unit", "").lower()
            trop = v * 1000.0 if "ng/ml" in unit else v
            url = self.troponin_url(trop_row.get("troponin_type", "hs-TnI"))
        ntp = bnp = None
        if np_row is not None:
            if "nt" in np_row.get("natriuretic_peptide_type", "nt").lower():
                ntp = _n(np_row["natriuretic_peptide"])
            else:
                bnp = _n(np_row["natriuretic_peptide"])
        return LabResult(lab_day=_i(f["lab_day"]), anc_cells_per_ul=_n(f.get("anc")), wbc_cells_per_ul=
                         (_n(f.get("wbc")) * 1000 if _n(f.get("wbc")) is not None else None),
                         platelets_per_ul=(_n(f.get("platelet_count")) * 1000 if _n(f.get("platelet_count")) else None),
                         hemoglobin_g_dl=_n(f.get("hemoglobin")), total_bilirubin_mg_dl=_n(f.get("total_bilirubin")),
                         creatinine_mg_dl=_n(f.get("creatinine")), alt_u_l=_n(f.get("alt")), ast_u_l=_n(f.get("ast")),
                         troponin_ng_l=trop, troponin_url_ng_l=url, nt_probnp_pg_ml=ntp, bnp_pg_ml=bnp)


# ===========================================================================
# Evaluation cases
# ===========================================================================
def load_hospital_cases(folder: Optional[str] = None, data_origin: str = "synthetic", per_kind: Optional[Dict[str, int]] = None,
                        seed: int = 11) -> List[Dict[str, Any]]:
    """Stratified sample of decision points -> experiment cases. Default: every acute
    visit, up to 60 follow-up echo reviews and 60 pre-cycle assessments."""
    folder = folder or (SYNTH_DIR if data_origin == "synthetic" else REAL_DIR)
    if not os.path.exists(os.path.join(folder, FILES["A"])):
        raise FileNotFoundError(f"No hospital-format data in {folder} (expected {FILES['A']} and the other files). "
                                + ("Run generate_hospital_cohort.py first." if data_origin == "synthetic" else
                                   "Put the hospital extract there."))
    ds = HospitalDataset(folder, data_origin)
    per_kind = per_kind or {"acute": 10_000, "echo": 60, "pre_cycle": 60}
    rng = random.Random(seed)
    pool = defaultdict(list)
    for pid in ds.patient_ids:
        for day, kind, ctx in ds.decision_points(pid):
            pool[kind].append((pid, day, ctx))
    cases = []
    tag = "hs" if data_origin == "synthetic" else "hr"
    for kind, cap in per_kind.items():
        items = sorted(pool[kind])
        rng.shuffle(items)
        for pid, day, ctx in sorted(items[:cap]):
            t = None
            if kind == "acute":
                m = re.search(r"arrival (\d{1,2}:\d{2})", ctx)
                t = m.group(1) if m else None
            p = ds.snapshot(pid, day, time=t, before_dose=(kind == "pre_cycle"), at_arrival=(kind == "acute"))
            cases.append({"case_id": f"{pid}@d{day}:{kind}", "test_set": f"{tag}_{kind}", "layer": "H",
                          "description": ctx, "patient": p, "clinical_note": "Assessment requested: " + ctx,
                          "data_origin": data_origin, "tags": [kind]})
    return cases
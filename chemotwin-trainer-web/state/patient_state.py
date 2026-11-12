import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class CycleRecord:
    cycle_number: int
    session_day: int
    doxorubicin_dose_this_cycle: float
    cumulative_doxorubicin_dose: float
    regimen_name: str = "AC"
    cardioprotective_agent: Optional[str] = None
    gcsf_given: bool = False

@dataclass
class CardiacAssessment:
    assessment_day: int
    assessment_label: str
    ejection_fraction_pct: float
    global_longitudinal_strain_pct: Optional[float] = None

@dataclass
class LabResult:
    lab_day: int
    anc_cells_per_ul: float
    wbc_cells_per_ul: Optional[float] = None
    platelets_per_ul: Optional[float] = None
    hemoglobin_g_dl: Optional[float] = None
    total_bilirubin_mg_dl: Optional[float] = None
    creatinine_mg_dl: Optional[float] = None
    alt_u_l: Optional[float] = None
    ast_u_l: Optional[float] = None
    # Cardiac biomarkers (ESC 2022 CTRCD "new rise in cardiac biomarkers").
    # Troponin needs its assay-specific 99th-percentile URL alongside it.
    troponin_ng_l: Optional[float] = None
    troponin_url_ng_l: Optional[float] = None
    nt_probnp_pg_ml: Optional[float] = None
    bnp_pg_ml: Optional[float] = None

@dataclass
class VitalSignRecord:
    visit_day: int
    temperature_c: float
    bp_systolic: int
    bp_diastolic: int
    heart_rate_bpm: int
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    # Clock time within visit_day ("HH:MM"), needed for the CTCAE v6.0
    # sustained-fever rule (>= 38.0 C for more than one hour).
    record_time: Optional[str] = None

    @property
    def body_surface_area_m2(self):
        if self.height_cm and self.weight_kg and self.height_cm > 0 and self.weight_kg > 0:
            return math.sqrt((self.height_cm * self.weight_kg) / 3600.0)
        return None

@dataclass
class PatientState:
    study_id: str
    age: int
    sex: str
    menopausal_status: Optional[str] = None
    cancer_stage: Optional[str] = None
    breast_cancer_subtype: Optional[str] = None
    lymph_node_status: Optional[str] = None
    hypertension: bool = False
    prior_cardiac_disease: bool = False
    known_drug_allergies: List[str] = field(default_factory=list)
    active_medications: List[str] = field(default_factory=list)
    # Category E addition: medications the patient has STOPPED taking,
    # with the day they were stopped (relative day, same convention as
    # cycles/labs/vitals). Needed for washout-window interaction checks
    # (e.g. FDA label Section 7.2: avoid anthracyclines for up to 7
    # months after stopping trastuzumab) that active_medications alone
    # can't answer, since that list only reflects medications right now.
    discontinued_medications: List[Dict[str, Any]] = field(default_factory=list)
    rxnorm_mapped_meds: List[Dict[str, Any]] = field(default_factory=list)
    cycles: List[CycleRecord] = field(default_factory=list)
    cardiac_assessments: List[CardiacAssessment] = field(default_factory=list)
    vitals: List[VitalSignRecord] = field(default_factory=list)
    labs: List[LabResult] = field(default_factory=list)
    # Institutional reference ranges, e.g. {"creatinine": {"lln": 0.5, "uln": 1.1}}
    # (hospital annex file). CTCAE v6.0 grades bilirubin, creatinine, ALT and
    # AST as multiples of these; defaults are used (and flagged) if absent.
    lab_reference_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Symptomatic heart failure status for CTCAE v6.0 "Heart failure" grading;
    # one of utils.ctcae_utils.HEART_FAILURE_STATUS_GRADES keys, or None.
    heart_failure_status: Optional[str] = None
    # Clinician-documented life-threatening FN (CTCAE v6.0 FN Grade 4).
    fn_life_threatening: bool = False

    @property
    def current_cycle_number(self) -> int:
        return self.cycles[-1].cycle_number if self.cycles else 1

    def get_latest_cardiac(self): return self.cardiac_assessments[-1] if self.cardiac_assessments else None
    def get_baseline_cardiac(self): return self.cardiac_assessments[0] if self.cardiac_assessments else None
    def get_current_cycle(self): return self.cycles[-1] if self.cycles else None
    def get_latest_labs(self): return self.labs[-1] if self.labs else None
    def get_baseline_labs(self): return self.labs[0] if self.labs else None
    def get_latest_vitals(self): return self.vitals[-1] if self.vitals else None

    def get_current_day(self) -> Optional[int]:
        """
        Best-estimate 'today' as a relative day, derived as the max day
        across every recorded data point (cycles, labs, vitals, cardiac
        assessments). Used for washout-window checks (e.g. how long ago
        was a medication stopped) where no single record is guaranteed
        to be the most recent one. Returns None if the patient has no
        dated records at all.
        """
        days = []
        if self.cycles:
            days.append(max(c.session_day for c in self.cycles))
        if self.labs:
            days.append(max(l.lab_day for l in self.labs))
        if self.vitals:
            days.append(max(v.visit_day for v in self.vitals))
        if self.cardiac_assessments:
            days.append(max(c.assessment_day for c in self.cardiac_assessments))
        return max(days) if days else None

    def compute_gls_relative_change(self) -> Optional[float]:
        baseline = self.get_baseline_cardiac()
        latest = self.get_latest_cardiac()
        if (not baseline or not latest or baseline.global_longitudinal_strain_pct is None
                or latest.global_longitudinal_strain_pct is None or baseline.global_longitudinal_strain_pct == 0):
            return None
        b_gls = abs(baseline.global_longitudinal_strain_pct)
        l_gls = abs(latest.global_longitudinal_strain_pct)
        return ((b_gls - l_gls) / b_gls) * 100.0
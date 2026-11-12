# Data-quality report: data/hospital_synth_v1

## 1. Files and columns

- A_demographics.csv: 300 rows; all requested columns present
- B_chemotherapy.csv: 1254 rows; all requested columns present
- C_her2_rt_courses.csv: 341 rows; all requested columns present
- D_vitals.csv: 3528 rows; all requested columns present
- E_cardiac.csv: 1196 rows; all requested columns present
- F_labs.csv: 3266 rows; all requested columns present
- G_medications.csv: 6222 rows; all requested columns present
- H_acute_care.csv: 58 rows; all requested columns present
- I_events.csv: 140 rows; all requested columns present
- annex_lab_reference_ranges.csv: 12 rows; all requested columns present

Patients: 300

## 2. ESSENTIAL field completeness (request: >= 70% of patients)

| File | Field | Patients with a value | OK |
|---|---|---|---|
| A_demographics.csv | age_at_first_dose | 100% | yes |
| A_demographics.csv | cancer_type | 100% | yes |
| A_demographics.csv | hypertension | 100% | yes |
| A_demographics.csv | known_drug_allergies | 100% | yes |
| A_demographics.csv | sex | 100% | yes |
| B_chemotherapy.csv | cumulative_doxorubicin_dose | 100% | yes |
| D_vitals.csv | record_time | 100% | yes |
| D_vitals.csv | temperature | 100% | yes |
| D_vitals.csv | weight | 100% | yes |
| E_cardiac.csv | lvef | 100% | yes |
| F_labs.csv | anc | 100% | yes |
| F_labs.csv | creatinine | 100% | yes |
| F_labs.csv | total_bilirubin | 100% | yes |

## 3. Conventions

- First doxorubicin session on Day 0: 300/300 patients
- Baseline echo on or before Day 0: 300/300
- Vital-sign times in HH:MM: all
- ANC magnitude: 0 of 2825 values < 30 (would indicate x10^9/L instead of cells/uL)
- Troponin results: 393; unit stated for 393; types hs-TnT 102, hs-TnI 260, TnI (conventional) 31

## 4. Inclusion criteria (request Section 3)

- Eligible: 300/300 (age >= 18, >= 1 doxorubicin cycle, baseline echo, >= 1 CBC)

## 5. Cohort profile and outcomes

- Age: mean 48.0; >= 60 y 18%
- Subtype: HR+/HER2- 47%, HER2+ 36%, triple-negative 14%, missing 3%
- Stage: II 45%, III 36%, IV 8%, missing 6%, I 5%
- Regimen: AC->Paclitaxel 168, AC->Paclitaxel (dose-dense) 74, AC 31, FAC 27
- Cumulative doxorubicin >= 250 mg/m2: 9%
- Anti-HER2 therapy after doxorubicin: 34%
- Patients with a neutropenic-fever encounter: 8.0% (literature: 10-20% for AC without G-CSF)
- LVEF fall >= 10 points to < 50% on follow-up: 6.7% (literature: 9% anthracycline overall, higher with trastuzumab)
- Documented events: Cardiotoxicity (heart failure or documented LV dysfunction) 91, febrile neutropenia 24, other 21, severe neutropenia 4

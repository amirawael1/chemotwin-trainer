"""
data/guideline_corpus.py
────────────────────────
Local guideline knowledge base for the Guideline Agent and the
evidence-alignment / hallucination check in safety_agent.py.

Each entry is a short PARAPHRASED summary of a source document, never
verbatim text. Order matters: for each gate, the FIRST tagged chunk is the
one cited on the recommended action (guideline_agent takes tagged_chunks[0]).

Sources held in data/guidelines/:
  - ASCO_2017_CardiacDysfunction_Armenian.pdf
  - ESC-ASCO_2023_Commentary_SparanoSahni.pdf   (secondary commentary on ESC 2022)
  - jeac106.pdf  = 2022 ESC Guidelines on cardio-oncology (Lyon et al.), Eur Heart J
    Cardiovasc Imaging 2022;23(10):e333-e465, co-published in Eur Heart J
    2022;43(41):4229-4361. Page numbers below (eNNN) refer to this version.
  - ESMO_2016_FebrileNeutropenia_Klastersky.pdf
  - CTCAE_v6_0_Final_Clean-Tracked-Mapping_w_OS_Jan2026.xlsx
Sources cited but NOT yet held locally (add the PDFs, then re-verify):
  - FDA Prescribing Information, Doxorubicin Hydrochloride Injection (rev. 3/2020)
  - KDIGO 2012 AKI guideline (used by Gate 6's >= 1.5 x baseline criterion)

REVISION NOTES (knowledge-layer fix, Oct 2026):
  * Added CTCAE v6.0 febrile-neutropenia and ESC 2022 / CTCAE v6.0 CTRCD
    definition chunks, so Gates 1, 2 and 4 cite the definitions they apply.
  * Gate 4: removed the claim that the > 15% GLS threshold is "not a formally
    adopted ESC recommendation". It IS part of the ESC 2022 CTRCD definition
    and of CTCAE v6.0 "Left ventricular dysfunction".
  * The ESMO 2020 (Curigliano) GLS 12%/5% chunk could not be verified against
    a held copy of that document, so it is moved to UNVERIFIED_CHUNKS and is
    no longer retrievable. Re-add it only after checking the PDF.
  * The ESC anti-HER2 interruption chunk is now labelled as anti-HER2-specific.
    It was the citation for holding DOXORUBICIN at Gate 2, which it does not
    cover.
  * Oct 2026, ESC guideline obtained (jeac106.pdf): added the anthracycline-
    specific CTRCD management chunk (Recommendation Table 24), now the primary
    citation for Gates 2 and 4; exact table/page citations added; the anti-HER2
    chunk now cites the guideline itself (Recommendation Table 25) instead of
    the commentary.
  * "Standard Doxorubicin Protocol" (not a citable source) replaced by
    ASCO 2017 Recommendations 2.2 and 4.3.
  * Added Gate 9 chunk (ASCO 2017 Rec. 1.1 risk factors).
  * Guideline disagreements are now declared explicitly in GUIDELINE_CONFLICTS
    (verified against the held sources) instead of being inferred from
    "more than one source tagged to a gate", which flagged complementary
    sources as conflicts.
"""

from typing import Any, Dict, List

GUIDELINE_CORPUS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------ Gate 1
    {
        "chunk_id": "febrile_neutropenia_01",
        "topic": "febrile_neutropenia",
        "gate_tags": ["gate_1_febrile_neutropenia_emergency"],
        "source": "ESMO Clinical Practice Guidelines — Management of Febrile Neutropenia",
        "citation": "Klastersky et al., ESMO Guidelines, Ann Oncol 2016;27(Suppl 5):v111-v118",
        "text": (
            "Febrile neutropenia is defined as an oral temperature above 38.3C, or two "
            "consecutive readings above 38.0C for two hours, together with an absolute "
            "neutrophil count below 0.5x10^9/L or expected to fall below that level. "
            "Patients assessed as high risk (MASCC score below 21) should be hospitalized "
            "and started on broad-spectrum intravenous antibacterial therapy without delay, "
            "given the high risk of bacterial sepsis; the first dose should be given within "
            "one hour of presentation, after blood cultures are drawn."
        ),
    },
    {
        "chunk_id": "ctcae_v6_febrile_neutropenia_01",
        "topic": "febrile_neutropenia",
        "gate_tags": ["gate_1_febrile_neutropenia_emergency"],
        "source": "NCI Common Terminology Criteria for Adverse Events v6.0",
        "citation": "NCI CTCAE v6.0, 'Febrile neutropenia' (MedDRA LLT 10016288) and 'Neutrophil count decreased' (LLT 10029366)",
        "text": (
            "CTCAE v6.0 grades febrile neutropenia as Grade 3 when the absolute neutrophil "
            "count is below 1000 per cubic millimetre together with either a single "
            "temperature above 38.3C or a temperature of at least 38.0C sustained for more "
            "than one hour; Grade 4 denotes life-threatening consequences needing urgent "
            "intervention. Neutrophil count decreased is graded 1 to 4 at below 1500, below "
            "1000, below 500 and below 100 cells per cubic millimetre respectively."
        ),
    },
    # ------------------------------------------------------------ Gates 2 and 4
    {
        "chunk_id": "esc_anthracycline_ctrcd_management_01",
        "topic": "cardiac_dysfunction",
        "gate_tags": ["gate_2_critical_cardiac_dysfunction", "gate_4_subclinical_cardiotoxicity"],
        "source": "2022 ESC Guidelines on cardio-oncology",
        "citation": "Lyon et al., 2022 ESC Guidelines on cardio-oncology, Eur Heart J Cardiovasc Imaging 2022;23(10):e333-e465 (co-published in Eur Heart J 2022;43(41):4229-4361), Recommendation Table 24 (p. e388) and Figure 25 (p. e387)",
        "text": (
            "During anthracycline chemotherapy, the ESC recommends temporarily interrupting the anthracycline "
            "and starting heart-failure therapy when moderate or severe asymptomatic CTRCD develops, temporarily "
            "interrupting it for moderate symptomatic CTRCD, and discontinuing it for severe symptomatic CTRCD, "
            "with a multidisciplinary decision on any restart. For mild asymptomatic CTRCD (LVEF of at least 50% "
            "with a significant GLS fall or a troponin or natriuretic peptide rise) the anthracycline should be "
            "continued with close monitoring, and ACE inhibitors, ARBs and/or beta-blockers should be considered. "
            "Mild symptomatic CTRCD calls for heart-failure therapy and a multidisciplinary decision on "
            "interruption versus continuation."
        ),
    },
    {
        "chunk_id": "esc_ctrcd_definition_01",
        "topic": "cardiac_dysfunction",
        "gate_tags": ["gate_4_subclinical_cardiotoxicity", "gate_2_critical_cardiac_dysfunction"],
        "source": "2022 ESC Guidelines on cardio-oncology (CTRCD definitions) / CTCAE v6.0",
        "citation": ("Lyon et al., 2022 ESC Guidelines on cardio-oncology, Eur Heart J Cardiovasc Imaging 2022;23(10):e333-e465 (co-published in Eur Heart J 2022;43(41):4229-4361), Table 3 (p. e347); adopted as NCI CTCAE v6.0 "
                     "'Left ventricular dysfunction' (LLT 10049694)"),
        "text": (
            "Asymptomatic cancer therapy-related cardiac dysfunction is mild when LVEF stays "
            "at or above 50% but GLS falls by more than 15% relative to baseline and/or "
            "cardiac biomarkers newly rise; moderate when LVEF falls by 10 or more "
            "percentage points to 40-49%, or by less than 10 points to 40-49% together with "
            "a GLS decline above 15% or a new biomarker rise; and severe when LVEF newly "
            "falls below 40%. Symptomatic CTRCD (heart failure) is graded mild to very "
            "severe by the treatment it requires."
        ),
    },
    {
        "chunk_id": "cardiac_dysfunction_withhold_antiher2_01",
        "topic": "cardiac_dysfunction",
        "gate_tags": ["gate_2_critical_cardiac_dysfunction"],
        "applies_to": "anti-HER2 therapy only",
        "source": "2022 ESC Guidelines on cardio-oncology",
        "citation": "Lyon et al., 2022 ESC Guidelines on cardio-oncology, Eur Heart J Cardiovasc Imaging 2022;23(10):e333-e465 (co-published in Eur Heart J 2022;43(41):4229-4361), Recommendation Table 25 (p. e390) and Figure 26",
        "text": (
            "During HER2-targeted therapy, the ESC recommends temporarily interrupting it for moderate or "
            "severe symptomatic CTRCD or severe asymptomatic CTRCD (LVEF below 40%), with a multidisciplinary "
            "approach for mild symptomatic CTRCD. For asymptomatic moderate CTRCD (LVEF 40-49%) HER2-targeted "
            "therapy should be continued with cardioprotective medication and frequent monitoring. This differs "
            "from anthracyclines, where moderate asymptomatic CTRCD warrants temporary interruption."
        ),
    },
    {
        "chunk_id": "subclinical_cardiotox_monitoring_01",
        "topic": "subclinical_cardiotoxicity",
        "gate_tags": ["gate_4_subclinical_cardiotoxicity"],
        "source": "ASCO Cardio-Oncology Clinical Practice Guideline (Armenian et al.)",
        "citation": "Armenian et al., ASCO Guidelines, J Clin Oncol 2017;35(8):893-911, Recommendation 4.2",
        "text": (
            "In patients receiving potentially cardiotoxic therapy, serum cardiac biomarkers "
            "or echocardiography-derived strain imaging (such as global longitudinal strain) "
            "may be used together with routine diagnostic imaging to help detect early or "
            "subclinical cardiac dysfunction. ASCO 2017 does not itself set a numeric GLS "
            "threshold and notes that no study has yet shown that acting on strain change "
            "alone improves clinical outcomes."
        ),
    },
    # ------------------------------------------------------------- Gates 8 and 9
    {
        "chunk_id": "cardiac_baseline_risk_01",
        "topic": "cardiac_baseline_risk",
        "gate_tags": ["gate_8_borderline_baseline_lvef_risk"],
        "source": "ASCO Cardio-Oncology Clinical Practice Guideline (Armenian et al.)",
        "citation": "Armenian et al., ASCO Guidelines, J Clin Oncol 2017;35(8):893-911, Recommendation 1.1",
        "text": (
            "Patients with cancer should be considered at increased risk for developing "
            "cardiac dysfunction if they have compromised cardiac function, including a "
            "borderline low left ventricular ejection fraction of 50% to 55%, at any time "
            "before or during treatment, particularly in combination with other risk "
            "factors such as older age (60 years or older) or multiple cardiovascular risk "
            "factors (hypertension, diabetes, dyslipidemia, obesity, smoking)."
        ),
    },
    {
        "chunk_id": "asco_increased_risk_criteria_01",
        "topic": "cardiac_baseline_risk",
        "gate_tags": ["gate_9_cumulative_risk_burden_advisory"],
        "source": "ASCO Cardio-Oncology Clinical Practice Guideline (Armenian et al.)",
        "citation": "Armenian et al., ASCO Guidelines, J Clin Oncol 2017;35(8):893-911, Recommendation 1.1",
        "text": (
            "Patients should be considered at increased risk of cardiac dysfunction if they "
            "receive high-dose anthracycline (doxorubicin 250 mg/m2 or more), or lower-dose "
            "anthracycline together with risk factors such as age 60 or older at treatment, "
            "two or more cardiovascular risk factors, or compromised cardiac function, or "
            "lower-dose anthracycline followed by trastuzumab."
        ),
    },
    # ------------------------------------------------- Gates 3, 5, 6, 7 (FDA)
    {
        "chunk_id": "hepatic_dosing_01",
        "topic": "hepatic_impairment",
        "gate_tags": ["gate_3_hepatic_impairment_hold"],
        "source": "Doxorubicin Hydrochloride Injection — FDA Prescribing Information",
        "citation": "FDA Prescribing Information for Doxorubicin Hydrochloride Injection, Revised 3/2020, Section 2.4 / Table 1, accessdata.fda.gov",
        "text": (
            "Doxorubicin dosing should be reduced by 50% for serum total bilirubin between "
            "1.2 and 3 mg/dL, and by 75% for serum total bilirubin between 3.1 and 5 mg/dL. "
            "Doxorubicin is contraindicated in patients with severe hepatic impairment, "
            "defined as Child-Pugh Class C or serum total bilirubin greater than 5 mg/dL; "
            "the drug should not be initiated, or should be discontinued, in this population "
            "due to decreased clearance and increased toxicity risk."
        ),
    },
    {
        "chunk_id": "renal_advisory_01",
        "topic": "renal_impairment",
        "gate_tags": ["gate_6_renal_impairment_advisory"],
        "source": "Doxorubicin Hydrochloride Injection — FDA Prescribing Information",
        "citation": "FDA Prescribing Information for Doxorubicin Hydrochloride Injection, Revised 3/2020, Section 12.3, accessdata.fda.gov",
        "text": (
            "Doxorubicin clearance is predominantly by metabolism and biliary excretion; "
            "approximately 40% of the dose appears in the bile within 5 days, while only 5% "
            "to 12% of the drug and its metabolites appear in the urine over the same period. "
            "The prescribing information gives no renal-based dosage adjustment for "
            "doxorubicin, consistent with its predominantly hepatic rather than renal "
            "clearance; co-administered renally-cleared medications and hydration status "
            "should still be reviewed in patients with elevated creatinine."
        ),
    },
    {
        "chunk_id": "allergy_contraindication_01",
        "topic": "allergy_contraindication",
        "gate_tags": ["gate_5_allergy_contraindication"],
        "source": "Doxorubicin Hydrochloride Injection — FDA Prescribing Information",
        "citation": "FDA Prescribing Information for Doxorubicin Hydrochloride Injection, Revised 3/2020, Section 4 (Contraindications), accessdata.fda.gov",
        "text": (
            "Severe hypersensitivity reaction to doxorubicin hydrochloride, including "
            "anaphylaxis, is a formal contraindication to further administration. The "
            "prescribing oncologist must be notified immediately and an alternative "
            "non-anthracycline regimen considered."
        ),
    },
    {
        "chunk_id": "drug_interaction_01",
        "topic": "drug_interaction",
        "gate_tags": ["gate_7_drug_interaction_alert"],
        "source": "Doxorubicin Hydrochloride Injection — FDA Prescribing Information",
        "citation": "FDA Prescribing Information for Doxorubicin Hydrochloride Injection, Revised 3/2020, Section 7.2, accessdata.fda.gov",
        "text": (
            "Concomitant use of trastuzumab and doxorubicin hydrochloride results in an "
            "increased risk of cardiac dysfunction; concomitant administration should be "
            "avoided. Patients receiving doxorubicin after stopping trastuzumab may also be "
            "at increased risk, since trastuzumab can persist in circulation for up to 7 "
            "months; anthracycline-based therapy should be avoided for up to 7 months after "
            "stopping trastuzumab where possible, and cardiac function should be monitored "
            "closely if an anthracycline is used sooner than that."
        ),
    },
    # ---------------------------------- G-CSF prophylaxis (Twin anticipatory advice)
    {
        "chunk_id": "esmo_gcsf_prophylaxis_01",
        "topic": "gcsf_prophylaxis",
        "gate_tags": [],
        "source": "ESMO Febrile Neutropaenia Clinical Practice Guidelines",
        "citation": "Klastersky et al., ESMO Guidelines, Ann Oncol 2016;27(Suppl 5):v111-v118 (primary and secondary G-CSF prophylaxis, Figure 1)",
        "text": (
            "Primary prophylaxis with G-CSF reduces the risk of febrile neutropenia by at least half. It is "
            "recommended when the regimen's febrile-neutropenia risk is 20% or more; at an intermediate risk "
            "of 10-20%, patient factors such as age over 65 and comorbidities should be weighed. Secondary "
            "prophylaxis, after a febrile episode in a previous cycle, is indicated when a dose reduction or "
            "delay is not desirable, for example with curative intent."
        ),
    },
    # ------------------------------------------------------- routine (no gate)
    {
        "chunk_id": "routine_monitoring_01",
        "topic": "routine_monitoring",
        "gate_tags": [],
        "source": "ASCO Cardio-Oncology Clinical Practice Guideline (Armenian et al.)",
        "citation": "Armenian et al., ASCO Guidelines, J Clin Oncol 2017;35(8):893-911, Recommendations 2.2 and 4.3",
        "text": (
            "Before potentially cardiotoxic therapy, clinicians should carry out a "
            "comprehensive assessment including history, examination, cardiovascular risk "
            "factor screening and an echocardiogram. During treatment, surveillance "
            "echocardiography may be offered to asymptomatic patients at increased risk, "
            "with frequency set by clinical judgement and patient circumstances."
        ),
    },
]

# Chunks whose content could not be checked against a held copy of the source.
# Deliberately EXCLUDED from GUIDELINE_CORPUS so they can never be retrieved
# or cited. Verify against the PDF before moving any back.
UNVERIFIED_CHUNKS: List[Dict[str, Any]] = [
    {
        "chunk_id": "subclinical_cardiotox_monitoring_02",
        "topic": "subclinical_cardiotoxicity",
        "gate_tags": ["gate_4_subclinical_cardiotoxicity"],
        "source": "ESMO Consensus Recommendations (Curigliano et al.)",
        "citation": "Curigliano et al., ESMO Consensus Recommendations, Ann Oncol 2020;31(2):171-190",
        "text": (
            "In asymptomatic patients undergoing treatment with cardiotoxic anticancer "
            "therapy who have normal left ventricular ejection fraction but a decline in "
            "average global longitudinal strain from baseline of 12% or more relative, or "
            "5% or more absolute, initiation of cardioprotective treatment (ACE inhibitor, "
            "ARB, and/or beta-blocker) should be considered. NOTE: this numeric threshold "
            "(12% relative / 5% absolute) is lower - i.e. more sensitive, triggering action "
            "sooner - than the 15% relative-decline figure this project's Safety Agent "
            "currently applies (see chunk subclinical_cardiotox_monitoring_01). The two "
            "guideline bodies genuinely disagree on this threshold; this project has chosen "
            "the higher, more specific ESC-aligned figure to reduce false-positive "
            "escalations, at the cost of somewhat later detection."
        ),
    },
]

# Explicit, source-verified disagreements between guideline bodies.
# Each entry was checked against the documents in data/guidelines/.
GUIDELINE_CONFLICTS: List[Dict[str, Any]] = [
    {
        "conflict_id": "fn_definition_anc_threshold",
        "kind": "definition",
        "kind_note": ("CTCAE grades adverse events for reporting; ESMO defines febrile neutropenia for "
                      "clinical management. Different operational definitions, not an error in either source."),
        "gate_tags": ["gate_1_febrile_neutropenia_emergency"],
        "positions": [
            {"source": "NCI CTCAE v6.0",
             "citation": "NCI CTCAE v6.0, 'Febrile neutropenia'",
             "position": "ANC < 1000/mm3 with a single temperature > 38.3C or >= 38.0C sustained > 1 h."},
            {"source": "ESMO 2016 Febrile Neutropenia Guideline",
             "citation": "Klastersky et al., Ann Oncol 2016;27(Suppl 5):v111-v118",
             "position": "ANC < 0.5 x 10^9/L (or expected to fall below it) with oral temperature > 38.3C "
                         "or two consecutive readings > 38.0C for 2 h."},
        ],
        "system_choice": ("Gate 1 applies the CTCAE v6.0 definition (more sensitive: also captures febrile "
                          "patients with ANC 500-999). Patients meeting the ESMO ANC criterion are labelled "
                          "and MASCC scoring is recommended to choose the management setting."),
        "verified_against": ["CTCAE_v6_0_Final_Clean-Tracked-Mapping_w_OS_Jan2026.xlsx",
                             "ESMO_2016_FebrileNeutropenia_Klastersky.pdf"],
    },
    {
        "conflict_id": "cardiac_surveillance_frequency_antiher2",
        "kind": "recommendation",
        "kind_note": "The two guidelines recommend different surveillance schedules.",
        "gate_tags": ["gate_7_drug_interaction_alert", "gate_8_borderline_baseline_lvef_risk",
                      "gate_9_cumulative_risk_burden_advisory"],
        "applies_when": "patient is receiving or about to receive anti-HER2 therapy",
        "requires_flag": "anti_her2_active",  # safety_output flag that must be True
        "positions": [
            {"source": "ASCO 2017",
             "citation": "Armenian et al., J Clin Oncol 2017;35(8):893-911, Recommendation 4.3",
             "position": "Surveillance imaging may be offered to increased-risk patients; frequency set by "
                         "clinician judgement."},
            {"source": "ESC 2022 (as summarised by Sparano & Sahni 2023, Table 1)",
             "citation": "Sparano & Sahni, JACC CardioOncology 2023;5(1):141-144",
             "position": "Echocardiography at baseline and every 3 months during anti-HER2 therapy for all "
                         "risk levels, plus cardiac biomarkers in higher-risk patients."},
        ],
        "system_choice": ("No single schedule is imposed; the surveillance recommendation shows both "
                          "positions and leaves the frequency to the clinician."),
        "verified_against": ["ASCO_2017_CardiacDysfunction_Armenian_pdf.pdf",
                             "ESC-ASCO_2023_Commentary_SparanoSahni_pdf.pdf"],
    },
]


def get_chunks_by_topic(topic: str) -> List[Dict[str, Any]]:
    return [c for c in GUIDELINE_CORPUS if c["topic"] == topic]


def get_chunks_by_gate(gate_name: str) -> List[Dict[str, Any]]:
    return [c for c in GUIDELINE_CORPUS if gate_name in c.get("gate_tags", [])]


def get_conflicts_by_gate(gate_name: str) -> List[Dict[str, Any]]:
    return [c for c in GUIDELINE_CONFLICTS if gate_name in c.get("gate_tags", [])]
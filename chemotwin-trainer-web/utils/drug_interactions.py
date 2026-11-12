"""
utils/drug_interactions.py
────────────────────────────
Local, curated drug-drug interaction checker for ChemoTwin.

Why this exists instead of calling an NLM API:
NLM discontinued the RxNav Drug-Drug Interaction API on January 2, 2024.
It has not been restored. RxNorm name/RxCUI lookup (rxnav_parser.py)
still works — only interaction checking is gone.

ChemoTwin's medication universe is small and known in advance (the
chemo agent itself, its usual cardioprotectants, the antibiotics used
for febrile-neutropenia intervention, and the cardiac drugs the
Intervention Agent may recommend). A small curated table is therefore
more reliable here than depending on a live third-party API, and is
easy to defend/cite in a thesis: every entry can be tied to a specific
guideline or drug-label statement, whereas an API's data provenance
(ONCHigh / DrugBank in NLM's case) was published but coarser-grained.

Extend INTERACTION_TABLE as your medication list grows. This is
intentionally NOT a general-purpose interaction database — it should
cover exactly the drugs relevant to this project's scope.
"""

from typing import Any, Dict, List, Tuple

# Each entry: (drug_a_keyword, drug_b_keyword, severity, description)
# Matching is done on lowercase substring against the raw medication
# strings in PatientState.active_medications.
INTERACTION_TABLE: List[Tuple[str, str, str, str]] = [
    (
        "doxorubicin",
        "trastuzumab",
        "HIGH",
        "Concurrent anthracycline + anti-HER2 therapy substantially increases "
        "cardiotoxicity risk; concurrent administration should be avoided, with sequencing per the specific regimen and prescribing guidance.",
    ),
    (
        "doxorubicin",
        "verapamil",
        "MODERATE",
        "Calcium channel blockers can increase anthracycline plasma levels and "
        "add to negative inotropic burden.",
    ),
    (
        "doxorubicin",
        "cyclosporine",
        "MODERATE",
        "Cyclosporine can increase doxorubicin exposure via P-glycoprotein inhibition.",
    ),
    (
        "warfarin",
        "fluconazole",
        "HIGH",
        "Azole antifungals (often used empirically in febrile neutropenia) "
        "significantly potentiate warfarin's anticoagulant effect.",
    ),
    (
        "warfarin",
        "vancomycin",
        "LOW",
        "Minor additive bleeding-risk monitoring advisory during concurrent antibiotic therapy.",
    ),
    (
        "beta-blocker",
        "verapamil",
        "HIGH",
        "Combined beta-blockade and non-dihydropyridine calcium channel blockade "
        "risks severe bradycardia/AV block.",
    ),
    (
        "ace inhibitor",
        "potassium",
        "MODERATE",
        "ACE inhibitors + potassium supplementation risk hyperkalemia; relevant if "
        "cardioprotective ACEi therapy is initiated per the Guideline Agent.",
    ),
]


def _matches(keyword: str, med_string: str) -> bool:
    return keyword.lower() in med_string.lower()


def check_interactions(active_medications: List[str]) -> List[Dict[str, Any]]:
    """
    Cross-checks a patient's active medication list against the local
    interaction table. Returns a list of triggered interaction dicts,
    each with the two matched medication strings, severity, and a
    plain-language description.
    """
    if not active_medications or len(active_medications) < 2:
        return []

    triggered: List[Dict[str, Any]] = []
    for i, med_a in enumerate(active_medications):
        for med_b in active_medications[i + 1:]:
            for kw_a, kw_b, severity, description in INTERACTION_TABLE:
                a_hits_a = _matches(kw_a, med_a)
                b_hits_b = _matches(kw_b, med_b)
                a_hits_b = _matches(kw_a, med_b)
                b_hits_a = _matches(kw_b, med_a)
                if (a_hits_a and b_hits_b) or (a_hits_b and b_hits_a):
                    triggered.append({
                        "medication_a": med_a,
                        "medication_b": med_b,
                        "severity": severity,
                        "description": description,
                    })
    return triggered


def check_allergy_contraindication(
    proposed_drug_keywords: List[str], known_allergies: List[str]
) -> List[Dict[str, str]]:
    """
    Checks a list of proposed intervention drug keywords (e.g. the
    drug classes the Intervention Agent is about to recommend) against
    the patient's documented allergy list. Broader than the
    doxorubicin-only check in safety_agent.py's Gate 5 — use this when
    validating a *specific* proposed action, not just the base chemo agent.
    """
    hits = []
    for allergy in known_allergies:
        for keyword in proposed_drug_keywords:
            if keyword.lower() in allergy.lower() or allergy.lower() in keyword.lower():
                hits.append({
                    "proposed_drug": keyword,
                    "documented_allergy": allergy,
                })
    return hits


# Trastuzumab persists in circulation for an extended period after the
# last dose. Per FDA Prescribing Information for Doxorubicin
# Hydrochloride Injection (Revised 3/2020), Section 7.2: anthracycline
# therapy should be avoided for up to 7 months after stopping
# trastuzumab where possible, and cardiac function should be monitored
# closely if used sooner. 7 months is approximated here as 210 days,
# consistent with the day-based conventions used elsewhere in
# PatientState (session_day, lab_day, etc.).
TRASTUZUMAB_WASHOUT_DAYS = 210


def check_trastuzumab_washout(
    active_medications: List[str],
    discontinued_medications: List[Dict[str, Any]],
    current_day: Any,
) -> List[Dict[str, Any]]:
    """
    Checks whether the patient is on (or starting) doxorubicin/an
    anthracycline within the trastuzumab washout window — i.e.
    trastuzumab was stopped recently enough that it may still be
    circulating, even though it no longer appears in
    active_medications. check_interactions() alone can't catch this,
    since it only compares medications that are BOTH currently active.

    Returns a list (empty if no washout risk), in the same shape as
    check_interactions()'s output, so callers can merge the two result
    lists without special-casing this check.
    """
    if current_day is None:
        return []

    on_anthracycline = any(
        "doxorubicin" in med.lower() or "anthracycline" in med.lower()
        for med in active_medications
    )
    if not on_anthracycline:
        return []

    hits: List[Dict[str, Any]] = []
    for entry in discontinued_medications:
        name = entry.get("name", "")
        stop_day = entry.get("stop_day")
        if "trastuzumab" not in name.lower() or stop_day is None:
            continue
        days_since_stopped = current_day - stop_day
        if 0 <= days_since_stopped < TRASTUZUMAB_WASHOUT_DAYS:
            hits.append({
                "medication_a": "Doxorubicin (active)",
                "medication_b": f"{name} (stopped {days_since_stopped} days ago)",
                "severity": "HIGH",
                "description": (
                    f"Trastuzumab was stopped {days_since_stopped} days ago, within the "
                    f"{TRASTUZUMAB_WASHOUT_DAYS}-day (~7 month) washout window during which "
                    f"it may still be circulating. FDA labeling recommends avoiding "
                    f"anthracycline therapy in this window where possible, and close cardiac "
                    f"monitoring if used sooner."
                ),
            })
    return hits
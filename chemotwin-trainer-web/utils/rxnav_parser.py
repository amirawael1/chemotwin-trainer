"""
utils/rxnav_parser.py
────────────────────────
RxNorm medication-name normalization for ChemoTwin (Category E).

NOTE on scope: this module only does name → RxCUI normalization via
NLM's RxNorm API, which is still live. It intentionally does NOT do
drug-drug interaction checking — NLM discontinued the RxNav
Drug-Drug Interaction API on January 2, 2024, and there is no
replacement NIH endpoint. For interaction checking, see
utils/drug_interactions.py, which uses a small curated local table
instead. This call is synchronous/blocking (a plain `requests.get`),
not asynchronous — call it at ingestion time (batch, offline), not
inside a live request path.
"""

import requests
from typing import Any, Dict, List, Optional

RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"


def normalize_medication_rxnorm(drug_name: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    """
    Queries NIH RxNav's REST API to map a local hospital medication
    string to an RxCUI (RxNorm Concept Unique Identifier).

    Returns a dict with keys: raw_input, rxcui (str or None), status
    ("MAPPED" | "UNMAPPED" | "ERROR").
    """
    try:
        response = requests.get(
            RXNAV_BASE_URL, params={"name": drug_name}, timeout=timeout_s
        )
        if response.status_code == 200:
            data = response.json()
            rxnorm_ids = data.get("idGroup", {}).get("rxnormId", [])
            if rxnorm_ids:
                return {
                    "raw_input": drug_name,
                    "rxcui": rxnorm_ids[0],
                    "status": "MAPPED",
                }
            return {"raw_input": drug_name, "rxcui": None, "status": "UNMAPPED"}
        return {"raw_input": drug_name, "rxcui": None, "status": "ERROR"}
    except requests.RequestException:
        return {"raw_input": drug_name, "rxcui": None, "status": "ERROR"}


def normalize_medication_list(drug_names: List[str]) -> List[Dict[str, Any]]:
    """
    Batch-normalizes a patient's active medication list. Intended to
    be run once at ingestion time when loading hospital data, with the
    results stored in PatientState.rxnorm_mapped_meds — not called
    repeatedly inside the agent pipeline.
    """
    return [normalize_medication_rxnorm(name) for name in drug_names]
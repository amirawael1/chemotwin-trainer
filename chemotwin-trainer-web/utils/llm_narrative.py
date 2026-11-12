"""
utils/llm_narrative.py
──────────────────────────
The actual Groq API call for ChemoTwin's clinical narrative synthesis.

Change log: this replaces two previously disconnected mechanisms:
  1. monitoring_agent.py's llm_client.generate_text()/callable(llm_client)
     duck-typing — dead code, since run_pipeline.py never passed an
     llm_client argument to run_monitoring_agent().
  2. test_llm_narration.py's standalone generate_clinical_narrative(),
     which worked from patient.to_summary_dict() alone rather than the
     Monitoring Agent's real risk predictions and SHAP feature
     attributions, and had a live bug (assigning to
     current_cycle_number, a read-only @property, would raise
     AttributeError if that script were actually run).

monitoring_agent.py's prompt-construction logic (which already
correctly pulls from real feature_attributions — attribution_source,
top drivers, percentages) is UNCHANGED and lives where it already was.
This module only owns the actual API call, so it's independently
testable/mockable and isn't duplicated between two files.
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed — fall back to whatever's already in
    # the real environment (os.getenv below still works for actual
    # shell-level env vars, just won't pick up a .env file).
    pass

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# Groq deprecated and fully decommissioned llama-3.3-70b-versatile on
# 2026-08-16 (announced 2026-06-17). Migrated to Groq's recommended
# replacement, openai/gpt-oss-120b. Configurable via env var so a
# future Groq deprecation doesn't require another code change.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

SYSTEM_PROMPT = (
    "You are an expert oncology clinical audit AI specializing in patient safety monitoring. "
    "Synthesize the provided patient monitoring data into a concise, professional, audit-ready "
    "clinical narrative.\n"
    "CRITICAL RULES:\n"
    "1. Use ONLY the risk percentages, categories, and contributing factors provided — never "
    "invent, estimate, or adjust a number yourself.\n"
    "2. When explaining WHY a risk is elevated or low, reference the named contributing factors "
    "exactly as given.\n"
    "3. If a prediction's source is the deterministic fallback rather than the ML ensemble, note "
    "in the narrative that this prediction used the rule-based fallback engine, so the reader "
    "knows it is not the trained ensemble's output.\n"
    "4. Do NOT print raw dictionary key names in the final output text — translate them into "
    "plain clinical language.\n"
    "5. Format all findings as clean, professional medical prose suitable for an EHR/audit record.\n"
    "6. Never state a CTCAE grade or an ESC CTRCD severity (mild, moderate, severe, subclinical) "
    "unless it appears in the AUTHORITATIVE CLINICAL GRADING section of the prompt, and copy it "
    "exactly. Do not use grades from memory: grading criteria change between CTCAE versions.\n"
    "7. A model's predicted risk (low/intermediate/high, or a percentage) is not a toxicity grade or "
    "a diagnosis; keep the two clearly separate."
)


def call_groq_narrative(prompt: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Sends a pre-built prompt (constructed by the caller from real
    monitoring output) to Groq and returns the generated narrative.

    Returns None — not an error string — if no API key is available
    or the groq package isn't installed, so callers can distinguish
    "skipped, no key" from "attempted and failed". Raises on an actual
    API failure so the caller decides how to handle/log it, rather
    than this function silently swallowing errors into narrative text.
    """
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not _GROQ_AVAILABLE:
        return None

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
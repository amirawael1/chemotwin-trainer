"""
utils/explainability.py
────────────────────────
Explainability module for ChemoTwin predictions.

Converts risk predictions (ML SHAP attributions or deterministic fallbacks) into 
structured explanation payloads, maintaining SHAP additivity properties and 
surfacing attribution source lineage.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Union

# Ensure project root is in sys.path even when running standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.toxicity_models import ToxicityRiskResult


def explain_prediction(result: Union[ToxicityRiskResult, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a structured explanation object with lineage tracking from a ToxicityRiskResult or dict.
    Handles both dict-based factors (ML SHAP) and string-based factors (deterministic fallbacks).
    """
    if isinstance(result, dict):
        toxicity_type = result.get("toxicity_type", "unknown_toxicity")
        risk_percentage = result.get("risk_percentage", 0.0)
        risk_category = result.get("risk_category", "UNKNOWN")
        top_contributing_factors = result.get("top_contributing_factors", [])
        missing_inputs = result.get("missing_inputs", [])
        prediction_confidence = result.get("prediction_confidence", 1.0)
        model_note = result.get("model_note", "")
        total_mag = result.get("total_model_magnitude", None)
    else:
        toxicity_type = getattr(result, "toxicity_type", "unknown_toxicity")
        risk_percentage = getattr(result, "risk_percentage", 0.0)
        risk_category = getattr(result, "risk_category", "UNKNOWN")
        top_contributing_factors = getattr(result, "top_contributing_factors", [])
        missing_inputs = getattr(result, "missing_inputs", [])
        prediction_confidence = getattr(result, "prediction_confidence", 1.0)
        model_note = getattr(result, "model_note", "")
        total_mag = getattr(result, "total_model_magnitude", None)

    # Determine attribution source lineage
    model_note_lower = model_note.lower()
    if "ensemble" in model_note_lower or "shap" in model_note_lower or "ml" in model_note_lower:
        attribution_source = "ML_SHAP"
    else:
        attribution_source = "DETERMINISTIC_FALLBACK"

    # Normalize string-based factors into standardized dicts if necessary
    normalized_factors = []
    for f in top_contributing_factors:
        if isinstance(f, dict):
            normalized_factors.append(f)
        elif isinstance(f, str):
            normalized_factors.append({
                "factor": f,
                "magnitude": 1.0,
                "direction": "HIGH",
                "description": f
            })

    # Calculate fallback total magnitude across normalized factors
    if total_mag is None or total_mag <= 0:
        total_mag = sum(abs(f.get("magnitude", 0.0)) for f in normalized_factors) or 1e-6

    attributions = []
    for f in normalized_factors:
        mag = f.get("magnitude", 0.0)
        share_pct = round((mag / total_mag) * 100, 2)
        sign = "+" if f.get("direction", "").upper() == "HIGH" or mag >= 0 else "-"
        
        factor_name = f.get("factor", "unknown_factor")
        attributions.append({
            "factor": factor_name,
            "feature": factor_name,
            "impact": f"{sign}{abs(share_pct):.1f}%",
            "magnitude": mag,
            "percentage_of_total_attribution": share_pct,
            "direction": f.get("direction", "NEUTRAL"),
            "description": f.get("description", "")
        })

    # Build plain-English summary sentence
    title_type = toxicity_type.replace('_', ' ').title()
    if len(attributions) >= 2:
        feat1 = attributions[0]['factor'].replace('_', ' ')
        feat2 = attributions[1]['factor'].replace('_', ' ')
        summary = (
            f"{title_type} risk is {risk_category} ({risk_percentage}%), "
            f"driven mainly by {feat1} and {feat2}."
        )
    elif len(attributions) == 1:
        feat1 = attributions[0]['factor'].replace('_', ' ')
        summary = (
            f"{title_type} risk is {risk_category} ({risk_percentage}%), "
            f"driven mainly by {feat1}."
        )
    else:
        summary = f"{title_type} risk is {risk_category} ({risk_percentage}%)."

    warning = None
    if missing_inputs:
        warning = (
            f"Prediction confidence reduced to {prediction_confidence} "
            f"due to missing input(s): {', '.join(missing_inputs)}."
        )

    return {
        "toxicity_type": toxicity_type,
        "risk_percentage": risk_percentage,
        "risk_category": risk_category,
        "attribution_source": attribution_source,
        "total_attribution_magnitude": round(float(total_mag), 4),
        "model_note": model_note,
        "explanation_summary": summary,
        "feature_attributions": attributions,
        "confidence": prediction_confidence,
        "data_completeness_warning": warning
    }


if __name__ == "__main__":
    print("Testing explainability.py standalone execution...")
    sample_result = ToxicityRiskResult(
        toxicity_type="cardiotoxicity",
        risk_score=0.59,
        risk_percentage=59.0,
        risk_category="HIGH",
        ctcae_grade="Grade 3",
        explanation_summary="",
        top_contributing_factors=[
            {"factor": "cumulative_dose", "direction": "HIGH", "magnitude": 0.18, "description": "High cumulative doxorubicin dose"},
            {"factor": "gls_relative_change", "direction": "HIGH", "magnitude": 0.10, "description": "Relative GLS reduction"}
        ],
        prediction_confidence=1.0,
        missing_inputs=[],
        model_note="ChemoTwin ML ensemble (RandomForest + GradientBoosting)",
        total_model_magnitude=0.56
    )
    explanation = explain_prediction(sample_result)
    print("\nExplanation output:")
    print(f"Attribution Source: {explanation['attribution_source']}")
    print(f"Total Model Magnitude: {explanation['total_attribution_magnitude']}")
    print(f"Summary: {explanation['explanation_summary']}")
    print(f"Top Factor 1 Share: {explanation['feature_attributions'][0]['percentage_of_total_attribution']}%")
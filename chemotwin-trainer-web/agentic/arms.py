"""
agentic/arms.py
───────────────
The arms of the pure-LLM vs agentic comparison. Every arm returns the same
dict shape so the runner and the scorer never special-case an arm.

  A_pure          single LLM call, record only                          (baseline)
  B_rag           single call + retrieved passages + CTCAE v6.0 text     (knowledge, no agency)
  C_tools         agent with tools, no verifier                          (tool-grounding only)
  D_full          tools + tool verifier + revision                       (proposed system)
  D_no_revise     verifier blocks but cannot send back for revision      (ablation: revision)
  D_llm_critic    tools + LLM critic instead of tool verifier            (ablation: verifier type)
  D_no_calc       agent without the grading/CTRCD/medication tools;
                  the verifier still has them                            (ablation: grounding)
  D_no_retrieval  agent without guideline search                         (ablation: retrieval)
  D_claims        D_full + LLM claim extraction in the verifier          (extension)
  C_preground     C_tools, but the core tools are run by the system before
                  the model's first turn                                 (extension: tool-use reliability)
  D_preground     D_full with the same pre-grounding                     (extension: tool-use reliability)
  A_fewshot, A_cot, A_role, A_selfcons
                  prompting-technique variants of A_pure (agentic/prompting.py)
D_llm_critic can use a different model as the critic (run_agentic_experiment.py --critic-provider/--critic-model):
that is the generator x judge matrix.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from agentic.orchestrator import ALL_TOOLS, AgenticConfig, run_agentic
from agentic.prompting import cot_llm, fewshot_llm, role_llm, selfcons_llm
from agentic.reasoning_agent import llm_with_rag, pure_llm
from agentic.render import render_patient


def _single(fn: Callable) -> Callable:
    def run(case: Dict[str, Any], model, data_origin: str = "synthetic", verifier_model=None) -> Dict[str, Any]:
        record = case.get("record_text") or render_patient(case["patient"], case.get("clinical_note"))
        r = fn(model, record, data_origin=data_origin)
        return {"status": "ERROR" if r.error else "RELEASED_UNVERIFIED", "released": not r.error,
                "assessment": r.assessment, "first_draft": r.assessment, "n_revisions": 0,
                "open_objections": [], "tool_trace": r.trace, "llm_calls": r.llm_calls,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "wall_time_s": round(r.latency_s, 2), "error": r.error, "record_text": record}
    return run


def _agentic(cfg: AgenticConfig) -> Callable:
    def run(case: Dict[str, Any], model, data_origin: str = "synthetic", verifier_model=None) -> Dict[str, Any]:
        return run_agentic(case["patient"], model, cfg, clinical_note=case.get("clinical_note"),
                           data_origin=data_origin, record_text=case.get("record_text"),
                           verifier_model=verifier_model)
    return run


ARMS: Dict[str, Callable] = {
    "A_pure": _single(pure_llm),
    "B_rag": _single(llm_with_rag),
    "C_tools": _agentic(AgenticConfig(name="C_tools", verifier=None)),
    "D_full": _agentic(AgenticConfig(name="D_full")),
    "D_no_revise": _agentic(AgenticConfig(name="D_no_revise", revise=False)),
    "D_llm_critic": _agentic(AgenticConfig(name="D_llm_critic", verifier="llm_critic")),
    "D_no_calc": _agentic(AgenticConfig(name="D_no_calc",
                                        tools=["search_guidelines", "data_completeness", "dose_and_organ_function"])),
    "D_no_retrieval": _agentic(AgenticConfig(name="D_no_retrieval",
                                             tools=[t for t in ALL_TOOLS if t != "search_guidelines"])),
    "D_claims": _agentic(AgenticConfig(name="D_claims", verifier_llm_extraction=True)),
    "C_preground": _agentic(AgenticConfig(name="C_preground", verifier=None, preground=True)),
    "D_preground": _agentic(AgenticConfig(name="D_preground", preground=True)),
    # Post-hoc improvement after the layer A error (LA-12992047): facts written only in words
    # ("febrile" without a value) are read by the tools and enforced by the verifier.
    "D_text": _agentic(AgenticConfig(name="D_text", text_findings=True)),
    # Prompting techniques (supervisor request): can a better prompt alone make the pure LLM safe?
    "A_fewshot": _single(fewshot_llm),
    "A_cot": _single(cot_llm),
    "A_role": _single(role_llm),
    "A_selfcons": _single(selfcons_llm),
}
MAIN_ARMS = ["A_pure", "B_rag", "C_tools", "D_full"]
ABLATION_ARMS = ["D_no_revise", "D_llm_critic", "D_no_calc", "D_no_retrieval"]
PREGROUND_ARMS = ["C_preground", "D_preground"]
PROMPT_ARMS = ["A_pure", "A_fewshot", "A_cot", "A_role", "A_selfcons"]
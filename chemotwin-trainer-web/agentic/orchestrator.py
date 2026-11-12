"""
agentic/orchestrator.py
───────────────────────
Ground → Verify → Revise.

  1. GROUND   the Clinical Reasoning Agent calls tools and drafts an assessment
  2. VERIFY   the verifier checks every claim and returns objections
  3. REVISE   hard objections go back to the SAME agent conversation (it keeps
              its tool results and may call tools again); repeat up to
              `max_revisions` times
  4. RELEASE  no hard objection left -> RELEASED (soft ones shown as warnings);
              only non-critical hard objections left -> RELEASED_WITH_WARNINGS
     ESCALATE hard objections remain  -> WITHHELD, sent to the clinician with
              the open objections; never shown as a recommendation

Every configuration used in the experiment is an `AgenticConfig`, so an
ablation is a one-line change (remove a tool, swap the verifier, switch off
revision) and the agent code itself never changes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agentic.reasoning_agent import AgentRun, ClinicalReasoningAgent
from agentic.render import render_patient
from agentic.tools import ToolBox
from agentic.verifier import LLMCritic, Objection, ToolVerifier

ALL_TOOLS = ["ctcae_grade", "esc_ctrcd", "check_medication", "dose_and_organ_function",
             "data_completeness", "search_guidelines"]


@dataclass
class AgenticConfig:
    name: str = "D_full"
    tools: List[str] = field(default_factory=lambda: list(ALL_TOOLS))
    verifier: Optional[str] = "tools"        # "tools" | "llm_critic" | None
    verifier_llm_extraction: bool = False    # LLM extracts claims from free text before tool checks
    revise: bool = True
    max_revisions: int = 2
    max_steps: int = 5
    preground: bool = False                  # system runs the core tools before the model's first turn
    text_findings: bool = False              # tools also read facts stated only in words (D_text)


def _objection_message(obs: List[Objection]) -> str:
    lines = ["VERIFIER OBJECTIONS — your assessment was checked against the record and the standards."]
    for i, o in enumerate(obs, 1):
        ev = ""
        if o.evidence.get("tool"):
            ev = " Evidence: " + json.dumps(o.evidence["tool"], default=str)[:600]
        lines.append(f"{i}. [{o.severity.upper()} {o.code}] {o.message}{ev}")
    lines.append("Fix every HARD objection (you may call tools again). Address SOFT ones where appropriate. "
                 'Then reply with {"final": <complete revised assessment>}.')
    return "\n".join(lines)


def run_agentic(patient: Any, model, config: Optional[AgenticConfig] = None, clinical_note: Optional[str] = None,
                data_origin: str = "synthetic", verifier_model=None,
                record_text: Optional[str] = None) -> Dict[str, Any]:
    config = config or AgenticConfig()
    t0 = time.time()
    record = record_text or render_patient(patient, clinical_note)
    text = record if config.text_findings else None
    tb = ToolBox(patient, enabled=config.tools, record_text=text)
    verify_tb = ToolBox(patient, record_text=text)  # the verifier always has every tool, even when the agent's are ablated
    agent = ClinicalReasoningAgent(model, max_steps=config.max_steps, data_origin=data_origin)
    run: AgentRun = agent.run(record, tb, preground=config.preground)
    rounds: List[Dict[str, Any]] = [{"round": 0, "assessment": run.assessment}]

    verifier = None
    if config.verifier == "tools":
        verifier = ToolVerifier(model=verifier_model or model, extract_with_llm=config.verifier_llm_extraction,
                                data_origin=data_origin, strict_not_evaluable=config.text_findings)
    elif config.verifier == "llm_critic":
        verifier = LLMCritic(verifier_model or model, data_origin=data_origin)

    status, objections = "RELEASED_UNVERIFIED", []
    if verifier is not None and not run.error:
        for r in range(config.max_revisions + 1):
            objections = verifier.verify(run.assessment, verify_tb, record)
            rounds[-1]["objections"] = [o.to_dict() for o in objections]
            hard = [o for o in objections if o.severity == "hard"]
            if not hard:
                status = "RELEASED"
                break
            if not config.revise or r == config.max_revisions:
                status = ("WITHHELD_ESCALATED" if any(o.critical for o in hard)
                          else "RELEASED_WITH_WARNINGS")
                break
            agent.messages.append({"role": "user", "content": _objection_message(objections)})
            before = run.assessment
            run.assessment = {}
            run = agent.run(record, tb, messages=agent.messages, run=run)
            if run.error:
                run.assessment = run.assessment or before
                status = "WITHHELD_ESCALATED"
                break
            rounds.append({"round": r + 1, "assessment": run.assessment})
    if run.error:
        status = "ERROR"

    vusage = getattr(verifier, "usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    return {
        "config": asdict(config),
        "status": status,
        "released": status in ("RELEASED", "RELEASED_UNVERIFIED", "RELEASED_WITH_WARNINGS"),
        "assessment": run.assessment,
        "first_draft": rounds[0]["assessment"],
        "rounds": rounds,
        "open_objections": [o.to_dict() for o in objections],
        "n_revisions": len(rounds) - 1,
        "tool_trace": run.trace,
        "llm_calls": run.llm_calls + vusage["calls"],
        "prompt_tokens": run.prompt_tokens + vusage["prompt_tokens"],
        "completion_tokens": run.completion_tokens + vusage["completion_tokens"],
        "wall_time_s": round(time.time() - t0, 2),
        "error": run.error,
        "record_text": record,
    }
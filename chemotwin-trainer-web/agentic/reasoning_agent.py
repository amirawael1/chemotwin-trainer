"""
agentic/reasoning_agent.py
──────────────────────────
Clinical Reasoning Agent: an LLM that decides which clinical tools to call,
reads their results and writes the structured assessment.

Tool protocol (model-agnostic, works with MedGemma in LM Studio as well as
gpt-oss-120b in the cloud). Each turn the model replies with ONE JSON object:

  {"thought": "...", "tool_calls": [{"tool": "ctcae_grade", "args": {"term": "all"}}, ...]}
  {"thought": "...", "final": { ...assessment... }}

Several tool calls per turn are allowed so a cloud run with a daily token
budget needs few round trips. The run stops after `max_steps` turns; if the
model has not answered by then it is asked once for its final assessment.

Also defines the two NON-agentic baselines, which use the same schema:
  pure_llm()      Arm A — the record only; the model grades from memory.
  llm_with_rag()  Arm B — the record plus retrieved guideline passages and
                  the official CTCAE v6.0 criteria text, but no tools.
Arm B answers "is it just missing knowledge?": if B still makes the errors
A makes, knowledge in the prompt is not enough and the agentic mechanisms
(tool-grounding, verification, revision) are what close the gap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentic.assessment import ASSESSMENT_SCHEMA, normalise
from agentic.llm import extract_json
from agentic.tools import CTCAE_TERMS, ToolBox, bm25_search, ctcae_terms

BASE_RULES = """Clinical rules you must follow:
- Grade toxicities with NCI CTCAE v6.0 (released 2025). Do not use CTCAE v5.0 thresholds or terms.
- Classify cardiac dysfunction with the ESC 2022 cardio-oncology definitions of CTRCD.
- Only state values that appear in the record. If an input needed for a grade is missing, use "not_evaluable"
  for that grade and list what is needed in data_requests.
- If safety-critical data are missing (current temperature, ANC or LVEF, or ANC and temperature from different
  days), choose "defer_pending_data" rather than guessing.
- Never propose a drug that matches a documented allergy or its drug class.
- Every action must cite its source.
Report all eight CTCAE terms: Neutrophil count decreased, Febrile neutropenia, Left ventricular dysfunction,
Heart failure, Creatinine increased, Blood bilirubin increased, Alanine aminotransferase increased,
Aspartate aminotransferase increased (grade 0 when the criteria are not met)."""

AGENT_SYSTEM = f"""You are the Clinical Reasoning Agent of a clinical decision-support system for doxorubicin
chemotherapy toxicity. You assist the treating oncology team; you never act on the patient yourself.

You have tools that read the patient record and apply the official standards. Use them: do not grade
toxicities, classify CTRCD or judge drug safety from memory when a tool can do it. Tool results are
authoritative; cite their evidence_id (for grades) and chunk_id (for guideline statements).

{BASE_RULES}

TOOLS
{{tools}}

PROTOCOL — reply with exactly ONE JSON object per turn and nothing else:
  to call tools:  {{"thought": "<short reasoning>", "tool_calls": [{{"tool": "<name>", "args": {{...}}}}]}}
  to finish:      {{"thought": "<short reasoning>", "final": <assessment>}}
You may request several tools in one turn. Before finishing, check every drug you propose with check_medication.

The final assessment must follow this schema:
{ASSESSMENT_SCHEMA}"""

PURE_SYSTEM = f"""You are an oncology clinical decision-support assistant for doxorubicin chemotherapy
toxicity. Assess the patient record and answer with ONE JSON object only, following this schema:
{ASSESSMENT_SCHEMA}

{BASE_RULES}"""


@dataclass
class AgentRun:
    assessment: Dict[str, Any]
    raw_final: Optional[Dict[str, Any]]
    trace: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None

    def add_usage(self, res) -> None:
        self.llm_calls += 1
        self.prompt_tokens += res.prompt_tokens or 0
        self.completion_tokens += res.completion_tokens or 0
        self.latency_s += res.latency_s or 0.0


def _clip(obj: Any, limit: int = 3500) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


class ClinicalReasoningAgent:
    def __init__(self, model, max_steps: int = 5, data_origin: str = "synthetic"):
        self.model = model
        self.max_steps = max_steps
        self.data_origin = data_origin

    PREGROUND_TOOLS = (("ctcae_grade", {"term": "all"}), ("esc_ctrcd", {}),
                       ("dose_and_organ_function", {}), ("data_completeness", {}))

    def run(self, record_text: str, toolbox: ToolBox,
            messages: Optional[List[Dict[str, str]]] = None, run: Optional[AgentRun] = None,
            preground: bool = False) -> AgentRun:
        """Fresh run when `messages` is None; otherwise continues an existing
        conversation (used by the revise step so the agent keeps its tool
        results and only fixes what the verifier objected to)."""
        run = run or AgentRun(assessment={}, raw_final=None)
        if messages is None:
            messages = [{"role": "system", "content": AGENT_SYSTEM.replace("{tools}", toolbox.describe())},
                        {"role": "user", "content": "PATIENT RECORD\n" + record_text +
                         "\n\nAssess this patient. Start by calling the tools you need."}]
            if preground:
                # Pre-grounding: the record-bound core tools are run by the system before the
                # model's first turn, so grounding no longer depends on the model choosing to
                # call them. The agent can still call any tool (e.g. check_medication) itself.
                results = []
                for name, args in self.PREGROUND_TOOLS:
                    if name not in toolbox.registry:
                        continue
                    out = toolbox.call(name, args)
                    run.trace.append({"step": -1, "type": "tool", "auto": True, "tool": name, "args": args,
                                      "evidence_id": out.get("evidence_id"), "error": out.get("error")})
                    results.append({"tool": name, "args": args, "result": out})
                if results:
                    messages[1]["content"] = ("PATIENT RECORD\n" + record_text +
                                              "\n\nThe core tools have already been run on this record; their "
                                              "results are below. Use them (cite their evidence_id), call further "
                                              "tools if needed (e.g. check_medication before proposing a drug), "
                                              "then give the final assessment.\n\nTOOL RESULTS\n"
                                              + _clip(results, 16000))
        self.messages = messages
        for step in range(self.max_steps + 1):
            if step == self.max_steps:
                messages.append({"role": "user", "content": "Tool budget used. Reply now with "
                                 '{"final": <assessment>} using the results you already have.'})
            res = self.model.chat(messages, data_origin=self.data_origin)
            run.add_usage(res)
            if res.text is None:
                run.error = res.error
                break
            messages.append({"role": "assistant", "content": res.text})
            # what the model wrote this turn (for diagnosing runs that never reach a final answer)
            run.trace.append({"step": step, "type": "turn", "text": res.text[:800],
                              "finish_reason": getattr(res, "finish_reason", None),
                              "completion_tokens": res.completion_tokens})
            obj = extract_json(res.text)
            if obj is None:
                run.trace.append({"step": step, "type": "unparseable", "text": res.text[:400]})
                messages.append({"role": "user", "content": "Your reply was not a single JSON object. "
                                 "Reply again with exactly one JSON object following the protocol."})
                continue
            if isinstance(obj.get("final"), dict) or ("toxicities" in obj and "tool_calls" not in obj):
                used_tools = any(t.get("type") == "tool" for t in run.trace)
                reminded = any(t.get("type") == "grounding_reminder" for t in run.trace)
                if not used_tools and not reminded and step < self.max_steps and toolbox.registry:
                    # Grounding rule: an answer written before any tool call is sent back once.
                    run.trace.append({"step": step, "type": "grounding_reminder"})
                    messages.append({"role": "user", "content":
                                     "You answered without calling any tool. Grades, CTRCD category and drug safety "
                                     "must come from the tools. Call the tools you need now, then give the final "
                                     "assessment."})
                    continue
                final = obj["final"] if isinstance(obj.get("final"), dict) else obj
                run.raw_final = final
                run.assessment = normalise(final)
                run.trace.append({"step": step, "type": "final", "thought": obj.get("thought", "")})
                break
            calls = obj.get("tool_calls")
            if calls is None and obj.get("tool"):
                calls = [{"tool": obj["tool"], "args": obj.get("args", {})}]
            if not calls:
                messages.append({"role": "user", "content": "Call a tool or give the final assessment."})
                continue
            results = []
            for c in calls[:8]:
                name, args = c.get("tool"), c.get("args") or {}
                out = toolbox.call(name, args)
                run.trace.append({"step": step, "type": "tool", "tool": name, "args": args,
                                  "evidence_id": out.get("evidence_id"), "error": out.get("error")})
                results.append({"tool": name, "args": args, "result": out})
            messages.append({"role": "user", "content": "TOOL RESULTS\n" + _clip(results, 16000)})
        if not run.assessment:
            run.assessment = normalise(None)
        return run


# ===========================================================================
# Non-agentic baselines
# ===========================================================================
def _single_shot(model, system: str, user: str, data_origin: str) -> AgentRun:
    """One answer. If the reply is not valid JSON the model gets ONE format
    reminder (the agent arms get the same courtesy), so a formatting slip is
    not scored as a clinical error. Persistent failures remain FORMAT_FAILURE."""
    run = AgentRun(assessment={}, raw_final=None)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        res = model.chat(messages, data_origin=data_origin)
        run.add_usage(res)
        if res.text is None:
            run.error = res.error
            run.assessment = normalise(None)
            return run
        obj = extract_json(res.text)
        run.trace.append({"type": "single_shot", "attempt": attempt, "text": res.text[:4000],
                          "parsed": obj is not None})
        if obj is not None and ("toxicities" in obj or isinstance(obj.get("final"), dict)):
            break
        messages += [{"role": "assistant", "content": res.text or ""},
                     {"role": "user", "content": "Your reply was not one complete JSON object in the required "
                                                 "schema. Reply again with only that JSON object."}]
    run.raw_final = obj
    run.assessment = normalise(obj)
    return run


def pure_llm(model, record_text: str, data_origin: str = "synthetic") -> AgentRun:
    return _single_shot(model, PURE_SYSTEM, "PATIENT RECORD\n" + record_text, data_origin)


def rag_context(record_text: str, k: int = 6) -> str:
    """Arm B context: retrieved guideline passages + official CTCAE v6.0
    criteria text for the eight graded terms. Retrieval uses the record text
    itself as the query (no tool choice, no reasoning about what to look up)."""
    passages = bm25_search(record_text + " febrile neutropenia cardiac dysfunction CTRCD anthracycline", k)
    lines = ["GUIDELINE PASSAGES (retrieved):"]
    for p in passages:
        lines.append(f"[{p['chunk_id']}] {p['source']} ({p['year']}): {p['text']}")
    lines.append("\nNCI CTCAE v6.0 CRITERIA (official text):")
    terms = ctcae_terms().get("terms", {})
    for official, _ in CTCAE_TERMS.values():
        t = terms.get(official, {})
        grades = "; ".join(f"G{g}: {t.get(f'grade_{g}')}" for g in range(1, 5) if t.get(f"grade_{g}"))
        lines.append(f"- {official}: {grades}")
    return "\n".join(lines)


def llm_with_rag(model, record_text: str, data_origin: str = "synthetic") -> AgentRun:
    user = (rag_context(record_text) + "\n\nPATIENT RECORD\n" + record_text +
            "\n\nUse the passages and criteria above; cite passages by their [chunk_id].")
    return _single_shot(model, PURE_SYSTEM, user, data_origin)
"""
agentic/prompting.py
────────────────────
Prompting-technique variants of the pure-LLM arm (supervisor request, Nov 2026).
Question: can a better prompt alone make a language model safe, without tools or a checker?

  A_pure       zero-shot (the existing baseline; unchanged)
  A_fewshot    zero-shot prompt + 2 worked examples with correct answers
  A_cot        asks for step-by-step reasoning in a "reasoning" field before the answer
  A_role       expert persona (consultant oncologist / cardio-oncologist)
  A_selfcons   self-consistency: 3 answers at temperature 0.7, majority vote per field

All variants use the same rules (BASE_RULES), the same output schema and the same scorer as A_pure.
Few-shot examples come from the hospital-format cohort and are built from the answer key itself, so they
are correct by the specification; they must not be scored on the same cases (FEWSHOT_IDS).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List

from agentic.assessment import ASSESSMENT_SCHEMA, TERMS_REQUIRED, normalise
from agentic.reasoning_agent import BASE_RULES, PURE_SYSTEM, AgentRun
from agentic.llm import extract_json, last_json_with

# Reasoning models (e.g. Nemotron Ultra) think silently before answering. With the longer prompts of these
# variants, the hidden reasoning can use up the whole output budget and leave an EMPTY visible answer (scored as
# FORMAT_FAILURE although nothing clinical went wrong). Three measures, all recorded in the trace:
#   1. the prompting variants get 8,192 output tokens (A_pure keeps 4,096);
#   2. an empty answer is retried once with 16,384 tokens;
#   3. if the visible answer is still empty but the model DID write a complete answer object inside its
#      reasoning, that object is used and flagged "answer_from_reasoning" (reported separately in the thesis).
# Anything else remains FORMAT_FAILURE.
PROMPT_MAX_TOKENS = 8192
GLITCH_TOKENS = 50      # a "finished" reply this short with no visible text is a broken service reply
GLITCH_RESENDS = 3      # resend the same request up to 3 times (the last one without JSON mode)
GLITCH_PAUSE_S = 15
_REMINDER = ("Your reply was not one complete JSON object in the required schema. "
             "Reply again with only that JSON object.")


def _valid(obj) -> bool:
    return isinstance(obj, dict) and ("toxicities" in obj or isinstance(obj.get("final"), dict))


def _ask(model, system: str, user: str, data_origin: str, run: AgentRun, budget_retry: bool):
    """Up to two attempts (answer + one format reminder). Returns the parsed object or None."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    obj = None
    for attempt in range(2):
        res, glitches = model.chat(messages, data_origin=data_origin), 0
        run.add_usage(res)
        # Broken service reply: the server says it finished, but returned (almost) nothing, e.g. one token
        # or a few random characters. Not the model's clinical answer, so the same request is sent again.
        while (res.text is not None and not res.text.strip() and (res.completion_tokens or 0) < GLITCH_TOKENS
               and getattr(res, "finish_reason", None) == "stop" and glitches < GLITCH_RESENDS):
            glitches += 1
            time.sleep(GLITCH_PAUSE_S)
            plain = glitches == GLITCH_RESENDS          # last resend: plain text instead of JSON mode
            res = model.chat(messages, data_origin=data_origin, json_mode=False) if plain \
                else model.chat(messages, data_origin=data_origin)
            run.add_usage(res)
        if res.text is None:
            run.error = res.error
            return None
        obj = extract_json(res.text)
        hidden = getattr(res, "reasoning_text", "") or ""
        step = {"type": "single_shot", "attempt": attempt, "text": res.text[:4000], "parsed": obj is not None,
                "finish_reason": getattr(res, "finish_reason", None),
                "completion_tokens": res.completion_tokens, "reasoning_chars": len(hidden),
                "max_tokens": getattr(model, "max_tokens", None)}
        if budget_retry:
            step["budget_retry"] = True
        if glitches:
            step["service_resends"] = glitches
        if not _valid(obj) and not res.text.strip():
            rescued = last_json_with(hidden, "toxicities")
            if _valid(rescued):
                obj = rescued
                step["answer_from_reasoning"] = True
                step["parsed"] = True
            else:
                step["reasoning_tail"] = hidden[-1500:]
        run.trace.append(step)
        if _valid(obj):
            return obj
        messages += [{"role": "assistant", "content": res.text or ""}, {"role": "user", "content": _REMINDER}]
    return obj


def _single_shot(model, system: str, user: str, data_origin: str) -> AgentRun:
    run = AgentRun(assessment={}, raw_final=None)
    old = getattr(model, "max_tokens", None)
    try:
        if old is not None:
            model.max_tokens = max(old, PROMPT_MAX_TOKENS)
        obj = _ask(model, system, user, data_origin, run, budget_retry=False)
        empty = not _valid(obj) and run.trace and all(not t.get("text") for t in run.trace)
        if empty and old is not None and not run.error:
            model.max_tokens = 2 * max(old, PROMPT_MAX_TOKENS)
            obj = _ask(model, system, user, data_origin, run, budget_retry=True)
        if run.error:
            run.assessment = normalise(None)
            return run
        run.raw_final = obj
        run.assessment = normalise(obj)
        return run
    finally:
        if old is not None:
            model.max_tokens = old

FEWSHOT_IDS = ("PT-0037@d21:pre_cycle", None)   # second example chosen automatically (an abnormal case)

COT_SYSTEM = PURE_SYSTEM + """

Think step by step before answering. Put your reasoning FIRST, in a field "reasoning" of the same JSON object:
1. list every relevant value from the record with its day;
2. for each CTCAE v6.0 term, name the threshold the value meets;
3. classify CTRCD with the ESC 2022 definitions;
4. check for missing safety-critical data, allergies and interactions;
5. decide, choose the urgency level and the actions.
Then fill in the remaining fields."""

ROLE_SYSTEM = f"""You are a consultant medical oncologist with a special interest in cardio-oncology and twenty
years of experience managing anthracycline and anti-HER2 toxicity. You are preparing your recommendation for the
multidisciplinary tumour board, carefully and according to the current guidelines.
Answer with ONE JSON object only, following this schema:
{ASSESSMENT_SCHEMA}

{BASE_RULES}"""


def _example_answer(case: Dict[str, Any]) -> Dict[str, Any]:
    """A correct assessment built from the answer key (specification module)."""
    from agentic.scoring import reference_for
    ref = reference_for(case)
    if ref["defer_required"]:
        dec = "defer_pending_data"
    elif ref["hold_required"]:
        dec = "hold"
    else:
        dec = "continue"
    actions = [{"action": d, "level": ref["min_level"], "citations": [d.split("(")[-1].rstrip(")")] if "(" in d
                else ["NCI CTCAE v6.0"]} for _, d in ref["required_actions"]]
    if not actions:
        actions = [{"action": "Proceed as planned with routine monitoring", "level": ref["min_level"],
                    "citations": ["NCI CTCAE v6.0"]}]
    return {"toxicities": [{"term": t, "grade": ref["grades"][t], "evidence": "NCI CTCAE v6.0"} for t in TERMS_REQUIRED],
            "ctrcd": {"asymptomatic": ref["esc"].get("asymptomatic_ctrcd") or "none",
                      "symptomatic": ref["esc"].get("symptomatic_ctrcd") or "not_documented"},
            "chemotherapy_decision": dec, "urgency_level": ref["min_level"], "actions": actions,
            "medications_proposed": [], "data_requests": list(ref["completeness"]["critical_missing"]),
            "guideline_disagreements": [], "summary": "Assessment according to CTCAE v6.0 and ESC 2022."}


@lru_cache(maxsize=1)
def fewshot_examples() -> List[Dict[str, Any]]:
    from agentic.render import render_patient
    from agentic.scoring import reference_for
    from agentic.testsets import TEST_SETS
    cases = TEST_SETS["hosp_synth"]()
    by = {c["case_id"]: c for c in cases}
    chosen = [by[FEWSHOT_IDS[0]]] if FEWSHOT_IDS[0] in by else []
    for c in cases:   # an acute case with a hold indication, for contrast
        if c["test_set"] == "hs_acute" and c not in chosen:
            ref = reference_for(c)
            if ref["hold_required"] and not ref["defer_required"] and ref["required_actions"]:
                chosen.append(c)
                break
    out = []
    for c in chosen[:2]:
        rec = c.get("record_text") or render_patient(c["patient"], c.get("clinical_note"))
        out.append({"case_id": c["case_id"], "record": rec, "answer": _example_answer(c)})
    return out


def _fewshot_messages_prefix() -> str:
    parts = ["Two worked examples follow, each a patient record and a correct answer.\n"]
    for i, ex in enumerate(fewshot_examples(), 1):
        parts.append(f"EXAMPLE {i} - PATIENT RECORD\n{ex['record']}\n\nEXAMPLE {i} - CORRECT ANSWER\n"
                     f"{json.dumps(ex['answer'], ensure_ascii=False)}\n")
    parts.append("Now assess the next patient in the same way.\n")
    return "\n".join(parts)


def fewshot_llm(model, record_text: str, data_origin: str = "synthetic") -> AgentRun:
    return _single_shot(model, PURE_SYSTEM, _fewshot_messages_prefix() + "\nPATIENT RECORD\n" + record_text,
                        data_origin)


def cot_llm(model, record_text: str, data_origin: str = "synthetic") -> AgentRun:
    return _single_shot(model, COT_SYSTEM, "PATIENT RECORD\n" + record_text, data_origin)


def role_llm(model, record_text: str, data_origin: str = "synthetic") -> AgentRun:
    return _single_shot(model, ROLE_SYSTEM, "PATIENT RECORD\n" + record_text, data_origin)


def _mode(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(json.dumps(v, sort_keys=True) for v in vals).most_common(1)[0][0]


def vote(assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Majority vote per field across samples (ties: the first sample's value)."""
    ok = [a for a in assessments if a and not a.get("_parse_failed")]
    if not ok:
        return assessments[0] if assessments else normalise(None)
    out = dict(ok[0])
    tox = []
    for term in TERMS_REQUIRED:
        grades = []
        for a in ok:
            g = next((t.get("grade") for t in a.get("toxicities", [])
                      if (t.get("canonical_term") or t.get("term")) == term), None)
            grades.append(g)
        m = _mode(grades)
        if m is not None:
            tox.append({"term": term, "canonical_term": term, "v5_term_used": False, "grade": json.loads(m),
                        "evidence": "self-consistency vote"})
    out["toxicities"] = tox
    for key in ("chemotherapy_decision", "urgency_level"):
        m = _mode([a.get(key) for a in ok])
        out[key] = json.loads(m) if m is not None else None
    m = _mode([a.get("ctrcd") for a in ok])
    out["ctrcd"] = json.loads(m) if m is not None else {}
    # actions, data requests, medications: from the first sample that agrees with the voted decision
    agree = next((a for a in ok if a.get("chemotherapy_decision") == out["chemotherapy_decision"]), ok[0])
    for key in ("actions", "data_requests", "medications_proposed", "guideline_disagreements", "summary"):
        out[key] = agree.get(key, out.get(key))
    return out


def selfcons_llm(model, record_text: str, data_origin: str = "synthetic", k: int = 3,
                 temperature: float = 0.7) -> AgentRun:
    run = AgentRun(assessment={}, raw_final=None)
    old = getattr(model, "temperature", None)
    samples = []
    try:
        if old is not None:
            model.temperature = temperature
        for i in range(k):
            r = _single_shot(model, PURE_SYSTEM, "PATIENT RECORD\n" + record_text, data_origin)
            run.llm_calls += r.llm_calls
            run.prompt_tokens += r.prompt_tokens
            run.completion_tokens += r.completion_tokens
            run.latency_s += r.latency_s
            run.trace += [dict(t, sample=i) for t in r.trace]
            if r.error:
                run.error = r.error
                break
            samples.append(r.assessment)
    finally:
        if old is not None:
            model.temperature = old
    run.assessment = vote(samples) if samples else normalise(None)
    run.raw_final = {"samples": len(samples)}
    return run

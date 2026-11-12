"""
utils/llm_client.py
───────────────────
Instrumented LLM call used by the C1 experiment (run_c1_experiment.py).

utils/llm_narrative.call_groq_narrative() returns only the text, which is
right for the pipeline. An experiment also needs latency, token counts,
the exact model id, retries on rate limits, and a mock backend so the
whole experiment can be dry-run without an API key. That is what this
module adds. It uses the same model setting (GROQ_MODEL) as the pipeline,
so experiment and production talk to the same model.
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from utils.llm_narrative import GROQ_MODEL, _GROQ_AVAILABLE

if _GROQ_AVAILABLE:
    from groq import Groq


@dataclass
class LLMResult:
    text: Optional[str]
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model: str = ""
    error: Optional[str] = None
    attempts: int = 1
    json_mode_used: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def extract_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parses the first JSON object in a model reply (tolerates code fences
    and leading prose). Returns None if no valid object is found."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg


class GroqChat:
    """Real backend. Retries rate-limit errors with exponential backoff."""

    def __init__(self, api_key: Optional[str] = None, model: str = GROQ_MODEL,
                 temperature: float = 0.1, max_retries: int = 6, min_interval_s: float = 2.0):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key or not _GROQ_AVAILABLE:
            raise RuntimeError("GROQ_API_KEY not set or groq package not installed "
                               "(use --dry-run to test the experiment without an API key).")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.is_mock = False

    def __call__(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        attempt, delay = 0, 5.0
        use_json = json_mode
        while True:
            attempt += 1
            t0 = time.time()
            try:
                kwargs = dict(model=self.model, temperature=self.temperature,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}])
                if use_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                self._last_call = time.time()
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    text=resp.choices[0].message.content,
                    latency_s=round(time.time() - t0, 3),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    model=getattr(resp, "model", self.model),
                    attempts=attempt,
                    json_mode_used=use_json,
                )
            except Exception as err:  # noqa: BLE001 - classify below
                self._last_call = time.time()
                if use_json and not _is_rate_limit(err):
                    # Model/endpoint rejected JSON mode (or the JSON it produced
                    # failed server-side validation): retry once as plain text;
                    # run_c1_experiment.parse_judge extracts the JSON itself.
                    use_json = False
                    continue
                if _is_rate_limit(err) and attempt <= self.max_retries:
                    time.sleep(delay + random.uniform(0, 1.5))
                    delay = min(delay * 2, 120.0)
                    continue
                return LLMResult(text=None, latency_s=round(time.time() - t0, 3), model=self.model,
                                 error=f"{type(err).__name__}: {err}", attempts=attempt,
                                 json_mode_used=use_json)


class MockChat:
    """Dry-run backend. `responder(system, user, json_mode)` returns text.
    Results are marked is_mock so analysis can stamp outputs as NOT RESULTS."""

    def __init__(self, responder: Callable[[str, str, bool], str]):
        self.responder = responder
        self.model = "MOCK"
        self.is_mock = True

    def __call__(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        t0 = time.time()
        text = self.responder(system, user, json_mode)
        return LLMResult(text=text, latency_s=round(time.time() - t0, 4),
                         prompt_tokens=len((system + user).split()),
                         completion_tokens=len(text.split()), model="MOCK",
                         json_mode_used=json_mode)
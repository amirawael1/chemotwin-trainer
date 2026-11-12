"""
agentic/llm.py
──────────────
One chat interface for every model the thesis uses, so all arms of the
comparison call models in exactly the same way.

All supported providers expose the OpenAI-compatible endpoint
POST {base_url}/chat/completions, so a single HTTP client covers them and no
SDK is needed (only the Python standard library):

  provider    where it runs        default model          key env var
  ─────────   ──────────────────   ────────────────────   ─────────────────
  groq        Groq cloud           openai/gpt-oss-120b    GROQ_API_KEY       (free: 200k tokens/day, 8k/min)
  nvidia      NVIDIA NIM cloud     nemotron-3-super-120b  NVIDIA_API_KEY     (free with NVIDIA Developer Program)
  gemini      Google AI Studio     gemini-2.5-flash       GEMINI_API_KEY     (free tier)
  openrouter  OpenRouter           ...:free models        OPENROUTER_API_KEY (free: ~50 requests/day)
  cerebras    Cerebras cloud       gpt-oss-120b           CEREBRAS_API_KEY   (paid)
  lmstudio    your laptop          (model loaded in app)  none
  ollama      your laptop          medgemma (set --model) none
  mock        nowhere (tests)      MOCK                   none

PRIVACY RULE (hospital data request): records from the hospital must never be
sent to an external service. Every request carries a `data_origin`; a
request with data_origin="hospital" to a cloud provider raises
PrivacyViolation before any network call is made. Synthetic patients may use
any provider.

Tool use is done with a plain-JSON protocol (see agentic/reasoning_agent.py)
rather than provider-specific function calling, because the local medical
models (MedGemma via LM Studio / Ollama) do not support native tool calls
reliably. The same protocol is then used for every model, which keeps the
comparison fair.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

def _load_env_file() -> None:
    """Reads API keys from the project's .env file (same file the v1 pipeline
    uses). Variables already set in the shell take precedence."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


_load_env_file()

# Free-tier limits change often: check each provider's console before a long run.
# `tpm` (tokens per minute) makes the client pace itself instead of hitting 429s.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b",
             "key_env": "GROQ_API_KEY", "local": False, "tpm": 8000},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "model": "nvidia/nemotron-3-super-120b-a12b",
               "key_env": "NVIDIA_API_KEY", "local": False, "stream": True},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash",
               "key_env": "GEMINI_API_KEY", "local": False},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "nvidia/nemotron-3-super-120b-a12b:free",
                   "key_env": "OPENROUTER_API_KEY", "local": False},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1", "model": "gpt-oss-120b",
                 "key_env": "CEREBRAS_API_KEY", "local": False},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": "medgemma-4b-it",
                 "key_env": None, "local": True, "timeout": 1800},
    "ollama": {"base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), "model": "medgemma",
               "key_env": None, "local": True, "timeout": 1800},
}

DATA_ORIGINS = ("synthetic", "hospital")


class PrivacyViolation(RuntimeError):
    """Raised when hospital data would be sent to a non-local model."""


class RateLimitExhausted(RuntimeError):
    """Raised when the provider keeps refusing for quota reasons (stop the run, resume later)."""


@dataclass
class LLMResult:
    text: Optional[str]
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model: str = ""
    error: Optional[str] = None
    attempts: int = 1
    finish_reason: Optional[str] = None   # "stop" = finished; "length" = hit max_tokens
    reasoning_text: str = ""              # hidden reasoning (tail only), for diagnosing empty answers


@dataclass
class ChatModel:
    """OpenAI-compatible chat client with rate-limit backoff and a privacy guard."""
    provider: str = "groq"
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096  # reasoning models spend part of this on hidden reasoning
    timeout_s: float = 180.0
    max_stream_s: float = 900.0  # whole streamed reply; longer = treated as a network failure and retried
    max_retries: int = 6
    min_interval_s: float = 1.0
    json_mode: bool = True
    _last_call: float = field(default=0.0, repr=False)
    usage: Dict[str, int] = field(default_factory=lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{self.provider}'. Choose from {sorted(PROVIDERS)} or use MockModel.")
        cfg = PROVIDERS[self.provider]
        self.model = self.model or os.getenv("AGENTIC_MODEL") or cfg["model"]
        self.base_url = (self.base_url or cfg["base_url"]).rstrip("/")
        self.is_local = cfg["local"]
        self.api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "not-needed"
        if not self.api_key:
            raise RuntimeError(f"{cfg['key_env']} is not set (needed for provider '{self.provider}').")
        self.is_mock = False
        self.tpm = cfg.get("tpm")
        # local GPUs (laptop, Kaggle T4) generate slowly: a long answer can take several minutes
        self.timeout_s = max(self.timeout_s, cfg.get("timeout", 0))
        # Streaming keeps long generations alive through gateways that drop idle
        # connections (seen with NVIDIA NIM as "Remote end closed connection").
        self.stream = bool(cfg.get("stream"))
        self._window: List[tuple] = []  # (time, tokens) of calls in the last minute

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: List[Dict[str, str]], data_origin: str = "synthetic",
             json_mode: Optional[bool] = None) -> LLMResult:
        if data_origin not in DATA_ORIGINS:
            raise ValueError(f"data_origin must be one of {DATA_ORIGINS}")
        if data_origin == "hospital" and not self.is_local:
            raise PrivacyViolation(
                f"Refusing to send hospital data to cloud provider '{self.provider}'. "
                "Use --provider lmstudio or ollama for hospital records.")
        use_json = self.json_mode if json_mode is None else json_mode
        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._pace(messages)
        attempt, delay = 0, 5.0
        while True:
            attempt += 1
            body: Dict[str, Any] = {"model": self.model, "messages": messages,
                                    "temperature": self.temperature, "max_tokens": self.max_tokens}
            if use_json:
                body["response_format"] = {"type": "json_object"}
            if self.stream:
                body["stream"] = True
                body["stream_options"] = {"include_usage": True}
            t0 = time.time()
            try:
                data = self._post_stream(body) if self.stream else self._post(body)
                self._last_call = time.time()
                usage = data.get("usage") or {}
                self.usage["calls"] += 1
                self._window.append((time.time(), (usage.get("prompt_tokens") or 0)
                                     + (usage.get("completion_tokens") or 0)))
                self.usage["prompt_tokens"] += usage.get("prompt_tokens") or 0
                self.usage["completion_tokens"] += usage.get("completion_tokens") or 0
                choice = data["choices"][0]
                msg = choice["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                return LLMResult(text=msg.get("content") or "", latency_s=round(time.time() - t0, 3),
                                 prompt_tokens=usage.get("prompt_tokens"),
                                 completion_tokens=usage.get("completion_tokens"),
                                 model=data.get("model", self.model), attempts=attempt,
                                 finish_reason=choice.get("finish_reason"), reasoning_text=reasoning[-8000:])
            except urllib.error.HTTPError as err:
                self._last_call = time.time()
                detail = err.read().decode("utf-8", "replace")[:500]
                if err.code == 429 or "rate" in detail.lower() and "limit" in detail.lower():
                    if "per day" in detail.lower() or "tokens per day" in detail.lower() or attempt > self.max_retries:
                        raise RateLimitExhausted(f"{self.provider} quota exhausted: {detail}")
                    time.sleep(delay + random.uniform(0, 1.5))
                    delay = min(delay * 2, 120.0)
                    continue
                if err.code == 400 and "called a tool" in detail.lower() and attempt <= 3:
                    # GPT-OSS on Groq sometimes emits a native tool call although our protocol is
                    # plain JSON and no tools are declared; Groq rejects it. The slip is random,
                    # so the same request is simply sent again (at most 3 times).
                    time.sleep(2)
                    continue
                if err.code == 400 and self.stream and "stream" in detail.lower():
                    self.stream = False  # endpoint rejected streaming options: plain requests
                    continue
                if use_json and err.code == 400:
                    use_json = False  # endpoint/model rejected JSON mode: retry as plain text
                    continue
                return LLMResult(text=None, latency_s=round(time.time() - t0, 3), model=self.model,
                                 error=f"HTTP {err.code}: {detail}", attempts=attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
                # Network drop or gateway timeout: back off 10, 20, 40, 80 s before giving up.
                self._last_call = time.time()
                if attempt <= 4:
                    time.sleep(10 * 2 ** (attempt - 1))
                    continue
                hint = (" Is LM Studio / Ollama running with its server started?" if self.is_local
                        else " Check the internet connection (VPN, Wi-Fi, firewall).")
                return LLMResult(text=None, latency_s=round(time.time() - t0, 3), model=self.model,
                                 error=f"NETWORK {type(err).__name__}: {err}.{hint}", attempts=attempt)

    def _pace(self, messages) -> None:
        """Wait until this request fits in the provider's tokens-per-minute budget."""
        if not self.tpm:
            return
        est = sum(len(m["content"]) for m in messages) // 4 + 1500  # prompt + typical reply
        while True:
            now = time.time()
            self._window = [(t, n) for t, n in self._window if now - t < 60]
            used = sum(n for _, n in self._window)
            if not self._window or used + est <= self.tpm:
                return
            time.sleep(max(1.0, 60 - (now - self._window[0][0]) + 0.5))

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
            # An explicit User-Agent is required: Cloudflare (in front of Groq) rejects
            # Python's default "Python-urllib" agent with HTTP 403, error code 1010.
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "AgenticChemoTwin/2.0 (research; python)",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))


    def _post_stream(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Server-sent events -> the same dict shape as a non-streamed reply.
        Reasoning deltas (reasoning_content) are ignored; only the answer is kept."""
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                     "User-Agent": "AgenticChemoTwin/2.0 (research; python)",
                     "Authorization": f"Bearer {self.api_key}"})
        parts, think, usage, model, finish = [], [], {}, self.model, None
        deadline = time.time() + self.max_stream_s
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            for raw in resp:
                # A stalled stream that keeps the connection alive never hits the socket
                # timeout; cap the whole reply so a run cannot hang for hours.
                if time.time() > deadline:
                    raise TimeoutError(f"streamed reply exceeded {self.max_stream_s:.0f} s")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                model = chunk.get("model", model)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        parts.append(piece)
                    hidden = delta.get("reasoning_content") or delta.get("reasoning")
                    if hidden:
                        think.append(hidden)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        text = "".join(parts)
        reasoning = "".join(think)
        if not usage:  # provider did not report usage: estimate (about 4 characters per token)
            usage = {"prompt_tokens": sum(len(m["content"]) for m in body["messages"]) // 4,
                     "completion_tokens": len(text) // 4}
        return {"model": model, "usage": usage,
                "choices": [{"message": {"content": text, "reasoning_content": reasoning[-8000:]},
                             "finish_reason": finish}]}


class MockModel:
    """Scripted model for tests and dry runs. `responder(messages) -> str`."""

    is_local = True
    is_mock = True

    def __init__(self, responder: Callable[[List[Dict[str, str]]], str], name: str = "MOCK"):
        self.responder = responder
        self.model = name
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @property
    def name(self) -> str:
        return f"mock:{self.model}"

    def chat(self, messages, data_origin: str = "synthetic", json_mode: Optional[bool] = None) -> LLMResult:
        t0 = time.time()
        text = self.responder(messages)
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += sum(len(m["content"].split()) for m in messages)
        self.usage["completion_tokens"] += len(text.split())
        return LLMResult(text=text, latency_s=round(time.time() - t0, 4), model=self.model)


def make_model(provider: str, model: Optional[str] = None, **kw) -> Any:
    return ChatModel(provider=provider, model=model, **kw)


def last_json_with(text: Optional[str], key: str) -> Optional[Dict[str, Any]]:
    """The LAST complete JSON object in `text` that contains `key` (used to recover an answer that a
    reasoning model wrote inside its hidden reasoning but never repeated as the visible answer)."""
    if not text:
        return None
    best, i = None, 0
    while True:
        start = text.find("{", i)
        if start == -1:
            return best
        obj = extract_json(text[start:])
        if isinstance(obj, dict) and key in obj:
            best = obj
        i = start + 1


def extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """First JSON object in a reply; tolerates code fences, prose and
    <think> blocks emitted by reasoning models."""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None
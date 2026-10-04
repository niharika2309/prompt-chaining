"""Thin OpenAI-compatible client for the local Qwen3.8-27B endpoint.

Endpoint facts (probed live):
  - response_format json_schema / json_object -> 400 "structured output needs
    xgrammar on the server"  =>  NO server-side structured output available.
  - plain text + chat_template_kwargs enable_thinking=false works and is fast.
  - the model is a VLM and a reasoning model (reasoning_content field).

Both parser approaches therefore use instruction-based JSON + Pydantic
validation, keeping the experiment controlled.
"""
import json
import time

from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            timeout=config.TIMEOUT,
        )
    return _client


class CallResult:
    def __init__(self):
        self.ok = False
        self.content: str | None = None      # raw model text (last attempt)
        self.data: dict | None = None        # parsed JSON object, if ok
        self.attempts = 0
        self.latency = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (fences tolerated)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    return json.loads(t[start : end + 1])


def chat_json(
    messages: list[dict],
    validate=None,
    tag: str = "",
) -> CallResult:
    """One logical JSON call with retry-on-invalid-JSON and retry-on-validation.

    `validate(data: dict) -> dict|None` is called after json.loads; return the
    (possibly repaired) data or None to trigger a corrective retry.
    """
    res = CallResult()
    msgs = list(messages)
    for attempt in range(config.MAX_RETRIES + 1):
        res.attempts = attempt + 1
        t0 = time.monotonic()
        resp = client().chat.completions.create(
            model=config.MODEL,
            messages=msgs,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING}},
        )
        res.latency += time.monotonic() - t0
        usage = resp.usage
        res.prompt_tokens += usage.prompt_tokens if usage else 0
        res.completion_tokens += usage.completion_tokens if usage else 0

        content = resp.choices[0].message.content or ""
        res.content = content
        data = None
        error = None
        try:
            data = extract_json(content)
            if validate is not None:
                data = validate(data)
            if data is None:
                error = "validation failed"
        except Exception as e:  # noqa: BLE001
            error = str(e)

        if data is not None and error is None:
            res.ok = True
            res.data = data
            return res

        # corrective retry: show the model what went wrong
        msgs.append({"role": "assistant", "content": content})
        msgs.append({
            "role": "user",
            "content": (
                f"Your reply could not be parsed ({error}). "
                "Reply again with a single valid JSON object only — "
                "no markdown, no commentary, no trailing text."
            ),
        })

    return res  # ok=False

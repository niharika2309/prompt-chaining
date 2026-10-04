"""Approach A — MONOLITHIC: one long prompt, one call.

The prompt carries L1 + L2 + ALL 12 family detail blocks so the model itself
decides which block applies. This is the "long prompt" side of the comparison.
"""
from ..core import prompts
from ..core import schema
from ..core.llm import chat_json


def _validate(data: dict, expected_family: str = None):
    """Tolerate a model-side "document" wrapper, validate, normalize.

    Some models nest the spec's fields under {"document": {...}} even though the
    spec renders them at top level; some flatten them. Accept either shape by
    merging: top-level ParsedDocument fields win, then the wrapper's fields.
    """
    if "document" in data and isinstance(data["document"], dict):
        merged = dict(data["document"])
        for k, v in data.items():
            if k != "document":
                merged.setdefault(k, v)
        data = merged
    try:
        doc = schema.ParsedDocument.model_validate(data)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"schema: {e}") from e

    fam = expected_family or doc.document_family
    if fam not in schema.FAMILIES:
        fam = "other"
    # normalize: only the family that matches document_family may be populated;
    # demote any other blocks (and any all-null blocks) to None.
    for f in ("rx", "lab", "imaging", "ecg", "ed", "discharge", "note",
              "correspondence", "assessment", "checklist", "certificate", "consent"):
        block = getattr(doc, f)
        if block is None:
            continue
        is_empty = all(v in (None, "", []) for v in block.model_dump().values() if v is not None)
        if f != fam or is_empty:
            setattr(doc, f, None)
    if doc.document_family != fam:
        doc.document_family = fam
    return doc.model_dump()


def parse(text: str, pdf_name: str) -> dict:
    all_fams = [f for f in schema.FAMILIES if f != "other"]
    system = (
        f"{prompts.PERSONA}\n\n"
        f"{prompts.RULES}\n\n"
        "Extract every field you can from the document below into the JSON object "
        "specified here. Populate the family detail block that matches the document\n"
        "type and set the other family blocks to null.\n"
        f"JSON specification:\n{prompts.full_spec(all_fams)}"
    )
    user = (
        f"Document text (extracted from {pdf_name}):\n=== DOCUMENT START ===\n{text}\n"
        "=== DOCUMENT END ===\n\nExtract all fields now."
    )
    res = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        validate=lambda d: _validate(d),
        tag="monolithic",
    )
    return {
        "approach": "monolithic",
        "pdf": pdf_name,
        "ok": res.ok,
        "final": res.data,
        "meta": {
            "latency": round(res.latency, 2),
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.total_tokens,
            "attempts": res.attempts,
            "raw_head": (res.content or "")[:300],
        },
    }

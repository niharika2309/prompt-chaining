"""Approach B — PROMPT CHAINING: two sequential calls, each output feeds the next.

  Step 1  classify : identify document_type + document_family (drives everything)
  Step 2  extract  : schema tailored to the identified family ONLY (short, focused)

The chain's structural advantages over the monolithic prompt:
  - step 2 sees a SMALL spec (one family block instead of all twelve)
  - steps are isolated, attributable, and individually retryable

A third "verify" step (re-check every field against the source) was
benchmarked in the 3-way comparison (EVALUATION_REPORT.md): on this task it
returned the draft unchanged in 49/50 documents while costing +62% latency /
+75% completion tokens, so it was removed from the pipeline.
"""
from ..core import prompts
from ..core import schema
from ..core.llm import chat_json


def _validate(data: dict, expected_family: str | None = None):
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


def _s1(text: str, pdf_name: str) -> dict:
    system = (
        f"{prompts.PERSONA}\n\n"
        "You are classifying an Australian medical document. Decide:\n"
        "- document_type: the specific document type\n"
        "- document_family: the broader family it belongs to\n"
        f"Use: {prompts.s1_classification_spec()}\n"
        "Base the decision on the document header, title, and content. "
        "Reply with JSON only."
    )
    user = (
        f"Document text (extracted from {pdf_name}):\n=== DOCUMENT START ===\n{text}\n"
        "=== DOCUMENT END ===\n\nClassify this document."
    )
    res = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tag="chain:s1",
    )
    if res.ok and isinstance(res.data, dict):
        inner = res.data.get("classification")
        if isinstance(inner, dict):
            res.data = inner
    return res


def _s2(text: str, pdf_name: str, classification: dict | None) -> dict:
    fam = (classification or {}).get("document_family", "other")
    if fam not in schema.FAMILIES:
        fam = "other"
    doc_types = [t for t, f in schema.TYPE_TO_FAMILY.items() if f == fam]
    doc_types = ", ".join(doc_types) if fam != "other" else "any document type"

    system = (
        f"{prompts.PERSONA}\n\n"
        f"{prompts.RULES}\n\n"
        f"Based on step 1, this document is a {fam} document "
        f"(type is one of: {doc_types}).\n"
        "Extract every field you can into the JSON object below. The spec covers\n"
        "identity, clinical core, and the detail block for THIS family only.\n"
        f"JSON specification:\n{prompts.full_spec([fam])}"
    )
    user = (
        f"Document text (extracted from {pdf_name}):\n=== DOCUMENT START ===\n{text}\n"
        "=== DOCUMENT END ===\n\nExtract all fields now."
    )
    return chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        validate=lambda d: _validate(d, expected_family=fam),
        tag="chain:s2",
    )


def parse(text: str, pdf_name: str) -> dict:
    steps = {}

    s1 = _s1(text, pdf_name)
    steps["s1_classify"] = s1
    classification = s1.data if s1.ok else None

    s2 = _s2(text, pdf_name, classification)
    steps["s2_extract"] = s2

    final = s2.data if (s2.ok and s2.data) else None

    lat = sum(s.latency for s in steps.values())
    ptok = sum(s.prompt_tokens for s in steps.values())
    ctok = sum(s.completion_tokens for s in steps.values())

    return {
        "approach": "chain",
        "pdf": pdf_name,
        "ok": final is not None,
        "final": final,
        "classification": classification,
        "meta": {
            "latency": round(lat, 2),
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "total_tokens": ptok + ctok,
            "attempts": sum(s.attempts for s in steps.values()),
            "steps": {
                k: {
                    "ok": s.ok,
                    "latency": round(s.latency, 2),
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "attempts": s.attempts,
                }
                for k, s in steps.items()
            },
            "raw_head": (s2.content or "")[:300] if s2 else "",
        },
    }

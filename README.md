# prompt-chaining

Side-by-side comparison of two LLM approaches to **medical document parsing**,
run on the same 50 synthetic Australian medical PDFs and the same Qwen3.8-27B
endpoint:

| Approach | Structure |
|---|---|
| **Monolithic** | One long prompt: full field spec (all 12 document-family blocks) + document text → single call |
| **Prompt chaining** | 3 calls: classify type/family → extract with a *family-tailored* spec → verify every field against the source text |

Both use identical persona, rules, Pydantic schema, temperature (0), and
thinking disabled — the only variable is prompt structure.

## Data

`data/synthetic-australian-medical-documents-sample/` (gitignored, 23 MB) —
downloaded from
[RootCauseAnalytics/synthetic-australian-medical-documents-sample](https://huggingface.co/datasets/RootCauseAnalytics/synthetic-australian-medical-documents-sample)
(CC-BY-NC 4.0, PHI-free, synthetic):

- 50 digital PDFs (`pdfs/`) + 50 scanned variants (`pdfs_scanned/`, phase 2)
- `ground_truth.csv` / `.jsonl` — 115-column per-doc labels (scoring target)
- `splits.json` — train/test partition
- 29 document types clustered into 12 families (see `schema.py`)

## The output schema (shared by both approaches)

`schema.py` derives its Pydantic model from the dataset's own ground truth:

- **L1 identity** (all docs): type, family, number, date, patient block, facility
- **L2 clinical core** (all docs): specialty, principal diagnosis, ICD codes, additional diagnoses
- **L3 detail** (one family block): fields present in ≥50% of the family's documents;
  the long tail goes to a `key_values` catch-all

## Usage

```bash
# one-shot: extract -> run both approaches -> evaluate -> reports
doc_env/bin/python main.py all --docs pilot --concurrency 3 --tag pilot11

# or step by step
doc_env/bin/python main.py extract --docs pilot
doc_env/bin/python main.py run --approach both --docs pilot --tag pilot11
doc_env/bin/python main.py evaluate --run reports/runs/<dir>
doc_env/bin/python main.py report   --run reports/runs/<dir>
```

`--docs`: `pilot` (11 stratified docs) | `all` (50) | comma-separated filenames.

Outputs land in `reports/runs/<ts>_<tag>/` (gitignored):

| File | Contents |
|---|---|
| `<approach>.jsonl` | raw model outputs + per-call latency/tokens/attempts |
| `eval_<approach>.json` | per-field verdicts vs ground truth |
| `metrics.json` | aggregate metrics for both approaches |
| `summary.md` | markdown summary |
| `side_by_side.html` | the side-by-side: GT \| monolithic \| chain, colour-coded |

## Metrics

- **field_acc** — (correct + correctly-abstained) / all scored field pairs
- **recall** — over fields populated in ground truth (partial counts 0.5)
- **precision** — over fields the model populated
- **hallucination rate** — model values whose tokens are absent from the source text
- **clean docs** — docs with zero wrong values and zero source-unsupported values
- **latency / tokens** — per document (chain = sum of its 3 calls)

Scoring is controlled: the scored field inventory follows the document's
ground-truth family, so both approaches are measured on the same denominator
even if one misclassifies the family. Values parked in the `key_values`
catch-all (what a misclassified doc degenerates to) do not count toward typed
fields. Four fields are excluded as unscorable (`n/a` in reports): `patient.sex`
(never printed on the documents), `clinical.icd_codes` (GT populated but most
docs print no code), `clinical.specialty` (GT = case specialty, docs print the
issuing department, and verbatim rules forbid inference), `document_number`
(GT id is a synthetic-library batch id in a footer the model is told to ignore).

## LLM endpoint

`http://100.117.48.99:8888/v1` — model `Qwen3.8-27B` (VLM, reasoning model).
No server-side structured output (xgrammar not installed), so both approaches
use instruction-based JSON + Pydantic validation with corrective retries.
`enable_thinking=false` keeps calls fast; flip `ENABLE_THINKING` in `config.py`
for an ablation.

## Modules

| File | Role |
|---|---|
| `config.py` | endpoint, model, retry/concurrency, paths, pilot doc set |
| `schema.py` | Pydantic output schema + 29 types → 12 family mapping |
| `prompts.py` | shared persona, rules, JSON-spec renderer |
| `llm.py` | OpenAI-compatible client, JSON extraction, retry, accounting |
| `extract_text.py` | pypdf text extraction with caching |
| `parse_monolithic.py` | approach A |
| `parse_chain.py` | approach B (classify → extract → verify) |
| `evaluate.py` | field mapping, normalization, verdicts, metrics |
| `report.py` | markdown + HTML side-by-side |
| `main.py` | CLI |

## Phase 2 (not yet built)

Scanned PDFs via the model's vision input (render pages → images), to measure
OCR-failure behaviour of both approaches.

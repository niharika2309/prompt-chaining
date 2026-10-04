# Evaluation Report — Monolithic Prompt vs Prompt Chaining

**Task:** structured extraction of 50 synthetic Australian medical PDFs (29 document types, 12 families)
**Model:** Qwen3.8-27B @ `100.117.48.99:8888`, temperature 0, thinking off, no server-side structured output

| Approach | Structure | Calls/doc |
|---|---|---|
| **Monolithic** | one call: full spec (L1 + L2 + **all 12** family blocks) + document text | 1 |
| **Chain** | ① classify type/family → ② extract with a **family-tailored** spec | 2 |

Both use identical persona, rules, Pydantic schema, temperature and endpoint — the only variable is prompt structure. The temp-0 model is deterministic: finals are byte-identical across same-day runs (verified 50/50), so quality and token numbers are exactly reproducible.

| Run | Contents | Role |
|---|---|---|
| `20261003_081135_full50` (quiet window) | monolithic; `chain_quiet_steps.json` (per-doc s1+s2 stats) | clean latency source |
| `20261003_192116_full50` | monolithic + chain | canonical 2-way (quality, tokens) |
| `20261003_075445_pilot11` (11 docs) | monolithic | sanity check |

---

## Executive summary

Your working hypothesis — *prompt chaining is more efficient than one big prompt* — **is confirmed.**

| | Monolithic (1 call) | Chain (2 calls) |
|---|---|---|
| Field accuracy | 0.757 | 0.760 (CI on Δ includes 0) |
| Recall | 0.825 | 0.818 (n.s.) |
| Precision | 0.714 | 0.722 (n.s., directional) |
| Hallucination rate | 0.001¹ | 0.001¹ |
| Clean docs | 12/50 | 12/50 |
| **Tokens/doc** | 6,410 | **4,988 (−22%, p<0.001)** |
| **Latency/doc** (quiet window) | 18.4 s | **16.9 s (−8%, n.s.)**² |
| Throughput @ 3 workers | 587 docs/h | 639 docs/h (+9%) |
| Illustrative cost @ $0.15/$0.60 per Mtok | 1.49 ¢ | **1.21 ¢ (−19%)** |
| Schema-valid first try | 28/50 (56%) | 32/50 (64%) |
| Family classification | 47/50 (94%) | 46/50 (92%) |
| LLM calls/doc (incl. retries) | 1.44 | 2.36 |

¹ one value each — the same tokenizer artifact (a trailing dot in the scorer's token class); with the dot stripped, hallucination rate = 0.000 for both. See §2.5.
² derived: per-doc s1+s2 latencies from the quiet session (the chain's two calls, measured in that same window). See §3.1.

**Bottom line.**

1. **Quality is a statistical tie** (n = 50; CIs on the differences include 0 — §2). The architecture does not change the answer.
2. **The focused-spec mechanism is a real, measured saving.** The chain's extract prompt is 25% smaller than the monolithic prompt (the 11 unused family blocks, ≈ 2.4 k tokens, are genuinely removed), which more than pays for the ~1.1 k-token classification step → **the chain is 22% cheaper in tokens on 49 of 50 documents** and a hair faster on latency.
3. **A verify step was tested and is gone.** A 3-step variant (classify → extract → verify) was benchmarked on the same 50 docs: +49% latency and +19% tokens versus the monolithic, and — the decisive measurement — the verify pass **returned the draft unchanged on 49 of 50 documents** (the one doc it touched scored identically). It bought nothing measurable, so it was removed from the codebase (§5).
4. **Recommendation: use the 2-step chain** — cheaper, routing-capable, observable, zero measured quality cost. Monolithic remains a fine, slightly-more-expensive baseline (§7).

---

## 1. The metric framework — what to compare, and why

A defensible architecture comparison needs **four families of metrics**. Any one alone gives a misleading verdict (latency alone would "prove" chaining useless; accuracy alone would "prove" it free). For a structured-extraction service the decision is a joint one: *which architecture gives the right answer, reliably, within budget?*

### A. Output quality — is the data right?

Scored against the dataset's own `ground_truth.csv` (115 labels/doc), with a controlled denominator (the scored field inventory is fixed per document by its GT family, so both approaches are measured on the same fields even when one misclassifies the family).

| Metric | Definition | Why it matters |
|---|---|---|
| **Field accuracy** | (correct + correctly-abstained) / all scored field pairs | The number a consumer of the schema sees: "is this cell trustworthy?" |
| **Recall** | correct (+0.5·partial) over fields populated in GT | Downstream workflows fail silently on *missing* values. |
| **Precision** | matches over fields the model populated | The opposite risk — wrong values get *used*. |
| **Hallucination rate** | model values whose tokens are absent from the source text / model-populated | The most dangerous failure in medical data. GT sparsity makes precision noisy; a source-anchored check is robust. |
| **Clean-doc rate** | docs with zero wrong values and zero source-unsupported values | The audit-oriented summary: what fraction can ship with zero review? |
| **Error mix** | counts of wrong / missed / extra / partial | Two systems can tie on accuracy with opposite error profiles (misses vs fabrications); the fix strategy differs. |

### B. Efficiency — what does it cost?

| Metric | Definition | Why it matters |
|---|---|---|
| **Latency** (mean, median, p95, max per doc) | wall-clock per document | User-facing; for a chain it is the *sum* of sequential calls. |
| **Tokens** (prompt / completion split) | per document, summed over calls | The price driver — and the **split is the diagnostic**: it separates *context re-send* cost (prompt) from *re-generation* cost (completion). |
| **Derived cost** | tokens × price in/out | Makes the tradeoff monetary. (Absolute $ is endpoint-specific; we quote tokens + an illustrative price.) |
| **Derived throughput** | workers × 3600 / mean latency | What a fixed-size GPU pool processes per hour. |
| **Call count / retries** | HTTP calls per doc | Each call consumes a worker slot and a request; corrective retries compound. |

### C. Structural reliability — does the design fail gracefully?

This is where prompt *structure* should matter most, independent of model quality.

| Metric | Definition | Why it matters |
|---|---|---|
| **First-try schema validity** | fraction of docs whose JSON passes Pydantic validation on attempt 1 | A focused spec should fail less; retries are pure waste. |
| **Retry distribution** | attempts per doc | Bounded vs unbounded waste under the retry policy. |
| **Parse success** | docs yielding a valid record | Hard-failure rate of the pipeline as a whole. |
| **Family classification accuracy** | declared family vs GT family | For the chain this *drives* step 2 (tailored spec); for the monolithic it *routes* the validator. |
| **Misclassification cost** | fields lost when the family is wrong | Both validators demote the wrong family block → all L3 fields become `missed`. A 1-step error cascades. |
| **Confidence calibration** | s1 `confidence` vs actual correctness | A calibrated score is the only sane trigger for conditional behaviour ("route/verify only if unsure"). |

### D. Robustness to heterogeneity — where does each design break?

| Metric | Why it matters |
|---|---|
| **Per-family quality breakdown** (dirty values by GT family) | These are not one system — each is 12+ effective configurations. A win in `rx` and a loss in `ed` must be visible. |
| **Per-field verdict flips** (fixed vs broken vs the other approach) | Localizes *which* fields the architecture change helps or hurts — the actionable list for prompt work. |
| **Worst-case docs** | Production SLAs are set by the tail, not the mean. |

### And the statistics to judge any of them

Both approaches parse the *same* documents, so every per-doc quantity supports a **paired** test:

- Paired doc-level bootstrap (20 000 resamples, seed 7) → 95% CI on the *difference* of each aggregate.
- Wilcoxon signed-rank for per-doc latency/tokens/dirty values.
- Exact sign test and Fisher's exact for binary per-doc outcomes.

**Power check:** at n = 50 the 95% CI on the per-doc field-accuracy difference spans roughly ±0.8 pp. Detecting a true 0.5 pp difference at 80% power needs **~400 documents**. Any quality verdict below that n is "not shown different", never "equally good".

---

## 2. Measured results — quality

### 2.1 Headline accuracy (n = 50 docs; 1,368 / 1,366 scored field pairs)

| | Monolithic | Chain | Δ chain (bootstrap 95% CI on per-doc Δ) |
|---|---|---|---|
| Field accuracy | 0.757 | 0.760 | +0.002 [−0.005, +0.009] — **n.s.** |
| Recall | 0.825 | 0.818 | −0.007 [−0.027, +0.009] — **n.s.** |
| Precision | 0.714 | 0.722 | +0.007 [−0.001, +0.016], p(>base)=0.96 — n.s. but directional |
| Hallucination rate | 0.001 | 0.001 | ≈0 — n.s. (artifact, §2.5) |
| Clean docs | 12/50 | 12/50 | (same 24%, different document mixes) |
| Dirty values/doc | 1.82 | 1.76 | −0.06, Wilcoxon p = 0.37 — **n.s.** (chain cleaner on 5 docs, dirtier on 3, tied 42) |

**Verdict: statistically indistinguishable on quality at this n.**

### 2.2 Error mix (where the small differences live)

| Verdict | Monolithic | Chain | Read |
|---|---|---|---|
| correct | 599 | 594 | ≈ tie |
| partial | 54 | 54 | tie |
| **wrong** | 90 | **87** | chain: slightly fewer wrong values (directional precision edge) |
| **missed** | **16** | 24 | chain: 1 extra family misclassification vs mon (4 vs 3) → L3 blocks demoted → 8 extra misses |
| **extra** | 172 | 163 | GT is sparsely labelled; both over-extract similarly; most extras are source-supported (not counted as dirty) |

Two opposing micro-effects, netting to ≈ 0: the chain's focused spec trims a few wrong values, while its one extra misclassification loses recalled fields.

### 2.3 Per-field verdict flips (chain vs monolithic)

- **Chain fixed 14 field verdicts, broke 12** — a wash.
- Fixed (14): `lab.lab_ref`, `checklist.procedure`, `imaging.{radiologist, modality, examination}`, `assessment.va_{left,right}`, `ed.{triage_category, ed_clinician, arrival_datetime}`, `correspondence.recipient`, `document_family` (0028 — a mon misclassification the chain got right), `facility.ward`, `note.note_date`.
- Broken (12): `imaging.procedure` (×2), `document_family` (×2 = the two s1 misclassifications mon got right: 0021, 0035), `ed.{destination_hospital, ambulance_priority}`, `rx.medications`, `note.{anaesthetic_type, duration_minutes, procedure, operation_date}`, `facility.provider`.
- The two `document_family` breaks are **cascade breaks** — one wrong s1 label costs the whole L3 block (see 0035 in §4.2). The same cascade exists in the monolithic (its 3 misclassifications behave identically), so this is not a chain-specific tax — but it is why a family-conflict guard matters for *any* design.

### 2.4 Per-family dirty values (mean/doc)

| Family | n | Monolithic | Chain | Better |
|---|---|---|---|---|
| rx | 8 | 0.75 | 0.62 | chain |
| assessment | 7 | 2.43 | 2.14 | chain |
| imaging | 5 | 2.80 | 2.60 | chain |
| correspondence | 5 | 0.80 | 0.60 | chain |
| ed | 5 | 3.20 | 3.40 | monolithic (marginal) |
| discharge | 5 | 1.80 | 1.80 | tie |
| note | 4 | 3.00 | 3.00 | tie |
| lab / checklist / ecg | 3 each | 0.67 / 0.00 / 2.67 | same | tie |

The chain's small aggregate edges are concentrated in `rx`/`assessment`/`correspondence`. `ed` and `note` are genuinely hard for **all** variants (3–4 dirty values/doc) — the residual error is *model* error, not *architecture* error. No family shows a significant gap.

### 2.5 The one "hallucination" each — a scoring artifact

Each approach "fabricated" exactly one value: `clinical.additional_diagnoses = "cellulitis"` on two different vascular-ultrasound documents (mon: 0031; chain: 0008). **Both are the same failure mode** (promoting the findings' mention into an "additional diagnosis" slot the GT leaves empty) and **both are actually source-supported**: the word does appear in the document ("consistent with cellulitis."). The flag is a tokenizer artifact — the scorer's token class `[a-z0-9%.]+` keeps a trailing period, so the source token is `cellulitis.` and never matches the model's clean `cellulitis`. With the dot stripped, **hallucination rate = 0.000 for both approaches** and clean docs remain 12/50 each. Verdicts unaffected; the artifact should be fixed in `evaluate.py` (recommendation 2).

### 2.6 Pilot (11 docs) sanity check

On the pilot run — which was benchmarked *before* the verify step was removed — monolithic and chain were a **perfect tie on every quality metric** (all verdict counts identical to 4 decimals; raw outputs differ, normalization collapses the differences), with the chain +33% latency / +23% tokens. The full-50 runs are the real comparison; the pilot only confirms the direction. (The pilot's chain result files were removed with the verify version; the pilot directory now holds the monolithic run.)

---

## 3. Measured results — efficiency

### 3.1 Latency and tokens (per document, paired)

**Latency is quoted from the quiet-window session** (morning, same endpoint state for all calls; the evening session ran under spiky shared-endpoint load — per-doc queue ratios 0.2×–21× — and is used only for load-invariant metrics). The chain's quiet-window latency is the per-doc sum of its s1+s2 latencies from that same session (`chain_quiet_steps.json`). Tokens and quality come from the evening run — both are load-invariant and byte-identical across same-day runs.

| | Monolithic | Chain | Δ chain vs mon |
|---|---|---|---|
| Mean latency | 18.38 s | **16.89 s** | **−1.5 s (−8%), p = 0.43 n.s.** (25 faster / 24 slower docs) |
| Median / p95 | 17.0 / 35.7 s | 15.4 / 31.9 s | −8% / −11% |
| Mean tokens | 6,410 | **4,988** | **−1,422 (−22%), p < 0.001** (49/50 fewer) |
| — prompt tokens | 5,243 | 3,962 (−25%) | — |
| — completion tokens | 1,167 | 1,026 (−12%) | — |
| LLM calls/doc (incl. retries) | 1.44 | 2.36 | +64% |
| Throughput @ 3 workers | 587 docs/h | 639 docs/h | **+9%** |

### 3.2 Where the token differences come from (quiet-window step decomposition)

| Step | Latency | % of approach | Prompt tok | Completion tok | Attempts |
|---|---|---|---|---|---|
| s1 classify | 2.6 s | 15% | 1,098 | 71 | 1.00 |
| s2 extract | 14.3 s | 85% | 2,864 | 955 | 1.36 |
| **chain total** | **16.9 s** | | 3,962 | 1,026 | |
| monolithic (1 call) | 18.4 s | | 5,243 | 1,167 | 1.44 |

The economics, token by token:

1. **The focused spec pays for the classification step and then some.** s2's prompt (2,864) is **45% smaller** than the monolithic prompt (5,243) — the 11 unused family blocks (≈ 2.4 k tokens, the exact measured difference between the two prompts, which share persona/rules/document) are genuinely removed. The classify step (a second copy of the document + its small spec) costs 1,098 tokens. Net: **the chain's prompt is 25% smaller than the monolithic's** — the one efficiency gain that survives all the way to the ledger.
2. **Sequential calls ⇒ latency adds, but the arithmetic comes out even.** The chain's calls are data-dependent, so latency is the sum of its steps: 2.6 s + 14.3 s ≈ 16.9 s, statistically tied with the monolithic's 18.4 s — the s2 speed-up (smaller spec → shorter generation) offsets the s1 overhead.
3. **Retries are slightly cheaper in the chain**: 18 docs needed one corrective retry vs 22 for the monolithic (s2 first-try 64% vs 56%, Fisher p = 0.54 — n.s., direction matches the focused-spec mechanism).

### 3.3 Money (illustrative — the endpoint is self-hosted, so quote tokens and let your $/token decide)

At a stated assumption of $0.15 / $0.60 per Mtok input/output (a mid-size public price point; output weighs heavier per token):

- Monolithic **1.49 ¢/doc** · Chain **1.21 ¢/doc (−19%)**
- On a 100 k-document year: the chain saves ≈ $280 of tokens and buys routing/observability.

---

## 4. Measured results — structural reliability

### 4.1 First-try validity, retries, parse success

| | Monolithic | Chain |
|---|---|---|
| Schema-valid first try | 28/50 (56%) | 32/50 (64%) — +8 pp, Fisher p = 0.54 (n.s.) |
| Corrective retries | 22 docs × 1 | 18 docs × 1 (all at s2) |
| Parse success | 50/50 | 50/50 |

Directionally the focused spec does its job (fewer malformed attempts); at n = 50 the difference is not significant.

### 4.2 Family classification — the chain's load-bearing step (and its shared cost)

- Monolithic: **47/50 (94%)** · Chain: **46/50 (92%)**
- Misclassified (mon): 0015 (checklist vs discharge), 0028 + 0034 (fluid order → `other`, GT `rx`)
- Misclassified (chain): 0015, 0034, 0021 (ambulance → `other`, GT `ed`), 0035 (anaesthetic record → `other`, GT `note`)
- **Cost made concrete (0035):** a wrong family demotes the whole L3 block — the chain loses 5 `note.*` fields that the monolithic captured. Note the same cascade exists in the monolithic (its 3 misclassifications behave identically under the same validator): this is a *shared* design risk of the schema, not a chain-specific one — but it is the strongest argument for a **family-conflict guard** (e.g. s2 refuses a family whose key header fields don't appear in the text; or L3 is extracted spec-agnostically and routed afterwards).

### 4.3 Confidence — the routing signal is (currently) useless

s1 reports `confidence: 1.0` for **all 50 documents — including the 4 it got wrong** (correct: mean 1.0, incorrect: mean 1.0). The model is not giving a usable calibration, so confidence-based routing or conditional extra passes cannot be built on this signal as-is; s1 would need logprob-based or self-consistency calibration first (recommendation 6).

### 4.4 The advantage that costs nothing: observability

The chain's step-level accounting (the §3.2 table exists *because* of it) is a genuine operational win: per-step latency, tokens, attempts, pass/failure are logged and attributable; the monolithic run is a black box of one ~6.4 k-token call. For production triage — "why did this document fail?" — the chain is more debuggable by construction. This is the strongest defensible argument for chaining independent of the efficiency arithmetic.

---

## 5. The verify step: measured, then removed

Before this report settled on the 2-step design, a 3-step variant (classify → extract → **verify**: re-read source + draft, re-emit the corrected JSON) was benchmarked on the same 50 docs, same day, same endpoint. Its result files and code are no longer in the repo; the measurements that drove the removal:

| | Monolithic | 2-step chain | 3-step chain |
|---|---|---|---|
| Field accuracy | 0.757 | 0.760 | 0.760 |
| Recall / Precision | 0.825 / 0.714 | 0.818 / 0.722 | 0.818 / 0.722 |
| Clean docs | 12/50 | 12/50 | 12/50 |
| Mean latency (quiet) | 18.4 s | 16.9 s | 27.4 s |
| Mean tokens | 6,410 | 4,988 | 7,625 |

- **The verify pass was a no-op on 49 of 50 documents** — s3 returned the s2 draft byte-identical. The one document it modified (0024) scored identically (same verdicts, same clean status, same dirty count).
- It cost +62% latency and +75% completion tokens versus the 2-step chain (it re-sends the document *and* the draft, then re-emits the whole JSON: +1,862 prompt / +775 completion per doc), and +49% latency / +19% tokens versus the monolithic — to change nothing measurable.
- The one mechanism that would have justified an *unconditional* verify (a confidence trigger for a conditional one) was absent: s1 confidence was 1.0 on all 50 docs, the 4 wrong ones included (§4.3).

**Verdict: on this task, with this model, verification is pure overhead and was removed from the pipeline.** Revisit only under the conditions in §7 — e.g. a stronger/different verifier model, longer documents where errors concentrate, or calibrated s1 confidence making the pass conditional.

---

## 6. "Why is prompt chaining better?" — the arguments, checked against the data

| # | Argument | Mechanism | Verdict on this data |
|---|---|---|---|
| 1 | **Focused context** — each step sees only the spec it needs | s2 spec = 1 of 12 family blocks | **Confirmed and net-positive**: s2 prompt −45% vs mon; the classify step (a second copy of the document + its small spec) costs 1,098 tokens, less than the ≈ 2.4 k spec saving → **chain prompt −25%, total tokens −22%** |
| 2 | **Dedicated verification pass** | a second, check-only reading of source + draft | **Refuted, and removed**: 49/50 no-op; +62% latency / +75% completion versus the 2-step chain (§5) |
| 3 | **Error isolation** — a bad attempt is contained in one step; retries cheap and local | per-step retry + fallback | **Partially supported**: first-try validity +8 pp (n.s.), fewer docs needing retry (18 vs 22), zero parse failures. Real but small at this n. |
| 4 | **Specialization headroom** — steps can use different temperature/model/thinking (cheap classify + expensive extract) | design space | Untestable here (both steps same config) — the main *future* upside of the architecture. |
| 5 | **Observability & testability** — per-step metrics; classifier benchmarkable in isolation | step accounting | **Supported** (§4.4). Real, zero-cost, operationally valuable. |

**So why *is* prompt chaining better?** The 2-step chain is a strictly better *skeleton*: smaller per-step context that measurably shrinks the token bill, isolated and attributable steps, a router you can tune per step, and observability the monolithic can't match — at zero measured quality cost (and a directional precision edge). Your intuition was right.

---

## 7. When each design wins — decision guide

The equation: `value(focused spec + isolation + observability) − cost(extra call + re-send)`. Measured on short documents (median ~308 words, smaller than the 12-family spec itself):

**Use the chain (this includes the current task):**
- documents stay within the context window (the spec saving beats the re-send cost, as measured: −22% tokens);
- you need **type routing** downstream (different handlers/schemas per family) — s1 is 92% accurate at 1.1 k tokens / 2.6 s, and doubles as the router (calibrate its confidence first, §4.3);
- you want per-step accounting for production triage and regression testing.

**The monolithic remains a fine choice when:**
- you want the fewest moving parts (1 call, no router dependency) and the ~22% token saving is not worth a second call;
- documents are *very* short, where even s1's fixed cost starts to matter relative to the (already small) spec.

**A verify step is worth re-adding only when:**
- it can be shown to pay for itself: precision-critical fields, a **stronger/different** verifier model, or longer documents where errors concentrate and a second read has more to catch;
- you have **calibrated** s1 confidence to make it conditional (verify only uncertain documents) — at n = 50, unconditional verification was a 49/50 no-op;
- **prefix caching** makes the re-send nearly free *and* the re-emission is priced tolerably — the decision then reduces to "is the verify worth its completion tokens?", which needs a larger n to answer.

**Both designs need, before production:** a family-conflict guard (§4.2) so one misclassification cannot silently zero an L3 block, and a calibrated s1 score (§4.3).

---

## 8. Recommendations

1. **Ship the 2-step chain (classify → family-tailored extract) for this task** — 22% fewer tokens, statistically tied quality (directional precision edge), +9% throughput, plus routing and observability. `docparse/pipeline/chain.py` is the pipeline; `main.py run --approach both|chain` runs it.
2. **Fix the scorer artifact** in `evaluate.py` (trailing punctuation in the token class): flips the only "fabrications" to supported, hallucination rate 0.001 → 0.000. Cheap, and it makes the hallucination metric trustworthy for future runs.
3. **Add a family-conflict guard** to the validator (both approaches share the cascade risk, §4.2): e.g. s2 refuses a family whose key header fields don't appear in the text, or L3 is extracted spec-agnostically and routed afterwards.
4. **Spend prompt effort where the error actually lives:** `ed` and `note` run 3–4 dirty values/doc in *both* architectures — that is model error a per-family prompt will fix; the architecture question is settled either way.
5. **Scale n to ≥ 200–400 docs** (the dataset's remaining 450, plus scanned phase-2) to resolve sub-1 pp quality differences — and to re-test verification on longer/messier documents, where a second read might finally have something to catch.
6. **Calibrate s1's confidence** (logprobs or 3-sample self-consistency) to unlock conditional verify and downstream routing.
7. **Re-measure at scan time (phase 2):** vision tokens per call and OCR-noise behaviour may change the per-step economics entirely; the chain's step accounting is what will tell you.

---

## 9. Limitations

- **n = 50** (11 in pilot). Quality verdicts are "not shown different", bounded by the CIs in §2 (±~0.8–1 pp on the field-acc difference; ~400 docs would resolve 0.5 pp). Efficiency verdicts (tokens) *are* significant (−22%); the latency difference (−8%) is not, at this n.
- **Latency windows.** The canonical latency table uses the quiet-window session; the chain's latency is the per-doc s1+s2 sum *from that same session* (a same-window estimate, preserved in `chain_quiet_steps.json`), not a separate chain run. The evening session ran under spiky shared-endpoint load (per-doc queue ratios 0.2×–21×) and is used only for load-invariant metrics (quality, tokens, verdicts).
- **One model, one config.** No thinking, temperature 0, no structured output, no prefix cache. Results are properties of this stack; §7 lists the conditions that would change them.
- **Synthetic, text-extractable PDFs.** Scanned documents (phase 2) and vision input may weight the steps very differently.
- **GT sparsity.** `extra` verdicts are common (163–172) because GT labels fewer fields than the documents print; scoring deliberately does not penalize source-supported extras, so precision is measured against a sparse target.
- **Four documented field exclusions** (`patient.sex`, `icd_codes`, `specialty`, `document_number` — GT fields the documents never print in scorable form); excluded symmetrically from both approaches.
- **Determinism assumption.** Same-day cross-run finals were byte-identical (50/50), supporting the reproducibility used throughout; a different endpoint deployment could break it.
- Scorer artifact (trailing-dot tokens) affects `hallucination_rate` as described in §2.5; all other metrics are unaffected.

---

## Appendix — reproduction

```bash
# the 2-way benchmark (one session, both approaches)
doc_env/bin/python main.py run --approach both --docs all --concurrency 3 --tag full50
doc_env/bin/python main.py evaluate --run reports/runs/<dir>

# deep comparison: paired tests, step decomposition, cost accounting (any run dir)
doc_env/bin/python -m docparse.evaluation.analysis reports/runs/<dir> --out reports/analysis_<tag>.json

# canonical 2-way table for this report (quiet-window latency + evening quality/tokens)
doc_env/bin/python -m docparse.evaluation.canonical_table --out reports/canonical_2way.json

# browse: summary | side-by-side (GT | monolithic | chain, colour-coded) | raw
doc_env/bin/python -m streamlit run app.py
```

| File | Contents |
|---|---|
| `reports/runs/20261003_081135_full50/` | quiet-window monolithic run + `chain_quiet_steps.json` (canonical latency source) |
| `reports/runs/20261003_192116_full50/` | both approaches in one session (canonical 2-way) |
| `reports/runs/20261003_075445_pilot11/` | 11-doc pilot, monolithic (quality-tie sanity check) |
| `reports/canonical_2way.json`, `reports/analysis_full50.json` | the data behind every number in this report |
| `docparse/evaluation/analysis.py` | generic N-approach analysis |
| `docparse/evaluation/canonical_table.py` | assembles the canonical 2-way table |
| `docparse/pipeline/chain.py` | the 2-step chain: classify → extract |
| `reports/corpus/*.txt` | extracted document texts (source of the anti-hallucination check) |

*Report generated 2026-10-03 from the runs above. All statistics: paired doc-level tests; bootstrap seed 7, 20 000 resamples. Illustrative prices are stated assumptions, not quotes.*

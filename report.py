"""Reports: markdown summary + interactive-ish HTML side-by-side.

For every document: Ground Truth | Monolithic | Chain, cell by cell,
with verdict colouring — the actual "side by side" deliverable.
"""
import json
from pathlib import Path

import evaluate
import schema

VERDICT_CLASS = {
    "correct": "ok", "abstain": "ok", "partial": "partial",
    "wrong": "bad", "missed": "missed", "extra": "bad", "n/a": "na",
}


def display_value(v) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, list):
        return "; ".join(display_value(x) for x in v)
    if isinstance(v, dict):
        return " ".join(f"{k}: {display_value(x)}" for k, x in v.items())
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "..."


def gt_display(col: str, gt: dict) -> str:
    v = (gt or {}).get(col, "")
    return display_value(v) if v.strip() else "—"


def _pairs_for_doc(row: dict, gt: dict, rec: dict | None) -> list[tuple]:
    """(field_label, gt_text, model_text, verdict) for one document/approach."""
    if rec is None or not rec.get("final"):
        return []
    fam = (rec["final"].get("document_family") or "other")
    if fam not in schema.FAMILIES:
        fam = "other"
    gt_fam = schema.TYPE_TO_FAMILY.get(gt.get("document_type", ""), "other")

    def gv(m, gcol, free=False):
        # recompute display for the verdict rows we already scored
        return m

    out = []
    for v in row["verdicts"]:
        label = v["field"]
        out.append((label, v.get("g", "") and display_value(v["g"]),
                    v.get("m", "") and display_value(v["m"]),
                    "extra-fabricated" if (v["verdict"] == "extra" and v.get("fabricated")) else v["verdict"]))
    return out


def field_label_map(row_verdicts: list[dict], gt: dict, rec: dict) -> dict[str, str]:
    """Map verdict field -> GT column used, for GT cell display (best effort).

    Uses the GT family (verdict labels follow the GT family in evaluate.py)."""
    fam = schema.TYPE_TO_FAMILY.get(gt.get("document_type", ""), "other")
    m = {
        "document_type": "document_type",
        "document_family": "document_type",
        "document_number": "document_id",
        "patient.full_name": "patient_name", "patient.dob": "patient_dob",
        "patient.age": "patient_age", "patient.sex": "patient_sex",
        "patient.address": "patient_address", "patient.phone": "patient_phone",
        "patient.occupation": "patient_occupation", "patient.mrn": "mrn",
        "patient.medicare": "medicare", "patient.nok_name": "nok_name",
        "patient.nok_relationship": "nok_relationship", "patient.allergies": "allergies",
    }
    m.update({f"facility.{k}": c for k, c in [
        ("lhd", "hospital_lhd"), ("ward", "ward")]})
    m.update({f"clinical.{k}": c for k, c in [
        ("specialty", "specialty"), ("icd_codes", "principal_icd"),
        ("additional_diagnoses", "additional_diagnoses")]})
    if fam == "correspondence":
        m["clinical.principal_diagnosis"] = "principal_diagnosis_or_problem"
    else:
        m["clinical.principal_diagnosis"] = "principal_diagnosis"
    for k, c in evaluate.L3_MAP.get(fam, {}).items():
        if c:
            m[f"{fam}.{k}"] = c
    return m


# ---------------- markdown ----------------

def write_markdown(run_dir: Path, mon: list[dict], chain: list[dict],
                   gts: dict, mon_eval, chain_eval, mon_m, chain_m):
    lines = [
        "# Document parser comparison: monolithic prompt vs prompt chaining",
        "",
        "Model: `Qwen3.8-27B` @ 100.117.48.99:8888 (thinking off, temp 0)",
        f"Documents: {len(gts)} (synthetic Australian medical docs)",
        "",
        "**Scoring methodology.** Fields are scored against `ground_truth.csv`. "
        "Four fields are excluded (`n/a`, grey): `patient.sex` (never printed on the "
        "documents), `clinical.icd_codes` (GT populated but most docs print no code), "
        "`clinical.specialty` (GT = case specialty, docs print issuing department), "
        "`document_number` (GT id is a synthetic-library batch ID in a footer the model "
        "is told to ignore). `clean docs` counts a doc clean when it has no `wrong` "
        "values and no source-UNsupported values (source-supported extras, where GT is "
        "sparsely labelled, are not failures).",
        "",
        "## Headline comparison",
        "",
        "| Metric | Monolithic (1 call) | Chain (3 calls) |",
        "|---|---|---|",
    ]
    for k in ("field_acc", "recall", "precision", "hallucination_rate"):
        lines.append(f"| {k} | {mon_m.get(k)} | {chain_m.get(k)} |")
    lines.append(f"| clean docs | {mon_m.get('clean_docs')} | {chain_m.get('clean_docs')} |")
    lines.append(f"| mean latency (s/doc) | {mon_m.get('mean_latency')} | {chain_m.get('mean_latency')} |")
    lines.append(f"| mean total tokens/doc | {mon_m.get('mean_tokens')} | {chain_m.get('mean_tokens')} |")
    lines.append(f"| schema-valid first try | {mon_m.get('first_try')} | {chain_m.get('first_try')} |")
    lines += ["", "## Per-document (model field verdicts vs ground truth)", ""]
    def dirty(row):
        return sum(1 for v in row["verdicts"]
                   if v["verdict"] == "wrong" or (v["verdict"] == "extra" and v.get("fabricated")))

    for rec in mon:
        fam = (rec.get("final") or {}).get("document_family", "?")
        mrow = next((e for e in mon_eval if e["pdf"] == rec["pdf"]), {"verdicts": []})
        crows = next((e for e in chain_eval if e["pdf"] == rec["pdf"]), {"verdicts": []})
        lines.append(f"- `{rec['pdf']}` (family {fam}) — "
                     f"monolithic {dirty(mrow)} dirty values, chain {dirty(crows)} dirty values")
    p = run_dir / "summary.md"
    p.write_text("\n".join(lines))
    return p


# ---------------- html ----------------

def write_html(run_dir: Path, mon: list[dict], chain: list[dict], gts: dict,
               mon_eval, chain_eval, mon_m, chain_m):
    def doc_row(rec, ev, gt):
        if not rec or not rec.get("final"):
            return "<td class='bad' colspan=2>PARSE FAILED</td>"
        fam = (rec["final"].get("document_family") or "other")
        fmap = field_label_map(ev["verdicts"], gt, rec)
        cells = []
        for v in ev["verdicts"]:
            lab = v["field"]
            gcol = fmap.get(lab)
            gtext = gt_display(gcol, gt) if gcol else "n/a"
            mtext = display_value(v.get("m")) if v.get("m") else "—"
            cls = "fab" if (v["verdict"] == "extra" and v.get("fabricated")) else VERDICT_CLASS.get(v["verdict"], "")
            cells.append(
                f"<tr><td class='f'>{lab}</td><td>{gtext}</td>"
                f"<td class='{cls}'>{mtext}</td><td>{v['verdict']}</td></tr>")
        return "\n".join(cells)

    rows = []
    for pdf, gt in gts.items():
        mrec = next((r for r in mon if r["pdf"] == pdf), None)
        crec = next((r for r in chain if r["pdf"] == pdf), None)
        mev = next((e for e in mon_eval if e["pdf"] == pdf), {"verdicts": []})
        cev = next((e for e in chain_eval if e["pdf"] == pdf), {"verdicts": []})
        type_html = gt.get("document_type", "")
        fam = schema.TYPE_TO_FAMILY.get(type_html, "other")
        rows.append(f"""
<h2>{pdf} <small>({type_html} / {fam})</small></h2>
<table>
<tr><th>field</th><th>ground truth</th><th>monolithic</th><th>verdict</th></tr>
{doc_row(mrec, mev, gt)}
</table>
<table>
<tr><th>field</th><th>ground truth</th><th>chain</th><th>verdict</th></tr>
{doc_row(crec, cev, gt)}
</table>""")

    def mrow(name, a, b):
        return f"<tr><td>{name}</td><td>{a}</td><td>{b}</td></tr>"

    summary = "\n".join([
        mrow("field accuracy", mon_m.get("field_acc"), chain_m.get("field_acc")),
        mrow("recall (GT-populated fields)", mon_m.get("recall"), chain_m.get("recall")),
        mrow("precision (model-populated fields)", mon_m.get("precision"), chain_m.get("precision")),
        mrow("hallucination rate", mon_m.get("hallucination_rate"), chain_m.get("hallucination_rate")),
        mrow("clean docs (no wrong/fabricated)", mon_m.get("clean_docs"), chain_m.get("clean_docs")),
        mrow("mean latency s/doc", mon_m.get("mean_latency"), chain_m.get("mean_latency")),
        mrow("mean tokens/doc", mon_m.get("mean_tokens"), chain_m.get("mean_tokens")),
        mrow("schema-valid first try", mon_m.get("first_try"), chain_m.get("first_try")),
    ])

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Monolithic vs Chain — {len(gts)} docs</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;margin:24px;color:#222}}
 h1{{font-size:22px}} h2{{font-size:15px;margin:18px 0 6px}}
 table{{border-collapse:collapse;margin-bottom:14px}}
 td,th{{border:1px solid #d0d0d0;padding:4px 8px;font-size:12.5px;text-align:left;vertical-align:top}}
 th{{background:#f4f4f4}} td.f{{font-family:ui-monospace,monospace;white-space:nowrap}}
 td.ok{{background:#e6f4e6}} td.partial{{background:#fdf6dc}}
 td.bad,td.missed{{background:#fbe3e3}} td.fab{{background:#e39aa5;font-weight:600}}
 small{{color:#888}}
 .legend span{{display:inline-block;padding:2px 8px;margin-right:6px;border:1px solid #ccc;font-size:12px}}
 .ok{{background:#e6f4e6}} .partial{{background:#fdf6dc}} .bad{{background:#fbe3e3}} .fab{{background:#e39aa5}} .missed{{background:#fbe3e3}} .na{{background:#f0f0f4;color:#888}}
</style></head><body>
<h1>Document parser: monolithic prompt vs prompt chaining</h1>
<p>Model Qwen3.8-27B @ 100.117.48.99:8888 · synthetic AU medical PDFs · thinking off</p>
<table><tr><th>metric</th><th>monolithic (1 call, full spec)</th><th>chain (classify → extract → verify)</th></tr>{summary}</table>
<p class="legend">legend: <span class="ok">correct/abstain</span><span class="partial">partial</span><span class="bad">wrong/missed</span><span class="fab">fabricated (not in source)</span><span class="na">not scored (see methodology)</span></p>
{''.join(rows)}
</body></html>"""
    p = run_dir / "side_by_side.html"
    p.write_text(html)
    return p

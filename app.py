"""Streamlit app: browse the monolithic-vs-chain comparison runs.

    doc_env/bin/python -m streamlit run app.py

Reads whatever is under reports/runs/<dir>/ (manifest, <approach>.jsonl,
eval_<approach>.json, metrics.json). Read-only: no API calls, no writes.
"""
import csv
import html
import json
from collections import defaultdict

import pandas as pd
import streamlit as st

import config
import evaluate
import schema

st.set_page_config(page_title="Monolithic vs Chain — medical doc parsing", layout="wide")

APPROACHES = ["monolithic", "chain"]
APPROACH_LABEL = {"monolithic": "Monolithic (1 call)", "chain": "Chain (classify→extract→verify)"}

CELL = {
    "correct": "background:#d9edda",
    "abstain": "background:#f2f2f5",
    "partial": "background:#fff3cd",
    "wrong": "background:#f8d7da",
    "missed": "background:#f8d7da",
    "extra": "background:#f8d7da",
    "n/a": "background:#ececf1;color:#888",
}
FAB = "background:#e39aa5;font-weight:600"
ICON = {"correct": "✅", "abstain": "·", "partial": "🟡", "wrong": "❌",
        "missed": "⬜", "extra": "➕", "n/a": "–"}


def cell_style(v: dict) -> str:
    if v.get("verdict") == "extra" and v.get("fabricated"):
        return FAB
    return CELL.get(v.get("verdict"), "")


def esc(x) -> str:
    if x is None or x == "" or x == []:
        return "<span style='color:#aaa'>—</span>"
    if isinstance(x, list):
        x = " ; ".join(str(i) for i in x)
    if isinstance(x, dict):
        x = " ".join(f"{k}: {v}" for k, v in x.items())
    s = str(x)
    if len(s) > 90:
        s = s[:87] + "..."
    return html.escape(s)


# ---------------- loading ----------------

def run_dirs() -> list[str]:
    if not config.RUNS_DIR.exists():
        return []
    return sorted(p.name for p in config.RUNS_DIR.iterdir() if p.is_dir())


def _load_json_any(path):
    txt = path.read_text()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


@st.cache_data(show_spinner=False)
def load_gt_all() -> dict[str, dict]:
    return {r["pdf_filename"]: r for r in csv.DictReader(open(config.GT_CSV))}


@st.cache_data(show_spinner=False)
def load_run(name: str) -> dict:
    d = config.RUNS_DIR / name
    out = {"name": name}
    mp = d / "manifest.json"
    out["manifest"] = json.loads(mp.read_text()) if mp.exists() else {}
    out["records"], out["evals"] = {}, {}
    for a in APPROACHES:
        p = d / f"{a}.jsonl"
        q = d / f"eval_{a}.json"
        out["records"][a] = _load_json_any(p) if p.exists() else []
        evs = _load_json_any(q) if q.exists() else []
        out["evals"][a] = {e["pdf"]: e for e in evs}
    mp2 = d / "metrics.json"
    out["metrics"] = json.loads(mp2.read_text()) if mp2.exists() else {}
    return out


# ---------------- field -> value helpers (display) ----------------

def model_value(final: dict, field: str):
    """Raw model value for a verdict field (pre-truncation)."""
    if not final:
        return None
    if "." not in field:
        return final.get(field)
    head, rest = field.split(".", 1)
    if "." in rest:  # e.g. no deeper nesting in our schema
        return None
    block = final.get(head)
    if isinstance(block, dict):
        v = block.get(rest)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return " ; ".join(f"{x.get('name','')} {x.get('dose','')}".strip() for x in v)
        return v
    return None


def gt_value(field: str, gt: dict, gt_fam: str):
    """Raw GT value for display. Accept-any columns: first populated wins."""
    if field == "document_type" or field == "document_family":
        return gt.get("document_type")
    if field.startswith("patient."):
        col = {"full_name": "patient_name", "dob": "patient_dob", "age": "patient_age",
               "sex": "patient_sex", "address": "patient_address", "phone": "patient_phone",
               "occupation": "patient_occupation", "mrn": "mrn", "medicare": "medicare",
               "nok_name": "nok_name", "nok_relationship": "nok_relationship",
               "allergies": "allergies"}.get(field.split(".", 1)[1])
        return gt.get(col) if col else None
    if field == "facility.name":
        for c in ("facility", "hospital_name", "gp_clinic", "specialist_clinic"):
            if gt.get(c):
                return gt[c]
        return None
    if field == "facility.lhd":
        return gt.get("hospital_lhd")
    if field == "facility.ward":
        return gt.get("ward")
    if field == "facility.provider":
        for c in evaluate.PROVIDER_COLS.get(gt_fam, []):
            if gt.get(c):
                return gt[c]
        return None
    if field == "clinical.specialty":
        return gt.get("specialty")
    if field == "clinical.principal_diagnosis":
        cols = (["principal_diagnosis", "principal_diagnosis_or_problem"]
                if gt_fam == "correspondence" else ["principal_diagnosis"])
        for c in cols:
            if gt.get(c):
                return gt[c]
        return None
    if field == "clinical.icd_codes":
        return gt.get("principal_icd")
    if field == "clinical.additional_diagnoses":
        return gt.get("additional_diagnoses")
    if field == "document_number":
        return gt.get("document_id")
    if "." in field:
        fkey = field.split(".", 1)[1]
        col = evaluate.L3_MAP.get(gt_fam, {}).get(fkey)
        return gt.get(col) if col else None
    return None


# ---------------- pages ----------------

def page_summary(data: dict, gts: dict):
    mon_m = data["metrics"].get("monolithic", {})
    chain_m = data["metrics"].get("chain", {})
    pdfs = [r["pdf"] for r in data["records"]["monolithic"] + data["records"]["chain"]]
    pdfs = sorted(set(pdfs))

    rows = []
    for k, label in [("field_acc", "Field accuracy"), ("recall", "Recall (GT-populated)"),
                     ("precision", "Precision (model-populated)"),
                     ("hallucination_rate", "Hallucination rate"),
                     ("clean_docs", "Clean docs"), ("first_try", "Schema-valid first try"),
                     ("mean_latency", "Mean latency s/doc"), ("mean_tokens", "Mean tokens/doc")]:
        rows.append({"metric": label,
                     "monolithic": str(mon_m.get(k, "—")) if mon_m.get(k) is not None else "—",
                     "chain": str(chain_m.get(k, "—")) if chain_m.get(k) is not None else "—"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Verdict distribution (all docs)")
    keys = ("correct", "partial", "wrong", "missed", "extra", "abstain", "n/a")
    rows = []
    for a in APPROACHES:
        vc = defaultdict(int)
        for pdf in pdfs:
            for v in data["evals"][a].get(pdf, {}).get("verdicts", []):
                vc[v["verdict"]] += 1
        rows.append({"approach": APPROACH_LABEL[a], **{k: vc.get(k, 0) for k in keys}})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("By document family (GT family)")
    fam_rows = defaultdict(lambda: {a: defaultdict(int) for a in APPROACHES})
    for pdf in pdfs:
        gt = gts.get(pdf, {})
        fam = schema.TYPE_TO_FAMILY.get(gt.get("document_type", ""), "other")
        for a in APPROACHES:
            for v in data["evals"][a].get(pdf, {}).get("verdicts", []):
                if v["verdict"] == "n/a":
                    continue
                fam_rows[fam][a][v["verdict"]] += 1
                if v["verdict"] == "wrong" or (v["verdict"] == "extra" and v.get("fabricated")):
                    fam_rows[fam][a]["dirty"] += 1
    out = []
    for fam in sorted(fam_rows):
        row = {"family": fam, "docs": sum(1 for p in pdfs
                 if schema.TYPE_TO_FAMILY.get(gts.get(p, {}).get("document_type", ""), "other") == fam)}
        for a in APPROACHES:
            c = fam_rows[fam][a]
            n = sum(v for k, v in c.items() if k != "dirty")
            row[f"{a} acc"] = round((c["correct"] + c["abstain"]) / n, 3) if n else None
            row[f"{a} dirty"] = c["dirty"]
        out.append(row)
    st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)

    st.subheader("Latency / tokens per document")
    rows = []
    for pdf in pdfs:
        for a in APPROACHES:
            rec = next((r for r in data["records"][a] if r["pdf"] == pdf), {})
            meta = rec.get("meta") or {}
            rows.append({"doc": pdf.replace(".pdf", "")[:38], "approach": a,
                         "latency_s": meta.get("latency"), "total_tokens": meta.get("total_tokens")})
    df = pd.DataFrame(rows).pivot(index="doc", columns="approach", values="latency_s")
    st.bar_chart(df)
    df = pd.DataFrame(rows).pivot(index="doc", columns="approach", values="total_tokens")
    st.bar_chart(df)


def _doc_table(gt: dict, rec_m: dict, rec_c: dict, ev_m: dict, ev_c: dict) -> str:
    gt_fam = schema.TYPE_TO_FAMILY.get(gt.get("document_type", ""), "other")
    fm = {v["field"]: v for v in ev_m.get("verdicts", [])}
    fc = {v["field"]: v for v in ev_c.get("verdicts", [])}
    fields = list(fm) + [f for f in fc if f not in fm]

    trs = ["<tr><th style='text-align:left'>field</th><th>ground truth</th>"
           "<th>monolithic</th><th>chain</th></tr>"]
    for f in fields:
        vm, vc = fm.get(f, {}), fc.get(f, {})
        gm = esc(gt_value(f, gt, gt_fam))
        mm = esc(model_value((rec_m or {}).get("final"), f))
        mc = esc(model_value((rec_c or {}).get("final"), f))
        trs.append(
            f"<tr><td class='f'>{html.escape(f)}</td><td>{gm}</td>"
            f"<td style='{cell_style(vm)}'>{ICON.get(vm.get('verdict'), '·')} {mm}</td>"
            f"<td style='{cell_style(vc)}'>{ICON.get(vc.get('verdict'), '·')} {mc}</td></tr>")
    return (f"<table><tr><th></th><th></th><th></th><th></th></tr>".replace("<tr>", "") +
            "".join(trs) + "</table>")


def page_sides(data: dict, gts: dict):
    pdfs = sorted({r["pdf"] for a in APPROACHES for r in data["records"][a]})
    if not pdfs:
        st.warning("No records in this run.")
        return
    pick = st.selectbox("document", pdfs,
                        format_func=lambda p: f"{p}  [{schema.TYPE_TO_FAMILY.get(gts.get(p, {}).get('document_type',''),'?')}]")
    gt = gts.get(pick, {})
    rec_m = next((r for r in data["records"]["monolithic"] if r["pdf"] == pick), {})
    rec_c = next((r for r in data["records"]["chain"] if r["pdf"] == pick), {})
    ev_m = data["evals"]["monolithic"].get(pick, {})
    ev_c = data["evals"]["chain"].get(pick, {})

    c1, c2 = st.columns(2)
    for col, (label, rec) in zip((c1, c2), (("monolithic", rec_m), ("chain", rec_c))):
        meta = rec.get("meta") or {}
        steps = meta.get("steps") or {}
        stp = "  ".join(f"{k[:12]}:{'✓' if s.get('ok') else '✗'} {s.get('attempts')}a {s.get('latency')}s"
                        for k, s in steps.items())
        col.metric(label, f"{meta.get('latency', '?')} s",
                   f"{meta.get('total_tokens', '?')} tok · {meta.get('attempts', '?')} attempts")
        if stp:
            col.caption(stp)
        if rec.get("classification"):
            col.caption("s1: " + json.dumps(rec["classification"]))

    st.markdown(_doc_table(gt, rec_m, rec_c, ev_m, ev_c), unsafe_allow_html=True)
    st.caption("legend: ✅ correct · abstain  🟡 partial  ❌ wrong  ⬜ missed  ➕ extra (red = value absent from source text)  – not scored")


def page_raw(data: dict):
    recs = data["records"]["monolithic"] + data["records"]["chain"]
    recs = [r for r in recs]
    opts = [f"{r['pdf']}  [{r['approach']}]" for r in recs]
    pick = st.selectbox("record", opts)
    r = next(x for x in recs if f"{x['pdf']}  [{x['approach']}]" == pick)
    st.json({"ok": r.get("ok"), "meta": r.get("meta"), "final": r.get("final")})


def main():
    st.title("Document parser comparison")
    st.caption("monolithic long prompt vs prompt chaining · "
               f"model {config.MODEL} · {config.BASE_URL} · thinking off, temp 0")

    runs = run_dirs()
    if not runs:
        st.info(f"No runs found under {config.RUNS_DIR}. "
                "Run `doc_env/bin/python main.py all --docs all` first.")
        return
    pick = st.sidebar.selectbox("run", runs, index=len(runs) - 1)
    data = load_run(pick)
    gts = load_gt_all()
    man = data["manifest"]
    st.sidebar.caption(f"{man.get('docs') and len(man.get('docs', [])) or '?'} docs · "
                       f"{man.get('created', '?')}")

    page = st.sidebar.radio("page", ["summary", "side-by-side", "raw output"])
    if page == "summary":
        page_summary(data, gts)
    elif page == "side-by-side":
        page_sides(data, gts)
    else:
        page_raw(data)


if __name__ == "__main__":
    main()

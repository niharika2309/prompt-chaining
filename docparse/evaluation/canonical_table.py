"""Build the canonical 2-way comparison table for the evaluation report.

Data assembly (all 50 docs, same model/endpoint/day):
  - quality      : 20261003_192116_full50 session (finals are byte-identical
                   to the quiet morning run — the temp-0 model is deterministic;
                   verified 50/50 cross-run)
  - latency      : QUIET morning session 20261003_081135_full50 — monolithic
                   measured; chain taken from chain_quiet_steps.json (per-doc
                   s1_classify + s2_extract latencies from that same quiet
                   session, extracted before the 3-step verify version was
                   removed). The evening session's raw latencies are
                   queue-noise-dominated (spiky shared endpoint) and are NOT
                   used for the latency table.
  - tokens       : load-invariant; taken from the evening run (identical to morning)

Prints the table + paired significance (Wilcoxon / bootstrap CI) as JSON to
stdout or --out.

Usage:
    doc_env/bin/python -m docparse.evaluation.canonical_table --out reports/canonical_2way.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from ..core import config

RUN_QUIET = config.RUNS_DIR / "20261003_081135_full50"
RUN_FULL = config.RUNS_DIR / "20261003_192116_full50"


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def by_pdf(recs):
    return {r["pdf"]: r for r in recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    monq = by_pdf(load(RUN_QUIET / "monolithic.jsonl"))
    mon3 = by_pdf(load(RUN_FULL / "monolithic.jsonl"))
    ch = by_pdf(load(RUN_FULL / "chain.jsonl"))
    steps = json.loads((RUN_QUIET / "chain_quiet_steps.json").read_text())["docs"]
    ev_mon = {r["pdf"]: r for r in json.loads((RUN_FULL / "eval_monolithic.json").read_text())}
    ev_ch = {r["pdf"]: r for r in json.loads((RUN_FULL / "eval_chain.json").read_text())}

    pdfs = sorted(mon3)
    assert set(ch) == set(monq) == set(mon3) == set(pdfs) and set(steps) == set(pdfs), \
        "pdf sets differ"

    # quiet-window latency per doc for each approach
    lat = {
        "monolithic": np.array([monq[p]["meta"]["latency"] for p in pdfs]),
        "chain": np.array([steps[p]["s1_classify"] + steps[p]["s2_extract"] for p in pdfs]),
    }
    # tokens: evening run (load-invariant; identical to morning)
    d2 = {"monolithic": mon3, "chain": ch}
    tok = {a: np.array([d2[a][p]["meta"]["total_tokens"] for p in pdfs]) for a in d2}
    ptok = {a: np.array([d2[a][p]["meta"]["prompt_tokens"] for p in pdfs]) for a in d2}
    ctok = {a: np.array([d2[a][p]["meta"]["completion_tokens"] for p in pdfs]) for a in d2}

    # quality from verdicts (evening session; deterministic == quiet)
    def quality(ev):
        out = {}
        for p in pdfs:
            vs = [v for v in ev[p]["verdicts"] if v["verdict"] != "n/a"]
            n = len(vs)
            out[p] = {
                "field_acc": sum(1 for v in vs if v["verdict"] in ("correct", "abstain")) / n if n else 0.0,
                "dirty": sum(1 for v in vs if v["verdict"] == "wrong" or
                             (v["verdict"] == "extra" and v.get("fabricated"))),
                "recall_den": sum(1 for v in vs if v["verdict"] != "abstain" and v.get("g")),
                "recall": sum(1.0 if v["verdict"] == "correct" else
                              (0.5 if v["verdict"] == "partial" else 0)
                              for v in vs if v["verdict"] != "abstain" and v.get("g")),
            }
        return out

    q = {"monolithic": quality(ev_mon), "chain": quality(ev_ch)}

    out = {"n_docs": len(pdfs), "summary": {}, "paired": {}}
    for a in ("monolithic", "chain"):
        s = q[a]
        out["summary"][a] = {
            "lat_mean": round(float(lat[a].mean()), 2), "lat_median": round(float(np.median(lat[a])), 2),
            "lat_p95": round(float(np.percentile(lat[a], 95)), 2),
            "tok_mean": round(float(tok[a].mean())),
            "prompt_mean": round(float(ptok[a].mean())),
            "completion_mean": round(float(ctok[a].mean())),
            "field_acc": round(float(np.mean([s[p]["field_acc"] for p in pdfs])), 4),
            "recall": round(float(
                np.sum([s[p]["recall"] for p in pdfs])
                / max(1, np.sum([s[p]["recall_den"] for p in pdfs]))), 4),
            "dirty_total": int(np.sum([s[p]["dirty"] for p in pdfs])),
        }

    base = "monolithic"
    a = "chain"
    pr = {}
    # latency (quiet window, paired)
    d = lat[a] - lat[base]
    w = stats.wilcoxon(lat[a], lat[base])
    pr["latency"] = {"mean_diff": round(float(d.mean()), 2),
                     "wilcoxon_p": round(float(w.pvalue), 4),
                     "a_faster_docs": int((d < 0).sum()), "a_slower_docs": int((d > 0).sum()),
                     "a_mean": round(float(lat[a].mean()), 2), "base_mean": round(float(lat[base].mean()), 2)}
    # tokens (paired)
    d = tok[a] - tok[base]
    w = stats.wilcoxon(tok[a], tok[base])
    pr["tokens"] = {"mean_diff": round(float(d.mean())),
                    "wilcoxon_p": round(float(w.pvalue), 4),
                    "a_fewer_docs": int((d < 0).sum()), "a_more_docs": int((d > 0).sum()),
                    "a_mean": round(float(tok[a].mean())), "base_mean": round(float(tok[base].mean()))}
    # field_acc (paired doc-level bootstrap)
    fa_a = np.array([q[a][p]["field_acc"] for p in pdfs])
    fa_b = np.array([q[base][p]["field_acc"] for p in pdfs])
    vals = np.column_stack([fa_b, fa_a])
    rng = np.random.default_rng(7)
    idx = np.arange(len(pdfs))
    diffs = []
    for _ in range(20000):
        s = rng.choice(idx, size=len(idx), replace=True)   # one shared draw: paired
        row = vals[s].mean(axis=0)
        diffs.append(row[1] - row[0])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pr["field_acc"] = {"mean_diff": round(float(diffs.mean()), 4),
                       "ci95": [round(float(lo), 4), round(float(hi), 4)],
                       "p_a_gt_base": round(float((diffs > 0).mean()), 4)}
    # dirty (paired)
    d_a = np.array([q[a][p]["dirty"] for p in pdfs])
    d_b = np.array([q[base][p]["dirty"] for p in pdfs])
    dd = d_a - d_b
    nz = int((dd != 0).sum())
    w = stats.wilcoxon(d_a, d_b) if nz else None
    pr["dirty"] = {"mean_diff": round(float(dd.mean()), 3),
                   "wilcoxon_p": round(float(w.pvalue), 4) if w is not None else None,
                   "a_cleaner_docs": int((dd < 0).sum()), "a_dirtier_docs": int((dd > 0).sum()),
                   "tied": len(pdfs) - nz}
    out["paired"] = pr

    s = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(s)
        print(f"wrote {args.out}")
    else:
        print(s)


if __name__ == "__main__":
    main()

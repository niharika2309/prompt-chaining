"""Deep comparison of the run's approaches (monolithic vs chain).

Approaches in a run dir are detected from the <name>.jsonl files present:
  monolithic  : 1 call, full 12-family spec
  chain       : 2 calls  classify -> extract (family-tailored spec)

Computes, per approach (paired by document where tests are applied):
  - headline quality (field acc / recall / precision / hallucination / clean docs)
    with doc-level bootstrap CIs and sign tests vs the monolithic baseline
  - efficiency (latency mean/p95, prompt/completion tokens, derived cost,
    derived throughput) with paired Wilcoxon tests
  - reliability (family classification, first-try validity, retries, parse ok,
    classification-confidence calibration)
  - chain step decomposition (s1/s2 latency + tokens)
  - token economics incl. the document re-send / spec-saving accounting
  - field verdict flips (each approach vs the monolithic baseline)
  - per-family dirty values, fabricated values (raw + tokenizer-fixed),
    family misclassifications, doc swings

Usage:
    doc_env/bin/python -m docparse.evaluation.analysis reports/runs/20261003_081135_full50 --out reports/analysis_full50.json

Prints a human-readable digest; --out writes the full results as JSON.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from ..core import config
from ..core import schema
from . import evaluate

ALL_APPROACHES = ("monolithic", "chain")
BASELINE = "monolithic"
LOWER_IS_BETTER = {"dirty", "latency", "tokens"}


# ---------------- loaders ----------------

def load_run(run_dir: Path) -> dict:
    run = {"manifest": json.loads((run_dir / "manifest.json").read_text()),
           "metrics": json.loads((run_dir / "metrics.json").read_text())
                      if (run_dir / "metrics.json").exists() else {}}
    for name in ALL_APPROACHES:
        p = run_dir / f"{name}.jsonl"
        if p.exists():
            recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            ev = (run_dir / f"eval_{name}.json")
            run[name] = recs
            run[name + "_eval"] = (json.loads(ev.read_text()) if ev.exists() else [])
    return run


def key(rec: dict) -> str:
    return rec["pdf"]


def gt_family_of(eval_row: dict | None) -> str:
    if eval_row:
        for v in eval_row["verdicts"]:
            if v["field"] == "document_family":
                return v.get("g") or "other"
    return "other"


def pair_docs(run: dict) -> dict[str, dict]:
    """Align all approaches per document."""
    by = {}
    for name in ALL_APPROACHES:
        if name not in run:
            continue
        by[name] = {key(r): r for r in run[name]}
        by[name + "_eval"] = {key(r): r for r in run[name + "_eval"]}
    pdfs = set()
    for name in ALL_APPROACHES:
        if name in by:
            pdfs |= set(by[name])
    docs = {}
    for pdf in sorted(pdfs):
        d = {"gt_fam": gt_family_of(by.get("chain_eval", {}).get(pdf) or
                                   by.get("monolithic_eval", {}).get(pdf))}
        for name in ALL_APPROACHES:
            if name in by:
                d[name] = by[name].get(pdf)
                d[name + "_eval"] = by[name + "_eval"].get(pdf, {"verdicts": []})
        docs[pdf] = d
    return docs


def has(d: dict, approach: str) -> bool:
    return d.get(approach) is not None and d.get(approach + "_eval") is not None


# ---------------- metrics ----------------

def _verdicts(d: dict, approach: str) -> list[dict]:
    return (d.get(approach + "_eval") or {}).get("verdicts", [])


def dirty_count(verdicts: list[dict]) -> int:
    return sum(1 for v in verdicts
               if v["verdict"] == "wrong" or (v["verdict"] == "extra" and v.get("fabricated")))


def m_dirty(d: dict, approach: str) -> float:
    return float(dirty_count(_verdicts(d, approach)))


def m_fieldacc(d: dict, approach: str) -> float:
    vs = [v for v in _verdicts(d, approach) if v["verdict"] != "n/a"]
    if not vs:
        return 0.0
    return sum(1 for v in vs if v["verdict"] in ("correct", "abstain")) / len(vs)


def m_recall(d: dict, approach: str) -> float:
    vs = [v for v in _verdicts(d, approach) if v["verdict"] != "n/a"]
    g = [v for v in vs if v["verdict"] != "abstain" and v.get("g")]
    if not g:
        return 0.0
    return sum(1.0 if v["verdict"] == "correct" else (0.5 if v["verdict"] == "partial" else 0)
               for v in g) / len(g)


def m_precision(d: dict, approach: str) -> float:
    vs = [v for v in _verdicts(d, approach) if v["verdict"] != "n/a"]
    m = [v for v in vs if v.get("m")]
    if not m:
        return 0.0
    return sum(1 for v in m if v["verdict"] in ("correct", "partial")) / len(m)


def m_halluc(d: dict, approach: str) -> float:
    vs = [v for v in _verdicts(d, approach) if v["verdict"] != "n/a"]
    m = [v for v in vs if v.get("m")]
    if not m:
        return 0.0
    return sum(1 for v in m if v["verdict"] == "extra" and v.get("fabricated")) / len(m)


def m_latency(d: dict, approach: str) -> float:
    return float((d.get(approach, {}).get("meta") or {}).get("latency") or 0)


def m_tokens(d: dict, approach: str) -> float:
    return float((d.get(approach, {}).get("meta") or {}).get("total_tokens") or 0)


METRICS = {
    "dirty": (m_dirty, True),
    "field_acc": (m_fieldacc, False),
    "recall": (m_recall, False),
    "precision": (m_precision, False),
    "halluc": (m_halluc, False),
    "latency": (m_latency, True),
    "tokens": (m_tokens, True),
}


def bootstrap_diff(docs: dict, approach: str, fn, n=20000, seed=7) -> dict:
    """Paired doc-level bootstrap of (approach - baseline), docs resampled."""
    pdfs = [p for p in docs if has(docs[p], approach) and has(docs[p], BASELINE)]
    if len(pdfs) < 2:
        return {"mean_diff": None, "ci95": None, "p_approach_gt_baseline": None}
    vals = np.array([[fn(docs[p], a) for a in (BASELINE, approach)] for p in pdfs])
    idx = np.arange(len(pdfs))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)   # one shared draw: paired
        row = vals[s].mean(axis=0)
        diffs.append(row[1] - row[0])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": round(float(diffs.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_approach_gt_baseline": round(float((diffs > 0).mean()), 4)}


# ---------------- token-artifact sensitivity ----------------

def _tokens_dotfree(s) -> list[str]:
    return [t.rstrip(".%,;:!?'\"") for t in
            re.findall(r"[a-z0-9%.]+", str(s).casefold())
            if t not in evaluate.STOP and len(t) > 2]


def recompute_fabricated(run: dict, approach: str, texts: dict[str, str]) -> dict:
    """Re-score 'extra' verdicts with the dot-free tokenizer (sensitivity)."""
    rows = run.get(approach + "_eval", [])
    fab_raw = fab_fixed = 0
    detail = []
    for r in rows:
        src = texts.get(r["pdf"], "")
        src_tok = set(_tokens_dotfree(src))
        for v in r["verdicts"]:
            if v["verdict"] != "extra":
                continue
            vt = [t for t in _tokens_dotfree(v.get("m") or "") if len(t) > 2]
            sup = True if not vt else sum(1 for t in vt if t in src_tok) / len(vt) >= 0.5
            fab_raw += int(v.get("fabricated"))
            if not sup:
                fab_fixed += 1
                detail.append({"pdf": r["pdf"], "field": v["field"], "value": (v.get("m") or "")[:60]})
    return {"fabricated_raw": fab_raw, "fabricated_dotfree": fab_fixed, "detail": detail}


# ---------------- analysis ----------------

def analyze(run: dict, texts: dict[str, str] | None = None) -> dict:
    docs = pair_docs(run)
    n = len(docs)
    names = [a for a in ALL_APPROACHES if any(has(d, a) for d in docs.values())]
    if texts is None:
        texts = {}
    out = {"n_docs": n, "approaches": names, "headline": {}, "paired_vs_baseline": {},
           "reliability": {}, "efficiency": {}, "steps": {}, "token_economics": {},
           "field_flips": {}, "per_family": {}, "misclassifications": {},
           "confidence": {}, "fabricated": {}, "doc_swings": []}

    # ---- headline per approach (recomputed from verdicts) ----
    for a in names:
        ev_rows = [d[a + "_eval"] for d in docs.values() if has(d, a)]
        vs = [v for r in ev_rows for v in r["verdicts"] if v["verdict"] != "n/a"]
        vc = Counter(v["verdict"] for v in vs)
        gt_n = sum(1 for v in vs if v["verdict"] != "abstain" and v.get("g"))
        m_n = sum(1 for v in vs if v.get("m"))
        fab = sum(1 for v in vs if v["verdict"] == "extra" and v.get("fabricated"))
        matches = vc["correct"] + vc["partial"]
        recs = [d[a] for d in docs.values() if has(d, a)]
        dirty_docs = {r["pdf"] for r in ev_rows
                      if any(v["verdict"] == "wrong" or (v["verdict"] == "extra" and v.get("fabricated"))
                             for v in r["verdicts"])}
        lat = np.array([r["meta"].get("latency") or 0 for r in recs])
        tok = np.array([r["meta"].get("total_tokens") or 0 for r in recs])
        ptok = np.array([r["meta"].get("prompt_tokens") or 0 for r in recs])
        ctok = np.array([r["meta"].get("completion_tokens") or 0 for r in recs])
        ok = sum(1 for r in recs if r.get("ok"))
        out["headline"][a] = {
            "n": len(recs), "parsed_ok": ok,
            "scored_fields": len(vs), "verdict_counts": dict(vc),
            "field_acc": round((vc["correct"] + vc["abstain"]) / len(vs), 4) if vs else None,
            "recall": round((vc["correct"] + 0.5 * vc["partial"]) / gt_n, 4) if gt_n else None,
            "precision": round(matches / m_n, 4) if m_n else None,
            "hallucination_rate": round(fab / m_n, 4) if m_n else None,
            "clean_docs": f"{len(recs) - len(dirty_docs)}/{len(recs)}",
        }
        out["efficiency"][a] = {
            "latency_mean_s": round(float(lat.mean()), 2),
            "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
            "latency_max_s": round(float(lat.max()), 2),
            "tokens_mean": round(float(tok.mean())),
            "prompt_tokens_mean": round(float(ptok.mean())),
            "completion_tokens_mean": round(float(ctok.mean())),
        }

    # ---- paired vs baseline ----
    for a in names:
        if a == BASELINE:
            continue
        res = {}
        for mname, (fn, lower) in METRICS.items():
            both = [d for d in docs.values() if has(d, a) and has(d, BASELINE)]
            if not both:
                continue
            a_arr = np.array([fn(d, a) for d in both])
            b_arr = np.array([fn(d, BASELINE) for d in both])
            d = a_arr - b_arr
            nz = int((d != 0).sum())
            entry = {"baseline_mean": round(float(b_arr.mean()), 4),
                     "approach_mean": round(float(a_arr.mean()), 4),
                     "mean_diff": round(float(d.mean()), 4),
                     "n_approach_lower": int((d < 0).sum()),
                     "n_approach_higher": int((d > 0).sum()),
                     "n_tied": len(d) - nz}
            if lower:
                entry["n_approach_better"] = int((d < 0).sum())
            else:
                entry["n_approach_better"] = int((d > 0).sum())
            w = stats.wilcoxon(a_arr, b_arr) if nz else None
            entry["wilcoxon_p"] = round(float(w.pvalue), 4) if w is not None else None
            if mname in ("dirty", "field_acc", "recall", "precision", "halluc"):
                entry.update(bootstrap_diff(docs, a, fn))
            res[mname] = entry
        out["paired_vs_baseline"][a] = res

    # ---- reliability ----
    for a in names:
        recs = [d[a] for d in docs.values() if has(d, a)]
        steps = ((recs[0].get("meta") or {}).get("steps") or {}) if recs else {}
        if "s2_extract" in steps:  # chain
            dec_src = "classification"
            first = [1 for r in recs
                     if ((r.get("meta") or {}).get("steps") or {}).get("s2_extract", {}).get("attempts") == 1]
        else:
            dec_src = "final"
            first = [1 for r in recs if (r.get("meta") or {}).get("attempts") == 1]
        att = np.array([(r.get("meta") or {}).get("attempts") or 0 for r in recs])
        fam_ok = 0
        mis = []
        for d in docs.values():
            if not has(d, a):
                continue
            dec = ((d[a].get(dec_src) or {}).get("document_family") or "other")
            if dec not in schema.FAMILIES:
                dec = "other"
            if dec == d["gt_fam"]:
                fam_ok += 1
            else:
                mis.append({"pdf": d[a]["pdf"], "declared": dec, "gt": d["gt_fam"]})
        fam_n = sum(1 for d in docs.values() if has(d, a))
        out["reliability"][a] = {
            "family_acc": f"{fam_ok}/{fam_n} ({fam_ok / fam_n:.0%})" if fam_n else None,
            "first_try": f"{sum(first)}/{fam_n}" if fam_n else None,
            "first_try_acc": round(sum(first) / fam_n, 3) if fam_n else None,
            "attempts_mean": round(float(att.mean()), 2),
            "attempts_dist": dict(Counter(att)),
        }
        out["misclassifications"][a] = mis
        if dec_src == "classification":
            conf = {"correct": [], "incorrect": []}
            for d in docs.values():
                if not has(d, a):
                    continue
                c = d[a].get("classification") or {}
                dec = c.get("document_family") or "other"
                bucket = "correct" if dec == d["gt_fam"] else "incorrect"
                if isinstance(c.get("confidence"), (int, float)):
                    conf[bucket].append(float(c["confidence"]))
            out["confidence"][a] = {k: {"mean": round(float(np.mean(v)), 3), "n": len(v)}
                                    for k, v in conf.items() if v}

    # ---- step decomposition for chain-like approaches ----
    for a in names:
        skeys = [k for k in (([d[a] for d in docs.values() if has(d, a)][0]
                              .get("meta") or {}).get("steps") or {})] if any(has(d, a) for d in docs.values()) else []
        if not skeys:
            continue
        steps = {}
        recs = [d[a] for d in docs.values() if has(d, a)]
        tot_lat = np.array([r["meta"].get("latency") or 0 for r in recs])
        for s in skeys:
            lat = np.array([(r["meta"].get("steps") or {}).get(s, {}).get("latency") or 0 for r in recs])
            pt = np.array([(r["meta"].get("steps") or {}).get(s, {}).get("prompt_tokens") or 0 for r in recs])
            ct = np.array([(r["meta"].get("steps") or {}).get(s, {}).get("completion_tokens") or 0 for r in recs])
            at = np.array([(r["meta"].get("steps") or {}).get(s, {}).get("attempts") or 0 for r in recs])
            steps[s] = {"latency_mean_s": round(float(lat.mean()), 2),
                        "latency_pct_of_doc": round(float((lat / tot_lat.mean()).mean()), 3) if tot_lat.mean() else None,
                        "prompt_tokens_mean": round(float(pt.mean())),
                        "completion_tokens_mean": round(float(ct.mean())),
                        "attempts_mean": round(float(at.mean()), 2)}
        out["steps"][a] = steps

    # ---- token economics ----
    te = {}
    for a in names:
        m = out["efficiency"][a]
        te[a] = m | {"extra_total_tokens_vs_baseline": None,
                     "throughput_docs_per_hour_at_3_workers": round(3 * 3600 / m["latency_mean_s"], 0)}
    for a in names:
        if a != BASELINE and BASELINE in names:
            te[a]["extra_total_tokens_vs_baseline"] = (
                out["efficiency"][a]["tokens_mean"] - out["efficiency"][BASELINE]["tokens_mean"])
    out["token_economics"] = te

    # ---- field flips: each approach vs baseline ----
    def verdict_key(v: dict) -> str:
        if v["verdict"] == "extra" and v.get("fabricated"):
            return "fabricated"
        return v["verdict"]

    GOOD = {"correct", "partial", "abstain"}
    BAD = {"wrong", "missed", "fabricated"}

    def flips(a: str, b: str):
        """Field verdicts where approach `a` differs from approach `b`:
        fixed = a good where b bad/extra; broken = a bad/extra where b good."""
        fixed, broken = Counter(), Counter()
        for d in docs.values():
            if not (has(d, a) and has(d, b)):
                continue
            av = {v["field"]: v for v in _verdicts(d, a)}
            bv = {v["field"]: v for v in _verdicts(d, b)}
            for f in sorted(set(av) & set(bv)):  # sorted: set order is hash-randomized
                ka, kb = verdict_key(av[f]), verdict_key(bv[f])
                if ka == kb:
                    continue
                if (ka in GOOD and kb in BAD) or (ka in GOOD and kb == "extra"):
                    fixed[f] += 1
                elif (ka in BAD and kb in GOOD) or (ka == "extra" and kb in GOOD):
                    broken[f] += 1
        return fixed, broken

    for a in names:
        if a != BASELINE:
            fixed, broken = flips(a, BASELINE)
            out["field_flips"][f"{a}_vs_{BASELINE}"] = {
                "n_fixed": sum(fixed.values()), "n_broken": sum(broken.values()),
                "fixed": dict(fixed.most_common(8)), "broken": dict(broken.most_common(8))}
    # ---- per-family dirty ----
    pf = defaultdict(lambda: defaultdict(float))
    pfn = defaultdict(int)
    for d in docs.values():
        f = d["gt_fam"]
        pfn[f] += 1
        for a in names:
            if has(d, a):
                pf[f][a] += dirty_count(_verdicts(d, a))
    out["per_family"] = {
        f: {"n_docs": pfn[f],
            **{a: round(pf[f][a] / pfn[f], 2) for a in names if pf[f].get(a) is not None or a in pf[f]}}
        for f in sorted(pfn, key=lambda x: -pfn[x]) if pfn[f] >= 2
    }

    # ---- fabricated values (raw + dot-free sensitivity) ----
    for a in names:
        out["fabricated"][a] = recompute_fabricated(run, a, texts)

    # ---- doc swings ----
    for d in docs.values():
        vals = {a: dirty_count(_verdicts(d, a)) for a in names if has(d, a)}
        if len(set(vals.values())) > 1:
            any_pdf = next((d[a]["pdf"] for a in names if has(d, a)), "?")
            out["doc_swings"].append({"pdf": any_pdf, **vals})
    return out


# ---------------- reporting ----------------

def fmt(x, pct=False):
    if x is None:
        return "?"
    return f"{x:+.1%}" if pct else f"{x:+.4f}"


def digest(res: dict) -> str:
    L = []
    L.append(f"n_docs = {res['n_docs']}   approaches = {res['approaches']}")
    H, E, R = res["headline"], res["efficiency"], res["reliability"]
    cols = res["approaches"]
    L.append("\n== headline quality ==")
    L.append(f"  {'metric':18s}" + "".join(f" {c:>14s}" for c in cols))
    for k in ("field_acc", "recall", "precision", "hallucination_rate", "clean_docs"):
        L.append(f"  {k:18s}" + "".join(f" {H[c].get(k)!s:>14s}" for c in cols))
    L.append(f"  {'verdicts':18s}" + "".join(f" {H[c]['verdict_counts']!s:>14s}" for c in cols))
    L.append("\n== efficiency ==")
    for k in ("latency_mean_s", "latency_p95_s", "tokens_mean", "prompt_tokens_mean",
              "completion_tokens_mean"):
        L.append(f"  {k:18s}" + "".join(f" {E[c].get(k)!s:>14s}" for c in cols))
    L.append("\n== paired vs monolithic ==")
    for a, m in res["paired_vs_baseline"].items():
        for mk, r in m.items():
            line = f"  {a:11s} {mk:10s} base={r['baseline_mean']:<9.4f} appr={r['approach_mean']:<9.4f} diff={fmt(r['mean_diff'])}"
            if r.get("wilcoxon_p") is not None:
                line += f" wilcoxon_p={r['wilcoxon_p']}"
            if "ci95" in r:
                line += f" ci95=[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] p>base={r['p_approach_gt_baseline']}"
            line += f"  appr_better={r.get('n_approach_better')} higher={r['n_approach_higher']} lower={r['n_approach_lower']} tied={r['n_tied']}"
            L.append(line)
    L.append("\n== reliability ==")
    for c in cols:
        r = R[c]
        L.append(f"  {c:11s} family={r['family_acc']}  first_try={r['first_try']}  "
                 f"attempts={r['attempts_mean']} dist={r['attempts_dist']}")
    for c, v in res.get("confidence", {}).items():
        L.append(f"  {c:11s} s1 confidence {v}")
    L.append("\n== steps ==")
    for c, s in res["steps"].items():
        for sname, v in s.items():
            L.append(f"  {c:11s} {sname:14s} lat {v['latency_mean_s']}s ({v['latency_pct_of_doc']:.0%}) "
                     f"prompt {v['prompt_tokens_mean']} comp {v['completion_tokens_mean']} att {v['attempts_mean']}")
    L.append("\n== token economics ==")
    for c in cols:
        t = res["token_economics"][c]
        L.append(f"  {c:11s} total {t['tokens_mean']} (prompt {t['prompt_tokens_mean']} / completion "
                 f"{t['completion_tokens_mean']})  d_vs_base {t['extra_total_tokens_vs_baseline']}  "
                 f"throughput {t['throughput_docs_per_hour_at_3_workers']}/h")
    L.append("\n== field flips ==")
    for k, v in res["field_flips"].items():
        L.append(f"  {k:22s} fixed {v['n_fixed']}  broken {v['n_broken']}")
        if v["n_broken"]:
            L.append(f"      broken: {v['broken']}")
    L.append("\n== per-family dirty (mean/doc) ==")
    L.append(f"  {'family':15s}" + "".join(f" {c:>14s}" for c in cols))
    for f, v in res["per_family"].items():
        L.append(f"  {f:15s}" + "".join(f" {v.get(c, '?')!s:>14s}" for c in cols) + f"   (n={v['n_docs']})")
    L.append("\n== fabricated ==")
    for c in cols:
        f = res["fabricated"][c]
        L.append(f"  {c:11s} raw {f['fabricated_raw']}  dotfree {f['fabricated_dotfree']} "
                 f"{f['detail'][:3]}")
    L.append(f"\n== doc swings ({len(res['doc_swings'])} docs differ) ==")
    for s in res["doc_swings"]:
        L.append(f"  {s['pdf']}  " + "  ".join(f"{k}={v}" for k, v in s.items() if k != "pdf"))
    L.append("\n== family misclassifications ==")
    for c, rows in res["misclassifications"].items():
        L.append(f"  {c}: {len(rows)}" + ("; " + ", ".join(f"{r['pdf']} ({r['declared']} vs {r['gt']})"
                                                           for r in rows[:12]) if rows else ""))
    return "\n".join(L)


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_texts(pdfs) -> dict:
    out = {}
    for p in pdfs:
        c = config.CORPUS_DIR / (Path(p).stem + ".txt")
        if c.exists():
            out[p] = c.read_text()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = load_run(Path(args.run_dir))
    pdfs = {key(r) for a in ALL_APPROACHES if a in run for r in run[a]}
    res = analyze(run, load_texts(pdfs))
    print(digest(res))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(_py(res), indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

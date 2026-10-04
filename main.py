"""CLI for the monolithic-vs-chain document parser comparison.

  python main.py extract  [--docs pilot|all|f1.pdf,f2.pdf]
  python main.py run      --approach both --docs pilot [--concurrency 3] [--tag pilot]
  python main.py evaluate --run reports/runs/<dir>
  python main.py report   --run reports/runs/<dir>
  python main.py all      --docs pilot [--concurrency 3]

Run dir layout: reports/runs/<ts>_<tag>/{<approach>.jsonl, eval_*.json,
summary.md, side_by_side.html, manifest.json}
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from docparse.core import config
from docparse.evaluation import evaluate
from docparse.pipeline import chain, extract_text, monolithic
from docparse.reporting import report

APPROACHES = {
    "monolithic": monolithic,
    "chain": chain,              # 2-step: classify -> extract
}


def select_docs(spec: str) -> list[str]:
    if spec == "pilot":
        return config.PILOT_DOCS
    if spec == "all":
        return sorted(p.name for p in config.PDF_DIR.glob("*.pdf"))
    names = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [n for n in names if not (config.PDF_DIR / n).exists()]
    if missing:
        sys.exit(f"PDFs not found: {missing}")
    return names


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file; also tolerates a single pretty-printed JSON array."""
    if not path.exists():
        return []
    text = path.read_text()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def load_corpus(names: list[str]) -> dict[str, str]:
    out = {}
    for n in names:
        c = config.CORPUS_DIR / (Path(n).stem + ".txt")
        if c.exists():
            out[n] = c.read_text()
    return out


def load_gt(names: list[str]) -> dict[str, dict]:
    out = {}
    with open(config.GT_CSV) as f:
        for row in csv.DictReader(f):
            if row["pdf_filename"] in names:
                out[row["pdf_filename"]] = row
    return out


def run_approach(approach: str, names: list[str], texts: dict[str, str],
                 out_path: Path, concurrency: int, quiet: bool = False):
    mod = APPROACHES[approach]
    records: list[dict] = []
    t0 = time.monotonic()
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(mod.parse, texts[n], n): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = {"approach": approach, "pdf": n, "ok": False, "final": None,
                       "meta": {"error": repr(e)}}
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if not quiet:
                status = "ok " if rec["ok"] else "FAIL"
                extra = ""
                if rec["meta"].get("steps"):
                    st = rec["meta"]["steps"]
                    extra = " [" + ", ".join(f"{k[:6]}={'ok' if v['ok'] else 'FAIL'}"
                                              for k, v in st.items()) + "]"
                print(f"  [{status}] {n} {rec['meta'].get('latency', '?')}s "
                      f"{rec['meta'].get('total_tokens', '?')}tok{extra}")
    print(f"  {approach}: {sum(r['ok'] for r in records)}/{len(records)} ok "
          f"in {time.monotonic() - t0:.0f}s")
    return records


def cmd_extract(args):
    names = select_docs(args.docs)
    print(f"Extracting {len(names)} PDFs -> {config.CORPUS_DIR}")
    extract_text.extract_all(names)
    print("done.")


def cmd_run(args):
    names = select_docs(args.docs)
    texts = extract_text.extract_all(names)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.RUNS_DIR / f"{ts}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "tag": args.tag, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.MODEL, "base_url": config.BASE_URL,
        "enable_thinking": config.ENABLE_THINKING,
        "temperature": config.TEMPERATURE, "max_tokens": config.MAX_TOKENS,
        "docs": names,
    }, indent=2))

    if args.approach == "both":
        approaches = ["monolithic", "chain"]
    else:
        approaches = [args.approach]
    for a in approaches:
        print(f"Running {a} on {len(names)} docs (concurrency {args.concurrency})")
        run_approach(a, names, texts, run_dir / f"{a}.jsonl", args.concurrency)
    print(f"\nRun dir: {run_dir}")
    return run_dir


def cmd_evaluate(args):
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = config.ROOT / run_dir

    # Load every approach present in the run dir (order = canonical).
    names = [a for a in ("monolithic", "chain")
             if (run_dir / f"{a}.jsonl").exists()]
    recs_by = {a: load_jsonl(run_dir / f"{a}.jsonl") for a in names}
    all_recs = [r for a in names for r in recs_by[a]]
    gts = load_gt([r["pdf"] for r in all_recs])
    texts = load_corpus(list(gts))

    metrics = {}
    for name in names:
        recs = recs_by[name]
        if not recs:
            continue
        rows = []
        for r in recs:
            gt = gts.get(r["pdf"], {})
            src = texts.get(r["pdf"], "")
            rows.append(evaluate.eval_record(r, gt, src))
        (run_dir / f"eval_{name}.json").write_text(json.dumps(rows, indent=1))
        metrics[name] = evaluate.full_metrics(recs, rows)
        print(f"\n=== {name} ===")
        for k, v in metrics[name].items():
            print(f"  {k:20s} {v}")

    if metrics:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        if len(metrics) >= 2:
            keys = ("field_acc", "recall", "precision", "hallucination_rate",
                    "clean_docs", "mean_latency", "mean_tokens", "first_try")
            head = f"  {'metric':22s}" + "".join(f" {a:>12s}" for a in metrics)
            print("\n=== comparison ===" + head)
            for k in keys:
                print(f"  {k:22s}" + "".join(f" {str(metrics[a].get(k)):>12s}" for a in metrics))
    return run_dir


def cmd_report(args):
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = config.ROOT / run_dir
    mon = load_jsonl(run_dir / "monolithic.jsonl")
    chain = load_jsonl(run_dir / "chain.jsonl")
    gts = load_gt([r["pdf"] for r in mon + chain])
    mon_eval = load_jsonl(run_dir / "eval_monolithic.json")
    chain_eval = load_jsonl(run_dir / "eval_chain.json")
    metrics = {}
    for name in ("monolithic", "chain"):
        if (run_dir / f"eval_{name}.json").exists():
            recs = mon if name == "monolithic" else chain
            metrics[name] = evaluate.full_metrics(recs, load_jsonl(run_dir / f"eval_{name}.json"))
    p1 = report.write_markdown(run_dir, mon, chain, gts, mon_eval, chain_eval,
                               metrics.get("monolithic", {}), metrics.get("chain", {}))
    p2 = report.write_html(run_dir, mon, chain, gts, mon_eval, chain_eval,
                           metrics.get("monolithic", {}), metrics.get("chain", {}))
    print(f"wrote {p1}\nwrote {p2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_docs(p):
        p.add_argument("--docs", default="pilot",
                       help="pilot | all | comma-separated pdf filenames")

    pe = sub.add_parser("extract")
    add_docs(pe)
    pr = sub.add_parser("run")
    add_docs(pr)
    pr.add_argument("--approach", choices=["monolithic", "chain", "both"],
                    default="both")
    pr.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    pr.add_argument("--tag", default="run")
    pv = sub.add_parser("evaluate")
    pv.add_argument("--run", required=True)
    prp = sub.add_parser("report")
    prp.add_argument("--run", required=True)
    pa = sub.add_parser("all")
    add_docs(pa)
    pa.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    pa.add_argument("--tag", default="run")

    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "all":
        args.docs, args.approach = args.docs, "both"
        run_dir = cmd_run(args)
        args.run = str(run_dir)
        cmd_evaluate(args)
        cmd_report(args)


if __name__ == "__main__":
    main()

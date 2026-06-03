"""Aggregate multi-seed train_full.py JSON logs into mean ± std.

Each log has the shape:
    {"train_loss": [...], "val_F": [...], "val_sAP10": [...], "epochs": [...]}

We extract per-run:
    best_F = max(val_F)
    sAP10_at_best_F = val_sAP10[argmax(val_F)]
    final_F = val_F[-1]
    final_sAP10 = val_sAP10[-1]

Then mean ± std across seeds.
"""
import argparse, glob, json, statistics, sys, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"No files match {args.pattern!r}")

    rows = []
    for f in files:
        with open(f) as fh:
            log = json.load(fh)
        seed = None
        m = re.search(r"seed(\d+)", f)
        if m:
            seed = int(m.group(1))
        val_F = log.get("val_F", [])
        val_sAP10 = log.get("val_sAP10", [])
        if not val_F or not val_sAP10:
            print(f"[skip] {f}: missing val_F/val_sAP10", file=sys.stderr)
            continue
        i_best = max(range(len(val_F)), key=lambda i: val_F[i])
        rows.append({
            "file": f,
            "seed": seed,
            "best_F": val_F[i_best],
            "sAP10_at_best_F": val_sAP10[i_best],
            "final_F": val_F[-1],
            "final_sAP10": val_sAP10[-1],
            "epochs": len(val_F),
        })

    if not rows:
        sys.exit("No valid logs.")

    summary = {}
    for k in ("best_F", "sAP10_at_best_F", "final_F", "final_sAP10"):
        vals = [r[k] for r in rows]
        summary[k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "values": vals,
        }

    out = {
        "n_runs": len(rows),
        "per_run": rows,
        "summary": summary,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {args.out}: {len(rows)} runs aggregated.")
    for k, v in summary.items():
        print(f"  {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()

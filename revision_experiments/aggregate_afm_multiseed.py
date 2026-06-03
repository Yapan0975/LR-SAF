"""Aggregate AFM YU-80 fine-tune across seeds {42, 17, 2024}."""
import argparse, glob, json, statistics, sys, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="logs/revision/afm_yu80_seed*.json")
    ap.add_argument("--out", default="logs/revision/afm_multiseed_aggregate.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit(f"No files match {args.pattern!r}")

    rows = []
    for f in files:
        with open(f) as fh:
            log = json.load(fh)
        m = re.search(r"seed(\d+)", f)
        seed = int(m.group(1)) if m else None
        F = log["val_F"]; A = log["val_sAP10"]
        i_best = max(range(len(F)), key=lambda i: F[i])
        rows.append({"file": f, "seed": seed,
                     "best_F": F[i_best], "sAP10_at_best_F": A[i_best],
                     "final_F": F[-1], "final_sAP10": A[-1]})

    summary = {}
    for k in ("best_F", "sAP10_at_best_F"):
        v = [r[k] for r in rows]
        summary[k] = {"mean": statistics.fmean(v),
                      "std": statistics.stdev(v) if len(v) > 1 else 0.0,
                      "n": len(v), "values": v}

    out = {"n_runs": len(rows), "per_run": rows, "summary": summary}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"AFM YU-80 fine-tune across {len(rows)} seeds:")
    for k, v in summary.items():
        print(f"  {k:18s}: {v['mean']:.4f} ± {v['std']:.4f}   values={[f'{x:.4f}' for x in v['values']]}")


if __name__ == "__main__":
    main()

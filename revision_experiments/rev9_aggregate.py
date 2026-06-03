"""Aggregate rev9 batch results: no-TNNR multi-seed + v9 + v10."""
import glob, json, statistics


def best_F_sap(path):
    with open(path) as fh:
        d = json.load(fh)
    F = d["val_F"]; A = d["val_sAP10"]
    i = max(range(len(F)), key=lambda k: F[k])
    return F[i], A[i], F[-1], A[-1]


def main():
    print("=== no-TNNR multi-seed ===")
    files = sorted(glob.glob("logs/revision/no_tnnr_seed*.json"))
    rows = []
    for f in files:
        bF, bA, fF, fA = best_F_sap(f)
        seed = f.split("seed")[-1].split(".")[0]
        rows.append((seed, bF, bA))
        print(f"  seed={seed}: best F={bF:.4f}  sAP10={bA:.4f}")
    if rows:
        Fs = [r[1] for r in rows]; As = [r[2] for r in rows]
        print(f"  mean: F={statistics.fmean(Fs):.4f} ± {statistics.stdev(Fs) if len(Fs)>1 else 0:.4f}, "
              f"sAP10={statistics.fmean(As):.4f} ± {statistics.stdev(As) if len(As)>1 else 0:.4f}")

    print("\n=== EXP-C LR-SAF (with TNNR) multi-seed ===")
    files = sorted(glob.glob("logs/revision/train_seed*.json"))
    lr_rows = []
    for f in files:
        bF, bA, fF, fA = best_F_sap(f)
        seed = f.split("seed")[-1].split(".")[0]
        lr_rows.append((seed, bF, bA))
        print(f"  seed={seed}: best F={bF:.4f}  sAP10={bA:.4f}")
    if lr_rows:
        Fs = [r[1] for r in lr_rows]; As = [r[2] for r in lr_rows]
        print(f"  mean: F={statistics.fmean(Fs):.4f} ± {statistics.stdev(Fs) if len(Fs)>1 else 0:.4f}, "
              f"sAP10={statistics.fmean(As):.4f} ± {statistics.stdev(As) if len(As)>1 else 0:.4f}")

    print("\n=== Δ_TNNR_backbone (per-seed paired) ===")
    by_seed_no = {s: (F, A) for s, F, A in rows}
    by_seed_lr = {s: (F, A) for s, F, A in lr_rows}
    deltas = []
    for s in by_seed_lr:
        if s not in by_seed_no: continue
        dF = by_seed_lr[s][0] - by_seed_no[s][0]
        dA = by_seed_lr[s][1] - by_seed_no[s][1]
        deltas.append((s, dF, dA))
        print(f"  seed={s}: ΔF (LR-SAF − no-TNNR) = {dF:+.4f}, ΔsAP10 = {dA:+.4f}")
    if deltas:
        dFs = [d[1] for d in deltas]; dAs = [d[2] for d in deltas]
        print(f"  mean ΔF   = {statistics.fmean(dFs):+.4f} ± {statistics.stdev(dFs) if len(dFs)>1 else 0:.4f}")
        print(f"  mean ΔsAP = {statistics.fmean(dAs):+.4f} ± {statistics.stdev(dAs) if len(dAs)>1 else 0:.4f}")

    print("\n=== v9 (K=1 + bounded encoding, seed 42) ===")
    for f in sorted(glob.glob("logs/revision/v9_*.json")):
        bF, bA, fF, fA = best_F_sap(f)
        print(f"  {f}: best F={bF:.4f}  sAP10={bA:.4f}  | final F={fF:.4f}  sAP10={fA:.4f}")

    print("\n=== v10 (TV regularizer, seed 42) ===")
    for f in sorted(glob.glob("logs/revision/v10_*.json")):
        bF, bA, fF, fA = best_F_sap(f)
        print(f"  {f}: best F={bF:.4f}  sAP10={bA:.4f}  | final F={fF:.4f}  sAP10={fA:.4f}")


if __name__ == "__main__":
    main()

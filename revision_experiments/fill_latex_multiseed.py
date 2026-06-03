"""Fill the multi-seed LaTeX table placeholders from exp_c_aggregate.json.

Usage on remote:
    python3 revision_experiments/fill_latex_multiseed.py \
        --agg logs/revision/exp_c_aggregate.json \
        --template /home/server/Documents/yping/LR-SAF-LSD/latex/sections/07b_multiseed_to_paste.tex \
        --out      /home/server/Documents/yping/LR-SAF-LSD/latex/sections/07b_multiseed.tex
"""
import argparse, json, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True, help="exp_c_aggregate.json path")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.agg) as fh:
        d = json.load(fh)
    runs = sorted(d["per_run"], key=lambda r: r["seed"])
    by_seed = {r["seed"]: r for r in runs}
    F_sum = d["summary"]["best_F"]
    sap_sum = d["summary"]["sAP10_at_best_F"]

    with open(args.template) as fh:
        src = fh.read()

    def fmt(v):
        return f"{v:.4f}"

    rep = {
        r"\\todo{S42_F}": fmt(by_seed[42]["best_F"]) if 42 in by_seed else "—",
        r"\\todo{S42_sAP}": fmt(by_seed[42]["sAP10_at_best_F"]) if 42 in by_seed else "—",
        r"\\todo{S17_F}": fmt(by_seed[17]["best_F"]) if 17 in by_seed else "—",
        r"\\todo{S17_sAP}": fmt(by_seed[17]["sAP10_at_best_F"]) if 17 in by_seed else "—",
        r"\\todo{S24_F}": fmt(by_seed[2024]["best_F"]) if 2024 in by_seed else "—",
        r"\\todo{S24_sAP}": fmt(by_seed[2024]["sAP10_at_best_F"]) if 2024 in by_seed else "—",
        r"\\todo{MEAN_F}": fmt(F_sum["mean"]),
        r"\\todo{MEAN_sAP}": fmt(sap_sum["mean"]),
        r"\\todo{STD_F}": fmt(F_sum["std"]),
        r"\\todo{STD_sAP}": fmt(sap_sum["std"]),
    }
    for pat, val in rep.items():
        src = re.sub(pat, val, src)

    with open(args.out, "w") as fh:
        fh.write(src)
    print(f"Wrote {args.out}")
    print(f"Multi-seed YorkUrban-80 fine-tune ({F_sum['n']} seeds):")
    print(f"  best F   : {F_sum['mean']:.4f} ± {F_sum['std']:.4f}")
    print(f"  sAP-10   : {sap_sum['mean']:.4f} ± {sap_sum['std']:.4f}")


if __name__ == "__main__":
    main()

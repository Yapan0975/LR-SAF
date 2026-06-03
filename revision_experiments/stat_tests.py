"""Welch's t-test on the LR-SAF vs AFM F-measure and sAP-10 gap.

No scipy dependency: implements Welch + paired t-test with the t-pdf
integrated numerically via composite trapezoidal rule.

Usage:
    python3 stat_tests.py 'logs/revision/train_seed*.json' \
                          'logs/revision/afm_yu80_seed*.json'
"""
import glob, json, math, statistics, sys
from math import gamma, pi, sqrt


def best_F_and_sap(path):
    with open(path) as fh:
        d = json.load(fh)
    F = d["val_F"]; A = d["val_sAP10"]
    i = max(range(len(F)), key=lambda i: F[i])
    return F[i], A[i]


def t_pdf(x, df):
    return gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2)) \
        * (1 + x * x / df) ** (-(df + 1) / 2)


def two_sided_p(t, df):
    if not math.isfinite(t):
        return float("nan")
    a, b = abs(t), abs(t) + 12.0
    n = 2000
    h = (b - a) / n
    s = 0.5 * (t_pdf(a, df) + t_pdf(b, df))
    for k in range(1, n):
        s += t_pdf(a + k * h, df)
    return max(min(2 * s * h, 1.0), 0.0)


def welch_t(xs, ys):
    nx, ny = len(xs), len(ys)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    vx = statistics.variance(xs) if nx > 1 else 0.0
    vy = statistics.variance(ys) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return mx - my, float("inf"), float("nan"), float("nan")
    t = (mx - my) / se
    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx / nx) ** 2 / max(nx - 1, 1) + (vy / ny) ** 2 / max(ny - 1, 1)
    df = df_num / df_den if df_den > 0 else nx + ny - 2
    return (mx - my), t, df, two_sided_p(t, df)


def paired_t(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    diffs = [a - b for a, b in zip(xs, ys)]
    n = len(diffs)
    md = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    if se == 0:
        return md, float("inf"), n - 1, float("nan")
    t = md / se
    df = n - 1
    return md, t, df, two_sided_p(t, df)


def main():
    lr_files = sorted(glob.glob(sys.argv[1]))
    afm_files = sorted(glob.glob(sys.argv[2]))
    print(f"LR-SAF files: {lr_files}")
    print(f"AFM files:    {afm_files}")
    lr = [best_F_and_sap(f) for f in lr_files]
    afm = [best_F_and_sap(f) for f in afm_files]
    F_lr, sap_lr = [x[0] for x in lr], [x[1] for x in lr]
    F_afm, sap_afm = [x[0] for x in afm], [x[1] for x in afm]
    print(f"  LR-SAF F:   {F_lr}")
    print(f"  AFM F:      {F_afm}")
    print(f"  LR-SAF sAP: {sap_lr}")
    print(f"  AFM sAP:    {sap_afm}")

    print()
    for name, xs, ys in (("F", F_lr, F_afm), ("sAP-10", sap_lr, sap_afm)):
        md, t, df, p = welch_t(xs, ys)
        print(f"Welch {name:<6}: Delta={md:+.4f}  t={t:.3f}  df={df:.2f}  p={p:.3f}")
        pr = paired_t(xs, ys)
        if pr:
            md, t, df, p = pr
            print(f"Paired {name:<5}: Delta={md:+.4f}  t={t:.3f}  df={df}  p={p:.3f}")
        print()


if __name__ == "__main__":
    main()

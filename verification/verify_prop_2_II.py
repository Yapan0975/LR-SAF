"""
Numerical verification of Proposition 2-II:
  ||A_x||_{2K,*} <= C(Omega, K) * Delta_max,  where C(Omega, K) = sqrt(Omega - 2K) * Omega

Run: python3 verify_prop_2_II.py
"""
import numpy as np

np.random.seed(42)


def line_perp_normal(x_s, x_e):
    """Unit perpendicular vector to line segment."""
    d = x_e - x_s
    L = np.linalg.norm(d)
    e_para = d / L
    return np.array([-e_para[1], e_para[0]])


def afm_value(p, x_s, n_perp):
    """A_x component at pixel p for a line with perpendicular normal n_perp."""
    d_perp = np.dot(p - x_s, n_perp)
    return -n_perp[0] * d_perp


def random_K_junction_window(Omega, K, junction=None):
    """Construct A_x for a K-junction window.

    Returns: A_x [Omega, Omega], list of lines, Delta_max
    """
    if junction is None:
        junction = np.array([(Omega - 1) / 2.0, (Omega - 1) / 2.0])

    # K random line directions through junction
    angles = np.sort(np.random.uniform(0, 2 * np.pi, K))
    lines = []
    for a in angles:
        x_s = junction
        x_e = junction + 100 * np.array([np.cos(a), np.sin(a)])
        lines.append((x_s, x_e, line_perp_normal(x_s, x_e)))

    # Each pixel: pick the closest line (Voronoi)
    A_x = np.zeros((Omega, Omega), dtype=np.float64)
    deltas_all = []   # list of |B_j - B_k| at each pixel for all j != k
    for i in range(Omega):
        for j in range(Omega):
            p = np.array([float(i), float(j)])
            # find closest line
            dists = []
            vals = []
            for (x_s, x_e, n) in lines:
                d_perp = abs(np.dot(p - x_s, n))
                dists.append(d_perp)
                vals.append(afm_value(p, x_s, n))
            k_star = np.argmin(dists)
            A_x[i, j] = vals[k_star]
            # Delta at this pixel = max difference among the line values
            for a in range(K):
                for b in range(a + 1, K):
                    deltas_all.append(abs(vals[a] - vals[b]))
    Delta_max = max(deltas_all) if deltas_all else 0.0
    return A_x, lines, Delta_max


def tail_nuclear_norm(M, r):
    """||M||_{r, *} = sum of singular values beyond index r."""
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.sum(s[r:]))


def main():
    print("=== Verify Proposition 2-II ===\n")
    n_trials = 200
    for Omega in [9, 11, 15]:
        for K in [2, 3]:
            ratios = []
            for trial in range(n_trials):
                A_x, lines, Delta_max = random_K_junction_window(Omega, K)
                empirical = tail_nuclear_norm(A_x, 2 * K)
                C = np.sqrt(max(Omega - 2 * K, 0)) * Omega
                bound = C * Delta_max
                if bound > 0:
                    ratios.append(empirical / bound)
                else:
                    ratios.append(0.0)

            ratios = np.array(ratios)
            print(f"Omega={Omega}, K={K}: "
                  f"C(Omega,K)={np.sqrt(max(Omega-2*K,0))*Omega:.2f}, "
                  f"empirical/bound ratio: "
                  f"mean={ratios.mean():.3f}, max={ratios.max():.3f}, "
                  f"violations (>1): {int((ratios > 1).sum())}/{n_trials}")

    print("\n=== Sanity: axis-aligned should give EXACTLY rank <= 2K ===")
    # Two horizontal lines at y=Omega/3 and y=2Omega/3, parallel
    Omega = 11
    A_x = np.zeros((Omega, Omega))
    for i in range(Omega):
        for j in range(Omega):
            p = np.array([float(i), float(j)])
            # two parallel horizontal lines, perp = (0, 1)
            n = np.array([0.0, 1.0])
            # closer line determines value
            y1, y2 = Omega / 3, 2 * Omega / 3
            d1 = abs(j - y1); d2 = abs(j - y2)
            if d1 < d2:
                A_x[i, j] = afm_value(p, np.array([0.0, y1]), n)
            else:
                A_x[i, j] = afm_value(p, np.array([0.0, y2]), n)
    s = np.linalg.svd(A_x, compute_uv=False)
    print(f"  Two parallel horizontal lines, top-6 SVs: {np.round(s[:6], 3)}")
    print(f"  rank = {int((s > 1e-9).sum())} (axis-aligned, expect <= 2K = 4)")


if __name__ == '__main__':
    main()

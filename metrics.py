"""
LR-SAF: Evaluation metrics for line segment detection.

Implements:
  - F-measure (pixel-level, AFM-style with diagonal-relative tolerance)
  - sAP-K  (segment-level, LCNN/HAWP-style on 128x128 normalized image)
  - msAP  (mean over K in {5, 10, 15})
  - junction PR (placeholder, returns None if junctions not available)

Reference:
  AFM:  Xue et al., CVPR 2019, Sec 5.1
  sAP:  Zhou et al. (LCNN), ICCV 2019
"""
import numpy as np


# -----------------------------------------------------------------------------
# Pixel-level F-measure (AFM-style)
# -----------------------------------------------------------------------------
def rasterize_lines(lines, H, W):
    """Draw line segments to a binary H x W image.
       lines: [N, 4] (x1, y1, x2, y2). Returns set of (x, y) pixel coordinates.
    """
    pixels = set()
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        steps = max(int(np.ceil(steps)), 1)
        for k in range(steps + 1):
            t = k / steps
            x = int(round(x1 + t * dx))
            y = int(round(y1 + t * dy))
            if 0 <= x < W and 0 <= y < H:
                pixels.add((x, y))
    return pixels


def f_measure(pred_lines, gt_lines, H, W, tol_frac=0.01):
    """Pixel-level F-measure with diagonal-relative tolerance.
       tol_frac of image diagonal (default 0.01 per AFM paper).
    """
    if len(pred_lines) == 0:
        return {'P': 0.0, 'R': 0.0, 'F': 0.0,
                'TP_p': 0, 'TP_r': 0, 'N_pred': 0, 'N_gt': len(gt_lines)}
    if len(gt_lines) == 0:
        return {'P': 0.0, 'R': 0.0, 'F': 0.0,
                'TP_p': 0, 'TP_r': 0, 'N_pred': len(pred_lines), 'N_gt': 0}

    diag = np.sqrt(H * H + W * W)
    tol = tol_frac * diag

    pred_pix = rasterize_lines(pred_lines, H, W)
    gt_pix = rasterize_lines(gt_lines, H, W)
    pred_arr = np.array(list(pred_pix), dtype=np.float32)    # [Np, 2]
    gt_arr = np.array(list(gt_pix), dtype=np.float32)        # [Ng, 2]
    if len(pred_arr) == 0 or len(gt_arr) == 0:
        return {'P': 0.0, 'R': 0.0, 'F': 0.0,
                'TP_p': 0, 'TP_r': 0,
                'N_pred': len(pred_arr), 'N_gt': len(gt_arr)}

    # Use chunked KDTree-like via numpy broadcasting (small images: OK)
    # For each pred pixel, find min distance to any gt pixel.
    def chunked_min_dist(A, B, chunk=2048):
        """min over j of ||A_i - B_j||  for i in [|A|]."""
        out = np.full(len(A), np.inf, dtype=np.float32)
        for s in range(0, len(B), chunk):
            Bc = B[s:s + chunk]
            d = np.sqrt(((A[:, None, :] - Bc[None, :, :]) ** 2).sum(-1))  # [|A|, chunk]
            out = np.minimum(out, d.min(axis=1))
        return out

    d_p2g = chunked_min_dist(pred_arr, gt_arr)
    d_g2p = chunked_min_dist(gt_arr, pred_arr)

    tp_p = int((d_p2g <= tol).sum())
    tp_r = int((d_g2p <= tol).sum())

    P = tp_p / max(len(pred_arr), 1)
    R = tp_r / max(len(gt_arr), 1)
    F = 2 * P * R / max(P + R, 1e-12)
    return {'P': P, 'R': R, 'F': F,
            'TP_p': tp_p, 'TP_r': tp_r,
            'N_pred': len(pred_arr), 'N_gt': len(gt_arr)}


# -----------------------------------------------------------------------------
# Structural AP (sAP), LCNN-style
# -----------------------------------------------------------------------------
def normalize_to_128(lines, H, W):
    """Normalize line endpoints to a 128 x 128 frame (sAP convention)."""
    out = np.asarray(lines, dtype=np.float32).copy()
    out[:, 0::2] = out[:, 0::2] / W * 128.0
    out[:, 1::2] = out[:, 1::2] / H * 128.0
    return out


def pairwise_segment_dist_sq(pred, gt):
    """Sum of squared endpoint distances (with optional swap) between every pair.
       pred: [Np, 4], gt: [Ng, 4]. Returns [Np, Ng] of min(d1, d2) where d1,d2 are
       the two endpoint pairings.
    """
    # endpoint A: (x1,y1) -> dist to (gx1,gy1); B: (x2,y2) -> (gx2,gy2)
    p1 = pred[:, None, :2]       # [Np, 1, 2]
    p2 = pred[:, None, 2:]
    g1 = gt[None, :, :2]
    g2 = gt[None, :, 2:]

    d_aa = ((p1 - g1) ** 2).sum(-1) + ((p2 - g2) ** 2).sum(-1)   # straight order
    d_ab = ((p1 - g2) ** 2).sum(-1) + ((p2 - g1) ** 2).sum(-1)   # swapped order
    return np.minimum(d_aa, d_ab)   # [Np, Ng]


def s_ap(pred_lines, scores, gt_lines, H, W, thresholds=(5, 10, 15)):
    """Structural AP. pred sorted internally by score descending.
       thresholds: pixel thresholds in the 128x128 normalized frame.
       Returns dict {threshold -> AP}.
    """
    pred_lines = np.asarray(pred_lines, dtype=np.float32)
    gt_lines = np.asarray(gt_lines, dtype=np.float32)
    if len(pred_lines) == 0 or len(gt_lines) == 0:
        return {t: 0.0 for t in thresholds}

    # Sort predictions by score descending (higher score = more confident)
    order = np.argsort(-np.asarray(scores))
    pred_lines = pred_lines[order]

    pred_n = normalize_to_128(pred_lines, H, W)
    gt_n = normalize_to_128(gt_lines, H, W)

    d2 = pairwise_segment_dist_sq(pred_n, gt_n)  # [Np, Ng]

    results = {}
    for thr in thresholds:
        # LCNN/HAWP sAP: threshold is applied DIRECTLY to the summed squared
        # endpoint distance d2 (NOT thr**2). sAP-10 == d2 <= 10.
        gt_used = np.zeros(len(gt_n), dtype=bool)
        tp = np.zeros(len(pred_n), dtype=np.float32)
        fp = np.zeros(len(pred_n), dtype=np.float32)
        for i in range(len(pred_n)):
            # one-to-one greedy: assign to the nearest still-unused GT within thr
            cand = np.where((d2[i] <= thr) & (~gt_used))[0]
            if len(cand) > 0:
                j = cand[np.argmin(d2[i, cand])]
                tp[i] = 1.0
                gt_used[j] = True
            else:
                fp[i] = 1.0

        # Compute AP from cumulative PR
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(len(gt_n), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        # 11-point interpolation (PASCAL VOC 2007 style)
        ap = 0.0
        for r_thr in np.linspace(0, 1, 11):
            p_at_r = precision[recall >= r_thr]
            ap += (p_at_r.max() if len(p_at_r) > 0 else 0.0) / 11
        # Alternative (better): integrated AP
        ap_integ = 0.0
        prec_interp = np.maximum.accumulate(precision[::-1])[::-1]
        rec_diff = np.diff(np.concatenate([[0.0], recall]))
        ap_integ = float((rec_diff * prec_interp).sum())
        # Take the integrated version as standard for sAP
        results[thr] = ap_integ
    return results


def msap(pred_lines, scores, gt_lines, H, W):
    """mean sAP over thresholds 5, 10, 15 (per-image; prefer sap_dataset)."""
    aps = s_ap(pred_lines, scores, gt_lines, H, W, thresholds=(5, 10, 15))
    return sum(aps.values()) / len(aps), aps


def image_records(pred_lines, scores, gt_lines, H, W):
    """Return a per-image record for dataset-level sAP: predictions and GT
    normalized to the 128 frame, plus scores. Use with sap_dataset()."""
    pred_lines = np.asarray(pred_lines, dtype=np.float32)
    gt_lines = np.asarray(gt_lines, dtype=np.float32)
    rec = {'pred': normalize_to_128(pred_lines, H, W) if len(pred_lines) else
                   np.zeros((0, 4), np.float32),
           'score': np.asarray(scores, dtype=np.float32).reshape(-1),
           'gt': normalize_to_128(gt_lines, H, W) if len(gt_lines) else
                 np.zeros((0, 4), np.float32)}
    return rec


def sap_dataset(records, thresholds=(5, 10, 15)):
    """Dataset-level structural AP (the standard LCNN/HAWP protocol).

    records: list of per-image dicts from image_records() (pred/gt in the
    128 frame, score per prediction). All predictions across all images are
    pooled, sorted once by global confidence, and a single PR curve / AP is
    computed per threshold. Matching is one-to-one greedy within each image
    (each GT matched at most once), threshold applied to the summed squared
    endpoint distance. Returns {threshold -> AP}.
    """
    n_gt_total = int(sum(len(r['gt']) for r in records))
    out = {}
    for thr in thresholds:
        scores_all, tp_all = [], []
        for r in records:
            pred, score, gt = r['pred'], r['score'], r['gt']
            if len(pred) == 0:
                continue
            order = np.argsort(-score)
            pred_s = pred[order]
            score_s = score[order]
            if len(gt) == 0:
                scores_all.append(score_s)
                tp_all.append(np.zeros(len(pred_s), np.float32))
                continue
            d2 = pairwise_segment_dist_sq(pred_s, gt)   # [N, M]
            gt_used = np.zeros(len(gt), dtype=bool)
            tp = np.zeros(len(pred_s), np.float32)
            for i in range(len(pred_s)):
                cand = np.where((d2[i] <= thr) & (~gt_used))[0]
                if len(cand) > 0:
                    gt_used[cand[np.argmin(d2[i, cand])]] = True
                    tp[i] = 1.0
            scores_all.append(score_s)
            tp_all.append(tp)
        if not scores_all:
            out[thr] = 0.0
            continue
        scores_all = np.concatenate(scores_all)
        tp_all = np.concatenate(tp_all)
        idx = np.argsort(-scores_all)
        tp_cum = np.cumsum(tp_all[idx])
        fp_cum = np.cumsum(1.0 - tp_all[idx])
        recall = tp_cum / max(n_gt_total, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        prec_env = np.maximum.accumulate(precision[::-1])[::-1]
        rec_diff = np.diff(np.concatenate([[0.0], recall]))
        out[thr] = float((rec_diff * prec_env).sum())
    return out


# -----------------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Synthetic test: predictions == GT + small noise -> high F and sAP
    np.random.seed(0)
    gt = np.array([[10, 10, 90, 90], [50, 10, 50, 90], [10, 50, 90, 50]],
                  dtype=np.float32)
    noise = np.random.randn(*gt.shape) * 0.5
    pred = gt + noise
    scores = np.ones(len(pred))

    fm = f_measure(pred, gt, H=100, W=100)
    print(f"identity-ish: F={fm['F']:.3f} (P={fm['P']:.3f}, R={fm['R']:.3f})")

    aps = s_ap(pred, scores, gt, H=100, W=100)
    print(f"identity-ish sAP-5={aps[5]:.3f}, sAP-10={aps[10]:.3f}, sAP-15={aps[15]:.3f}")

    # Empty prediction
    fm0 = f_measure(np.zeros((0, 4)), gt, H=100, W=100)
    print(f"empty pred: F={fm0['F']:.3f}")

    # Half-correct: predictions only contains first 2 GT lines
    pred_half = gt[:2]
    scores_h = np.ones(2)
    aps_h = s_ap(pred_half, scores_h, gt, H=100, W=100)
    print(f"half preds: sAP-10={aps_h[10]:.3f} (expect ~0.5 since 2/3 GT covered)")

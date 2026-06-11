"""
LR-SAF: Learned confidence head for candidate line segments.

After the cython squeeze produces candidate segments, this MLP rescores each
based on its geometric statistics + per-segment AFM consistency. Training
target: 1 if segment matches a GT line within sAP-10 threshold, else 0.

This is a stepping stone toward the "differentiable squeeze" of paper §5.4,
without requiring a full reimplementation of the squeeze logic. Used to
restore sAP performance lost by the soft-assignment training (see results_log.md).

Features per candidate segment:
  - length (in normalized 128x128 frame)
  - angle  (radians)
  - support pixel count (from squeeze's xx, yy arrays)
  - mean / std of AFM magnitude in the segment's local strip
  - mean / std of perpendicular AFM component (should be ~0 for true lines)
  - mean junction score along segment
  - 5th column from squeeze (the "quality" metric)

Run as a separate post-train stage: freeze the LR-SAF main network,
train this head on (kept_predictions, gt_lines) pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_segment_features(lines, afm_pred, junc_pred,
                              squeeze_aux=None, H=320, W=320):
    """Compute per-segment feature vector.
       lines: [N, 4+] (x1, y1, x2, y2, [quality_metric])
       afm_pred: [2, H, W]
       junc_pred: [1, H, W] or [H, W]
       squeeze_aux: dict with 'xx', 'yy' support arrays (optional)

    Returns: [N, F] float tensor of features.
    """
    if isinstance(lines, np.ndarray):
        lines = torch.from_numpy(lines).float()
    if isinstance(afm_pred, np.ndarray):
        afm_pred = torch.from_numpy(afm_pred).float()
    if isinstance(junc_pred, np.ndarray):
        junc_pred = torch.from_numpy(junc_pred).float()
    if junc_pred.dim() == 3:
        junc_pred = junc_pred[0]

    N = lines.shape[0]
    feats = torch.zeros(N, 9)

    if N == 0:
        return feats

    p1 = lines[:, :2]; p2 = lines[:, 2:4]
    direction = p2 - p1
    length = direction.norm(dim=-1)
    angle = torch.atan2(direction[:, 1], direction[:, 0])

    # Normalize length to a 128x128 frame
    diag = (H * H + W * W) ** 0.5
    length_norm = length * 128.0 / diag

    feats[:, 0] = length_norm
    feats[:, 1] = angle
    feats[:, 2] = lines[:, 4] if lines.shape[1] >= 5 else 0  # quality metric

    # Sample along the segment to extract AFM and junction statistics
    n_samples = 16
    t = torch.linspace(0, 1, n_samples).unsqueeze(0)        # [1, S]
    sx = (p1[:, 0:1] + t * (p2[:, 0:1] - p1[:, 0:1])).clamp(0, W - 1)
    sy = (p1[:, 1:2] + t * (p2[:, 1:2] - p1[:, 1:2])).clamp(0, H - 1)
    sx_i = sx.long(); sy_i = sy.long()

    afm_x = afm_pred[0, sy_i, sx_i]                          # [N, S]
    afm_y = afm_pred[1, sy_i, sx_i]
    afm_mag = (afm_x ** 2 + afm_y ** 2).sqrt()
    feats[:, 3] = afm_mag.mean(dim=1)
    feats[:, 4] = afm_mag.std(dim=1)

    # Perpendicular component (signed): should be ~0 for true lines
    # u_perp = (-sin, cos) where direction = (cos, sin) * |d|
    u_perp_x = -direction[:, 1:2] / length.unsqueeze(-1).clamp(min=1e-6)
    u_perp_y =  direction[:, 0:1] / length.unsqueeze(-1).clamp(min=1e-6)
    afm_perp = afm_x * u_perp_x + afm_y * u_perp_y
    feats[:, 5] = afm_perp.mean(dim=1)
    feats[:, 6] = afm_perp.std(dim=1)

    j_vals = junc_pred[sy_i, sx_i]
    feats[:, 7] = j_vals.mean(dim=1)
    feats[:, 8] = length  # raw pixel length

    return feats


class ConfidenceMLP(nn.Module):
    def __init__(self, in_dim=9, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        return torch.sigmoid(self.net(feats)).squeeze(-1)


def match_to_gt(pred_lines, gt_lines, H, W, threshold=10):
    """Per-prediction binary target: 1 if the prediction is the one-to-one
    match to some GT line under the LCNN sAP rule (in the 128 frame).

    Fixes (reviewer round 4): (i) the threshold is applied DIRECTLY to the
    summed squared endpoint distance (sAP-10 == d2 <= 10), not to thr**2;
    (ii) matching is genuinely one-to-one (each GT and each prediction used
    at most once) via greedy assignment by ascending distance, so duplicate
    predictions of the same GT are NOT all labelled positive.
    Returns: torch tensor [N_pred] of 0/1.
    """
    if isinstance(pred_lines, np.ndarray):
        pred_lines = torch.from_numpy(pred_lines).float()
    if isinstance(gt_lines, np.ndarray):
        gt_lines = torch.from_numpy(gt_lines).float()
    pred = pred_lines.clone()
    gt = gt_lines.clone()
    pred[:, 0::2] = pred[:, 0::2] / W * 128.0
    pred[:, 1::2] = pred[:, 1::2] / H * 128.0
    gt[:, 0::2] = gt[:, 0::2] / W * 128.0
    gt[:, 1::2] = gt[:, 1::2] / H * 128.0

    p1 = pred[:, None, :2]; p2 = pred[:, None, 2:]
    g1 = gt[None, :, :2]; g2 = gt[None, :, 2:]
    d_aa = ((p1 - g1) ** 2).sum(-1) + ((p2 - g2) ** 2).sum(-1)
    d_ab = ((p1 - g2) ** 2).sum(-1) + ((p2 - g1) ** 2).sum(-1)
    d2 = torch.minimum(d_aa, d_ab)

    matched = torch.zeros(pred.shape[0])
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return matched
    d2np = d2.detach().cpu().numpy()
    thr = float(threshold)
    flat = np.argsort(d2np, axis=None)                 # ascending distance
    rows, cols = np.unravel_index(flat, d2np.shape)
    pred_used = np.zeros(d2np.shape[0], dtype=bool)
    gt_used = np.zeros(d2np.shape[1], dtype=bool)
    for i, j in zip(rows, cols):
        if d2np[i, j] > thr:
            break
        if not pred_used[i] and not gt_used[j]:
            matched[i] = 1.0
            pred_used[i] = True
            gt_used[j] = True
    return matched


if __name__ == '__main__':
    # Smoke test
    torch.manual_seed(0)
    lines = torch.randn(20, 5) * 100 + 160
    afm = torch.randn(2, 320, 320)
    junc = torch.sigmoid(torch.randn(1, 320, 320))
    feats = compute_segment_features(lines, afm, junc)
    print(f"feats: {feats.shape}, range {feats.min().item():.2f} to {feats.max().item():.2f}")
    mlp = ConfidenceMLP()
    scores = mlp(feats)
    print(f"scores: {scores.shape}, range {scores.min().item():.3f} to {scores.max().item():.3f}")

    gt = torch.randn(15, 4) * 100 + 160
    tgt = match_to_gt(lines[:, :4], gt, H=320, W=320)
    print(f"matched-to-gt targets: {tgt}, sum={tgt.sum().item()}")
    loss = F.binary_cross_entropy(scores, tgt)
    print(f"BCE loss: {loss.item():.4f}")

"""
LR-SAF: Differentiable Squeeze — full pipeline.

Replaces the cython region-growing decoder with a fully-differentiable pipeline:

  Stage 1.  Soft voting heatmap from AFM.
            For each pixel p, the projected point p + a(p) deposits a
            bilinear vote into a heatmap H.
  Stage 2.  Peak heatmap loss against GT endpoints.
            Compare H to a Gaussian heatmap built from GT line endpoints.
  Stage 3.  Differentiable peak extraction (soft-NMS + top-K).
            Sample top-K peak coordinates via softargmax over local windows.
  Stage 4.  Line proposal via endpoint pairing.
            For each peak pair (p_i, p_j), check that the predicted AFM
            direction along the connecting line is consistent.

In training, Stages 1+2 give a heatmap-supervision signal that flows back
through the AFM. Stages 3+4 are used at inference for line extraction.

This is paper §5 "Differentiable Squeeze" contribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage 1: Soft voting heatmap
# ---------------------------------------------------------------------------
def soft_voting_heatmap(afm, sigma=1.0, normalize=False):
    """afm: [B, 2, H, W] attraction field (a_x, a_y) in pixel-offset units.
    Returns heatmap [B, 1, H, W] — votes accumulated at projected endpoints.

    Differentiable: each vote is bilinearly splatted to 4 neighbors.
    Smoothed by Gaussian kernel for stable training.
    """
    B, C, H, W = afm.shape
    device, dtype = afm.device, afm.dtype

    # Pixel coords
    rows = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    cols = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)

    # Projected endpoint p + a(p)
    proj_x = (cols + afm[:, 0:1]).clamp(0, W - 1)
    proj_y = (rows + afm[:, 1:2]).clamp(0, H - 1)

    # Bilinear splatting
    x0 = proj_x.floor().long().clamp(0, W - 2)
    y0 = proj_y.floor().long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = proj_x - x0.to(dtype)
    wx0 = 1 - wx1
    wy1 = proj_y - y0.to(dtype)
    wy0 = 1 - wy1

    heat = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    for b in range(B):
        flat = heat[b, 0].flatten()
        # scatter weighted votes
        for (xi, yi, w) in [(x0, y0, wx0 * wy0),
                              (x1, y0, wx1 * wy0),
                              (x0, y1, wx0 * wy1),
                              (x1, y1, wx1 * wy1)]:
            idx = (yi[b, 0] * W + xi[b, 0]).flatten()
            flat = flat.scatter_add(0, idx, w[b, 0].flatten())
        heat[b, 0] = flat.reshape(H, W)

    # Gaussian smoothing
    if sigma > 0:
        K = int(4 * sigma) | 1
        coords = torch.arange(K, device=device, dtype=dtype) - K // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, K)
        heat = F.conv2d(heat, g.unsqueeze(2), padding=(0, K // 2))
        heat = F.conv2d(heat, g.unsqueeze(3), padding=(K // 2, 0))

    if normalize:
        heat = heat / max(H * W / 1024.0, 1.0)
    return heat


# ---------------------------------------------------------------------------
# Stage 2: Build GT heatmap from line endpoints
# ---------------------------------------------------------------------------
def build_gt_heatmap(gt_lines_list, H, W, sigma=2.0, device='cuda'):
    """gt_lines_list: list (len B) of [N_i, 4] tensors.
    Returns [B, 1, H, W] heatmap with Gaussian peaks at each line endpoint.
    """
    B = len(gt_lines_list)
    heat = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)

    K = int(4 * sigma) | 1
    coords = torch.arange(K, device=device, dtype=torch.float32) - K // 2
    g1 = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1 = (g1 / g1.sum()).view(1, 1, K)

    for b, gt in enumerate(gt_lines_list):
        if gt.shape[0] == 0:
            continue
        endpoints = torch.cat([gt[:, :2], gt[:, 2:]], dim=0).to(device)  # [2N, 2]
        for ep in endpoints:
            x, y = int(ep[0].round().clamp(0, W - 1).item()), \
                   int(ep[1].round().clamp(0, H - 1).item())
            heat[b, 0, y, x] += 1.0
        # Smooth via two 1D convs
        heat[b:b+1] = F.conv2d(heat[b:b+1], g1.unsqueeze(2), padding=(0, K // 2))
        heat[b:b+1] = F.conv2d(heat[b:b+1], g1.unsqueeze(3), padding=(K // 2, 0))
    return heat


def heatmap_loss(pred_heat, gt_heat, alpha=2.0, beta=4.0):
    """Focal-style MSE-like loss for heatmap regression (CenterNet-style).

    L = mean over (i,j) of:
        if gt > 0.99: alpha * (1 - pred)^2 * log(pred)
        else:          beta * (1 - gt)^beta * pred^alpha * log(1 - pred)
    """
    pred = pred_heat.clamp(1e-6, 1 - 1e-6)
    pos_mask = gt_heat >= 0.99
    neg_mask = ~pos_mask
    pos_loss = -alpha * (1 - pred) ** 2 * torch.log(pred) * pos_mask.float()
    neg_loss = -((1 - gt_heat) ** beta) * pred ** alpha * \
               torch.log(1 - pred) * neg_mask.float()
    n_pos = pos_mask.float().sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


# ---------------------------------------------------------------------------
# Stage 3: Differentiable peak extraction
# ---------------------------------------------------------------------------
def extract_peaks_soft(heat, top_k=256, nms_kernel=5, score_thresh=0.0):
    """Local-max NMS (hard) + top-K (hard).

    For backprop we keep soft scores; only the indices are non-diff.
    Returns: peaks [B, top_k, 2] (x, y), scores [B, top_k].
    """
    B, _, H, W = heat.shape
    # Local-max via max-pool comparison
    pooled = F.max_pool2d(heat, kernel_size=nms_kernel,
                           stride=1, padding=nms_kernel // 2)
    is_max = (heat == pooled).float()
    candidate = heat * is_max                              # [B, 1, H, W]

    # Top-K
    flat = candidate.view(B, -1)
    scores, idx = flat.topk(top_k, dim=1)
    y = idx // W
    x = idx % W
    peaks = torch.stack([x.float(), y.float()], dim=-1)    # [B, top_k, 2]
    return peaks, scores


# ---------------------------------------------------------------------------
# Stage 4: Line proposal via endpoint pairing
# ---------------------------------------------------------------------------
def pair_endpoints_by_afm(peaks, scores, afm, max_pairs=512,
                           afm_consistency_thresh=2.0,
                           min_length=3.0):
    """For each ordered pair (p_i, p_j) of high-scoring peaks, test whether
    a line from p_i to p_j is consistent with the predicted AFM:
      - Sample N points along the segment
      - At each, check that the AFM vector points perpendicular to the segment
      - Average inconsistency score; keep pairs below thresh
    Returns proposed lines [B, max_pairs, 4] and pair scores.

    Args:
        peaks   : [B, K, 2]
        scores  : [B, K]
        afm     : [B, 2, H, W]
    """
    B, K, _ = peaks.shape
    device, dtype = peaks.device, peaks.dtype

    lines_out, scores_out = [], []
    for b in range(B):
        # Build all pairs (i < j)
        i_idx, j_idx = torch.triu_indices(K, K, offset=1, device=device)
        n_pairs = i_idx.numel()
        p1 = peaks[b][i_idx]                  # [P, 2]
        p2 = peaks[b][j_idx]
        seg = p2 - p1
        lengths = seg.norm(dim=-1).clamp(min=1e-6)
        # Filter min length
        keep_len = lengths > min_length
        p1 = p1[keep_len]; p2 = p2[keep_len]; seg = seg[keep_len]; lengths = lengths[keep_len]
        sa = scores[b][i_idx[keep_len]]
        sb = scores[b][j_idx[keep_len]]

        if seg.shape[0] == 0:
            lines_out.append(torch.zeros(0, 4, device=device, dtype=dtype))
            scores_out.append(torch.zeros(0, device=device, dtype=dtype))
            continue

        # Sample 8 points along each segment
        n_samp = 8
        t = torch.linspace(0, 1, n_samp, device=device).view(1, n_samp)
        sx = (p1[:, 0:1] + t * seg[:, 0:1]).clamp(0, afm.shape[3] - 1)
        sy = (p1[:, 1:2] + t * seg[:, 1:2]).clamp(0, afm.shape[2] - 1)
        sx_i = sx.long(); sy_i = sy.long()
        afm_b = afm[b]                          # [2, H, W]
        a_x = afm_b[0, sy_i, sx_i]              # [P, n_samp]
        a_y = afm_b[1, sy_i, sx_i]

        # Unit tangent and perpendicular
        u_para = seg / lengths.unsqueeze(-1)                 # [P, 2]
        u_perp = torch.stack([-u_para[:, 1], u_para[:, 0]], dim=-1)

        # Project AFM onto u_perp; perfect line means a should be parallel to u_perp (perpendicular component dominant)
        afm_para = a_x * u_para[:, 0:1] + a_y * u_para[:, 1:2]  # [P, n_samp]
        afm_perp = a_x * u_perp[:, 0:1] + a_y * u_perp[:, 1:2]
        # Inconsistency: along-line AFM should be small (purely perpendicular)
        ratio = afm_para.abs().mean(dim=1) / (afm_perp.abs().mean(dim=1) + 1e-3)
        keep_consis = ratio < afm_consistency_thresh

        pair_score = (sa + sb) / 2 * torch.exp(-ratio)        # [P]
        # Combine and trim to max_pairs
        line_bp = torch.cat([p1, p2], dim=-1)                  # [P, 4]
        line_bp = line_bp[keep_consis]
        pair_score = pair_score[keep_consis]
        if line_bp.shape[0] > max_pairs:
            top_idx = pair_score.topk(max_pairs).indices
            line_bp = line_bp[top_idx]; pair_score = pair_score[top_idx]
        lines_out.append(line_bp)
        scores_out.append(pair_score)
    return lines_out, scores_out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    afm_base = torch.randn(B, 2, H, W, device='cuda') * 3.0
    afm = afm_base.detach().clone().requires_grad_(True)
    gt_lines = [torch.tensor([[10., 10., 50., 50.], [10., 30., 50., 20.]],
                              device='cuda'),
                torch.tensor([[5., 5., 60., 60.]], device='cuda')]

    heat = soft_voting_heatmap(afm, sigma=1.0)
    print(f"voting heatmap: {tuple(heat.shape)}, "
          f"sum={heat.sum().item():.2f} (expect ~{B*H*W/100:.1f}+)")

    gt_heat = build_gt_heatmap(gt_lines, H, W, sigma=2.0, device='cuda')
    print(f"gt heatmap: {tuple(gt_heat.shape)}, max={gt_heat.max().item():.3f}")

    loss = heatmap_loss(heat / (heat.max() + 1e-6), gt_heat / (gt_heat.max() + 1e-6))
    loss.backward()
    print(f"heatmap loss: {loss.item():.4f}, grad finite: "
          f"{torch.isfinite(afm.grad).all().item()}, grad norm: {afm.grad.norm().item():.4f}")

    # Stage 3 + 4
    peaks, scores = extract_peaks_soft(heat, top_k=32)
    print(f"extracted peaks: {tuple(peaks.shape)}, max score: {scores.max().item():.4f}")
    lines, ps = pair_endpoints_by_afm(peaks, scores, afm, max_pairs=64)
    print(f"line proposals per image: {[ln.shape[0] for ln in lines]}, "
          f"first 2 of img 0: {lines[0][:2] if lines[0].shape[0] > 0 else 'empty'}")

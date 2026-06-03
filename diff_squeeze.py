"""
LR-SAF: Differentiable Squeeze (sketch, to be expanded for paper §5.4).

Conceptually replaces the cython greedy region-grow with a soft, differentiable
voting pipeline:

  1. Forward AFM:  for each pixel p, v(p) = p + a(p)   [projected endpoint]
  2. Bilinear splat votes into a heatmap H(q) over the image lattice.
     H(q) = sum over p of  K(v(p), q)  where K is a 2D bilinear kernel.
  3. Peaks of H correspond to line endpoints (where many pixels project).
  4. Pair peaks into segments via attraction field connectivity.

For now we implement Steps 1-2 (soft voting), which is the gradient-bearing
part. Step 3-4 (peak detection + linking) remain the same NMS as before but
can be made differentiable later via differentiable-NMS approaches.

Usage:
    H = soft_voting_heatmap(afm_pred, sigma=1.0)
    # Loss can be computed against a target heatmap derived from GT endpoints.
"""
import torch
import torch.nn.functional as F


def soft_voting_heatmap(afm, sigma=1.0, normalize=True):
    """Soft-vote each pixel onto its projected endpoint location.

    Args:
        afm   : [B, 2, H, W] attraction field in pixel-offset units (decoded if encoded).
                channel 0 = x-offset (column), channel 1 = y-offset (row).
        sigma : Gaussian smoothing kernel std (in pixels) applied to the vote map.
        normalize: if True, divide by H*W so the heatmap scale is independent of size.

    Returns:
        heat : [B, 1, H, W] vote density.
    """
    B, C, H, W = afm.shape
    assert C == 2, "AFM must have 2 channels (a_x, a_y)"
    device = afm.device

    # Pixel grid (col, row) in float
    rows = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1).expand(B, 1, H, W)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W).expand(B, 1, H, W)

    # Projected endpoint per pixel:
    proj_x = cols + afm[:, 0:1]                                # [B, 1, H, W]
    proj_y = rows + afm[:, 1:2]

    # Clip projections to valid range
    proj_x = proj_x.clamp(0, W - 1)
    proj_y = proj_y.clamp(0, H - 1)

    # Bilinear splatting via grid_sample's adjoint (use scatter_add manually)
    heat = torch.zeros(B, 1, H, W, device=device)

    x0 = proj_x.floor().long().clamp(0, W - 2)
    y0 = proj_y.floor().long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = (proj_x - x0.float())                                # in [0,1]
    wx0 = 1 - wx1
    wy1 = (proj_y - y0.float())
    wy0 = 1 - wy1

    # Each pixel deposits a unit vote split bilinearly to 4 neighbors of proj point
    for b in range(B):
        idx00 = (y0[b, 0] * W + x0[b, 0]).flatten()
        idx01 = (y0[b, 0] * W + x1[b, 0]).flatten()
        idx10 = (y1[b, 0] * W + x0[b, 0]).flatten()
        idx11 = (y1[b, 0] * W + x1[b, 0]).flatten()
        w00 = (wx0[b, 0] * wy0[b, 0]).flatten()
        w01 = (wx1[b, 0] * wy0[b, 0]).flatten()
        w10 = (wx0[b, 0] * wy1[b, 0]).flatten()
        w11 = (wx1[b, 0] * wy1[b, 0]).flatten()
        flat = heat[b, 0].flatten()
        flat = flat.scatter_add(0, idx00, w00)
        flat = flat.scatter_add(0, idx01, w01)
        flat = flat.scatter_add(0, idx10, w10)
        flat = flat.scatter_add(0, idx11, w11)
        heat[b, 0] = flat.reshape(H, W)

    if sigma > 0:
        # Gaussian smoothing via separable 1D convolution
        K = int(4 * sigma) | 1
        coords = torch.arange(K, device=device, dtype=torch.float32) - K // 2
        g = torch.exp(- coords ** 2 / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, K)
        heat = F.conv2d(heat, g.unsqueeze(2), padding=(0, K // 2))
        heat = F.conv2d(heat, g.unsqueeze(3), padding=(K // 2, 0))

    if normalize:
        heat = heat / (H * W)
    return heat


def heatmap_loss(pred_heat, gt_endpoints, H, W, sigma_gt=2.0):
    """L2 loss between predicted heatmap and a GT heatmap made from endpoint pixels.

    Args:
        pred_heat   : [B, 1, H, W]
        gt_endpoints: list (len B) of [N_i, 2] tensors of (x, y) endpoint pixels
    """
    B = pred_heat.shape[0]
    gt_heat = torch.zeros_like(pred_heat)
    coords = torch.arange(int(4 * sigma_gt) | 1, device=pred_heat.device,
                          dtype=torch.float32) - (int(4 * sigma_gt) | 1) // 2
    g1 = torch.exp(- coords ** 2 / (2 * sigma_gt ** 2))
    g1 = g1 / g1.sum()
    K = g1.numel()

    for b in range(B):
        for x, y in gt_endpoints[b]:
            xi, yi = int(x.round()), int(y.round())
            if 0 <= xi < W and 0 <= yi < H:
                gt_heat[b, 0, max(0, yi - K // 2):yi + K // 2 + 1,
                              max(0, xi - K // 2):xi + K // 2 + 1] += 1.0
        # Optional: re-smooth GT for proper Gaussian peaks (skipped for brevity)

    return F.mse_loss(pred_heat, gt_heat)


if __name__ == '__main__':
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    afm = torch.zeros(B, 2, H, W, requires_grad=True)
    # Initialize with small random attraction so projection lands inside
    with torch.no_grad():
        afm[:, 0] = torch.randn(B, H, W) * 2
        afm[:, 1] = torch.randn(B, H, W) * 2
    afm = afm.detach().requires_grad_(True)
    heat = soft_voting_heatmap(afm, sigma=1.0)
    print(f"heatmap: {tuple(heat.shape)}, max={heat.max().item():.4f}, sum={heat.sum().item():.2f}")

    # Sanity: gradient flows through soft voting
    loss = heat.sum()
    loss.backward()
    print(f"grad norm: {afm.grad.norm().item():.4f}")
    print(f"all finite: {torch.isfinite(afm.grad).all().item()}")

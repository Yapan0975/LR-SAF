"""
LR-SAF: Soft Attraction Field target generation in pure PyTorch.

For an input set of line segments [N, 4] and image size (H, W), produce:
  - a_x, a_y: soft attraction vectors (after bounded encoding) [H, W]
  - t_star : projection parameter [0, 1] along each pixel's primary line  [H, W]
  - junc   : junction strength (entropy of top-K assignment weights)      [H, W]

Pure PyTorch, no CUDA op needed. Vectorized across all pixels and lines.
"""
import math
import torch


def line_segment_distance(p, x_s, x_e):
    """Distance from each pixel to each line segment (clipped projection).

    Args:
        p   : [HW, 2] pixel coordinates (x, y)
        x_s : [N, 2]  segment start points
        x_e : [N, 2]  segment end points

    Returns:
        dist  : [HW, N]   Euclidean distance
        t     : [HW, N]   clipped projection parameter in [0, 1]
        p_prj : [HW, N, 2] projection points
    """
    d = x_e - x_s                                # [N, 2]
    L2 = (d * d).sum(-1).clamp(min=1e-12)        # [N]
    # broadcast: [HW, 1, 2] - [1, N, 2] = [HW, N, 2]
    rel = p.unsqueeze(1) - x_s.unsqueeze(0)
    t = (rel * d.unsqueeze(0)).sum(-1) / L2.unsqueeze(0)   # [HW, N]
    t = t.clamp(0.0, 1.0)
    # projection points
    p_prj = x_s.unsqueeze(0) + t.unsqueeze(-1) * d.unsqueeze(0)   # [HW, N, 2]
    dist = (p.unsqueeze(1) - p_prj).norm(dim=-1)                  # [HW, N]
    return dist, t, p_prj


def compute_saf_target(lines, H, W, sigma=0.5, K=3, D_max=None,
                       bounded='afm', device='cpu'):
    """Compute Soft Attraction Field ground-truth for one image.

    Args:
        lines : tensor [N, 4]  (x_s, y_s, x_e, y_e)
        H, W  : image dimensions
        sigma : softmax temperature in distance space
        K     : top-K neighbors for soft assignment
        D_max : saturation distance for bounded encoding (default max(H,W)/4)
        bounded: 'afm'  -> AFM's log encoding: -sign(z) * log(|z|/scale + eps),
                          compatible with the original squeeze module.
                 'tanh' -> Our LR-SAF bounded encoding (unit ball, paper Eq 4.2).
                 None   -> raw pixel offsets.

    Returns:
        target: dict with keys
            'a_x', 'a_y'  : [H, W] attraction vector components (encoded as requested)
            't_star'       : [H, W] projection parameter to top-1 line
            'junc'         : [H, W] junction strength = 1 - max(w_k)
            'support'      : [H, W] indicator that any line is within D_max
    """
    if D_max is None:
        D_max = max(H, W) / 4.0
    N = lines.shape[0]
    K = min(K, N)
    lines = lines.to(device).float()
    x_s = lines[:, :2]
    x_e = lines[:, 2:]

    # Build pixel grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    p = torch.stack([xx, yy], dim=-1).reshape(-1, 2)   # [HW, 2]

    dist, t, p_prj = line_segment_distance(p, x_s, x_e)   # [HW, N], [HW, N], [HW, N, 2]

    # Top-K nearest lines per pixel
    top_d, top_idx = torch.topk(dist, K, dim=1, largest=False)        # [HW, K]
    top_t = torch.gather(t, 1, top_idx)                               # [HW, K]
    top_p = torch.gather(p_prj, 1, top_idx.unsqueeze(-1).expand(-1, -1, 2))   # [HW, K, 2]

    # Softmax weights over top-K (Eq. 4.1)
    logits = -top_d / sigma                                            # [HW, K]
    w = torch.softmax(logits, dim=1)                                   # [HW, K]

    # Soft attraction vector = sum_k w_k * (p'_k - p)
    delta = top_p - p.unsqueeze(1)                                     # [HW, K, 2]
    a = (w.unsqueeze(-1) * delta).sum(dim=1)                           # [HW, 2]

    # Encoding (compatible with AFM's squeeze module by default)
    if bounded == 'afm':
        # AFM training encoding (Eq. 8-9): size-normalize then log-stretch.
        # The squeeze module inverts this: z = sign(z') * exp(-|z'|) then mul by W/H.
        eps = 1e-6
        ax = a[:, 0] / W
        ay = a[:, 1] / H
        ax_enc = -torch.sign(ax) * torch.log(ax.abs() + eps)
        ay_enc = -torch.sign(ay) * torch.log(ay.abs() + eps)
        a = torch.stack([ax_enc, ay_enc], dim=-1)
    elif bounded == 'tanh':
        # LR-SAF custom bounded encoding (paper Eq. 4.2). Not squeeze-compatible.
        mag = a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        a_tilde = (a / mag) * torch.tanh(mag / D_max)
        a = a_tilde
    # else: leave as raw pixel offsets

    # t* (projection along top-1 line)
    t_star = top_t[:, 0]                                               # [HW]

    # Junction strength: 1 - max top-K weight  (high when ambiguous)
    junc = 1.0 - w.max(dim=1).values                                   # [HW]

    # Support mask: at least one line within D_max
    support = (top_d[:, 0] < D_max).float()                            # [HW]

    return {
        'a_x':     a[:, 0].reshape(H, W),
        'a_y':     a[:, 1].reshape(H, W),
        't_star':  t_star.reshape(H, W),
        'junc':    junc.reshape(H, W),
        'support': support.reshape(H, W),
        'D_max':   D_max,
        'sigma':   sigma,
        'K':       K,
    }


if __name__ == '__main__':
    # Smoke test
    lines = torch.tensor([
        [10.0, 20.0, 80.0, 90.0],
        [50.0, 10.0, 50.0, 90.0],
        [10.0, 50.0, 90.0, 50.0],
    ])
    out = compute_saf_target(lines, 100, 100, sigma=0.5, K=3)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={tuple(v.shape)}, range=[{v.min().item():.3f}, {v.max().item():.3f}]")
        else:
            print(f"  {k}: {v}")

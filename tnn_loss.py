"""
LR-SAF: Truncated Nuclear Norm loss with adaptive rank.

For predicted AFM (a_x, a_y) and optional junction strength s, compute the
window-wise TNNR penalty (Eq. 5.2 and 5.3 of the LR-SAF paper).

  L_tnnr = sum_W [ ||A_x(W)||_{r(W),*} + ||A_y(W)||_{r(W),*} ]
  r(W)  = 1 + floor((K_max - 1) * s(W))

All windows processed in a single batched SVD via torch.linalg.svdvals.
"""
import torch
import torch.nn.functional as F


def _unfold_windows(afm, window, stride):
    """afm: [B, 2, H, W] -> windows: [B, N, 2, window, window]."""
    B, C, H, Wd = afm.shape
    # F.unfold returns [B, C * window * window, num_windows]
    u = F.unfold(afm, kernel_size=window, stride=stride)              # [B, C*ws*ws, N]
    N = u.shape[-1]
    u = u.reshape(B, C, window, window, N).permute(0, 4, 1, 2, 3)     # [B, N, C, ws, ws]
    return u


def _truncated_singular_sum(M, r):
    """Sum of singular values beyond index r.
       M: [..., m, n]; r: int or Long[...] tensor matching prefix dims.
       Returns scalar sum of ||M[i,:,:]||_{r_i, *} over batch.
    """
    s = torch.linalg.svdvals(M)              # [..., min(m,n)]
    if isinstance(r, int):
        return s[..., r:].sum()
    # Adaptive r per item: build a mask [..., S]
    S = s.shape[-1]
    idx = torch.arange(S, device=s.device).expand(*s.shape)            # [..., S]
    r_exp = r.unsqueeze(-1)                                            # [..., 1]
    mask = (idx >= r_exp).float()
    return (s * mask).sum()


def tnn_loss(afm_pred, junc_pred=None,
             window=9, stride=4, K_max=4, fixed_r=None,
             normalize=True):
    """Compute the TNNR loss summed over windows.

    Args:
        afm_pred : [B, 2, H, W]  predicted attraction field (channels: a_x, a_y)
        junc_pred: [B, 1, H, W] or [B, H, W] junction-strength logits in [0, 1].
                   If None, fixed_r must be given.
        window   : window size (default 9)
        stride   : tiling stride (default 4)
        K_max    : maximum K for K-junction; sets r in [1, 2*K_max]
        fixed_r  : if int, use this rank uniformly and ignore junc_pred
        normalize: divide by total #windows for scale stability

    Returns:
        scalar tensor
    """
    B = afm_pred.shape[0]
    W = _unfold_windows(afm_pred, window, stride)                      # [B, N, 2, ws, ws]
    N = W.shape[1]

    # Per-window adaptive rank
    if fixed_r is not None:
        r_per_win = int(fixed_r)
    else:
        # Average junction strength inside each window
        if junc_pred.dim() == 4:
            junc_pred = junc_pred[:, 0]                                # [B, H, W]
        # Use the same unfold to average over each window
        Hs = junc_pred.unsqueeze(1)                                    # [B, 1, H, W]
        win_junc = _unfold_windows(Hs, window, stride)[:, :, 0]        # [B, N, ws, ws]
        s_win = win_junc.mean(dim=(-1, -2))                            # [B, N]
        s_win = s_win.clamp(0.0, 1.0)
        # r = 1 + floor((K_max - 1) * s)
        r_per_win = 1 + ((K_max - 1) * s_win).floor().long()           # [B, N]
        # Cap at window size
        r_per_win = r_per_win.clamp(max=window - 1)

    A_x = W[:, :, 0]                                                   # [B, N, ws, ws]
    A_y = W[:, :, 1]
    if isinstance(r_per_win, int):
        loss_x = _truncated_singular_sum(A_x, r_per_win)
        loss_y = _truncated_singular_sum(A_y, r_per_win)
    else:
        loss_x = _truncated_singular_sum(A_x, r_per_win)
        loss_y = _truncated_singular_sum(A_y, r_per_win)

    loss = loss_x + loss_y
    if normalize:
        loss = loss / max(B * N, 1)
    return loss


if __name__ == '__main__':
    # Smoke test: gradient flows, loss is finite
    torch.manual_seed(0)
    afm = torch.randn(2, 2, 64, 64, requires_grad=True)
    junc = torch.sigmoid(torch.randn(2, 1, 64, 64))
    loss = tnn_loss(afm, junc, window=9, stride=4, K_max=3)
    loss.backward()
    print(f"TNNR loss: {loss.item():.4f}")
    print(f"grad norm: {afm.grad.norm().item():.4f}")
    print(f"all finite: {torch.isfinite(afm.grad).all().item()}")

    # Fixed rank version
    loss_fixed = tnn_loss(afm.detach().clone().requires_grad_(),
                          fixed_r=2, window=9, stride=4)
    print(f"fixed-r=2 TNNR loss: {loss_fixed.item():.4f}")

"""LR-SAF: Bounded encoding and its inverse (Eq. 4.2 of the paper)."""
import torch


def encode_bounded(a, D_max, eps=1e-6):
    """tanh-saturate the attraction vector. a: [..., 2] or [..., 2, H, W]."""
    if a.shape[-1] == 2 and a.dim() <= 3:
        mag = a.norm(dim=-1, keepdim=True).clamp(min=eps)
        return (a / mag) * torch.tanh(mag / D_max)
    # Channel-first form [B, 2, H, W]
    mag = a.norm(dim=1, keepdim=True).clamp(min=eps)
    return (a / mag) * torch.tanh(mag / D_max)


def decode_bounded(a_tilde, D_max, eps=1e-6):
    """Inverse of tanh saturation. atanh is the inverse of tanh."""
    if a_tilde.shape[-1] == 2 and a_tilde.dim() <= 3:
        mag_t = a_tilde.norm(dim=-1, keepdim=True).clamp(min=eps, max=0.9999)
        mag = D_max * torch.atanh(mag_t)
        return (a_tilde / mag_t) * mag
    mag_t = a_tilde.norm(dim=1, keepdim=True).clamp(min=eps, max=0.9999)
    mag = D_max * torch.atanh(mag_t)
    return (a_tilde / mag_t) * mag


if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn(4, 2, 32, 32) * 10
    D_max = 80.0
    a_enc = encode_bounded(a, D_max)
    a_dec = decode_bounded(a_enc, D_max)
    err = (a - a_dec).abs().max().item()
    print(f"encode-decode round-trip max abs error: {err:.6f}")
    print(f"encoded magnitude range: [{a_enc.norm(dim=1).min().item():.4f}, "
          f"{a_enc.norm(dim=1).max().item():.4f}]")
    assert a_enc.norm(dim=1).max() <= 1.0
    print("bounded encoding stays in unit ball: OK")

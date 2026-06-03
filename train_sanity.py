"""
LR-SAF: Training sanity check.

Goal: verify the pipeline (data -> model -> losses -> backward -> step) runs
correctly with no NaNs and the loss decreases. Uses a tiny YorkUrban subset
since we don't yet have Wireframe pointlines locally.

Run:  python3 train_sanity.py
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

import torch
import torch.nn.functional as F

from saf_target import compute_saf_target
from tnn_loss import tnn_loss
from bounded import encode_bounded
from data import YorkUrbanSubset, collate_variable_lines
from model import build_lr_saf


def distance_weighted_l1(pred_a, gt_a, support, weight=10.0):
    """Eq. 5: distance-weighted L1 on attraction field.
       pred_a, gt_a: [B, 2, H, W]; support: [B, H, W] in {0,1}.
       Pixels inside support get weight w, others get 1.
    """
    diff = (pred_a - gt_a).abs().sum(dim=1)            # [B, H, W]
    w = support * weight + (1 - support) * 1.0
    return (w * diff).mean()


def make_saf_targets(lines_batch, mask_batch, H, W, device, K=3, sigma=2.0, D_max=80.0):
    """Compute SAF targets for a batch of variable-length line sets."""
    B = lines_batch.shape[0]
    targets = {'a': torch.zeros(B, 2, H, W, device=device),
               't_star': torch.zeros(B, 1, H, W, device=device),
               'junc': torch.zeros(B, 1, H, W, device=device),
               'support': torch.zeros(B, H, W, device=device)}
    for b in range(B):
        n = int(mask_batch[b].sum().item())
        if n == 0:
            continue
        lns = lines_batch[b, :n].to(device)
        out = compute_saf_target(lns, H, W, sigma=sigma, K=K,
                                 D_max=D_max, bounded=True, device=device)
        targets['a'][b, 0] = out['a_x']
        targets['a'][b, 1] = out['a_y']
        targets['t_star'][b, 0] = out['t_star']
        targets['junc'][b, 0] = out['junc']
        targets['support'][b] = out['support']
    return targets


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    print(f"torch:  {torch.__version__}")

    # Tiny dataset
    ds = YorkUrbanSubset(in_res=320, limit=8)
    print(f"dataset items: {len(ds)}")
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True,
                                     collate_fn=collate_variable_lines,
                                     num_workers=0)

    # Model
    model = build_lr_saf(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {n_params/1e6:.2f}M params, {n_trainable/1e6:.2f}M trainable")

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Loss weights (Eq. 5.1)
    LAMBDA_REC, LAMBDA_TNN, LAMBDA_JUNC, LAMBDA_T = 1.0, 0.05, 1.0, 0.5
    D_MAX = 80.0

    print("\n--- starting sanity training (3 epochs, ~12 steps) ---\n")
    history = []
    for epoch in range(3):
        epoch_t0 = time.time()
        for it, batch in enumerate(dl):
            optim.zero_grad()
            imgs = batch['image'].to(device)                              # [B, 3, 320, 320]
            B, _, H, W = imgs.shape

            # Targets
            with torch.no_grad():
                targets = make_saf_targets(batch['lines'], batch['lines_mask'],
                                           H, W, device, K=3, sigma=2.0, D_max=D_MAX)

            # Forward (predicted a is raw; need to bounded-encode for comparison)
            out = model(imgs)
            pred_a_raw = out['a']                                         # [B, 2, H, W]
            pred_a_enc = encode_bounded(pred_a_raw, D_max=D_MAX)
            pred_t = out['t_star']
            pred_junc = out['junc']

            # Losses
            L_rec = distance_weighted_l1(pred_a_enc, targets['a'], targets['support'])
            L_junc = F.binary_cross_entropy(pred_junc.clamp(1e-6, 1-1e-6),
                                            targets['junc'])
            # t* only where support
            L_t = (((pred_t.squeeze(1) - targets['t_star'].squeeze(1)).abs()
                    * targets['support']).sum() / targets['support'].sum().clamp(min=1))
            # TNNR on the bounded prediction
            L_tnnr = tnn_loss(pred_a_enc, junc_pred=pred_junc,
                             window=9, stride=8, K_max=3, normalize=True)

            L = (LAMBDA_REC * L_rec + LAMBDA_TNN * L_tnnr
                 + LAMBDA_JUNC * L_junc + LAMBDA_T * L_t)

            L.backward()
            # Sanity: grad finite
            gnorm = sum((p.grad.norm() ** 2
                         for p in model.parameters() if p.grad is not None)) ** 0.5
            assert torch.isfinite(L), f"loss NaN at epoch {epoch} iter {it}"
            assert torch.isfinite(torch.tensor(float(gnorm))), \
                f"grad NaN at epoch {epoch} iter {it}"

            optim.step()

            history.append((epoch, it, L.item(), L_rec.item(), L_tnnr.item(),
                            L_junc.item(), L_t.item(), float(gnorm)))
            print(f"ep{epoch} it{it}: L={L.item():.4f} | rec={L_rec.item():.4f} "
                  f"TNNR={L_tnnr.item():.4f} junc={L_junc.item():.4f} t={L_t.item():.4f} "
                  f"|grad|={gnorm:.3f}")
        print(f"epoch {epoch} done in {time.time()-epoch_t0:.1f}s")

    # Save checkpoint
    ckpt_path = ROOT + '/checkpoints/lr_saf_sanity.pth'
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({'model': model.state_dict(),
                'history': history,
                'config': {'D_max': D_MAX,
                           'lambda': (LAMBDA_REC, LAMBDA_TNN, LAMBDA_JUNC, LAMBDA_T)}},
               ckpt_path)
    print(f"\nsaved {ckpt_path}")

    # Summary
    print("\n=== SANITY SUMMARY ===")
    L_first = sum(h[2] for h in history[:2]) / 2
    L_last = sum(h[2] for h in history[-2:]) / 2
    print(f"avg total loss first 2 steps: {L_first:.4f}")
    print(f"avg total loss last  2 steps: {L_last:.4f}")
    print(f"reduction: {(1 - L_last/L_first)*100:.1f}%")
    if L_last < L_first:
        print("LOSS DECREASES — pipeline works")
    else:
        print("WARNING: loss did not decrease (may need more steps or LR tune)")


if __name__ == '__main__':
    main()

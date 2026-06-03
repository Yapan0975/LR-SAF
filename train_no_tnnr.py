"""Train a LR-SAF variant with TNNR DISABLED (lam_tnnr=0).
Same setup as train_full.py for direct A/B comparison.
"""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

import torch, torch.nn.functional as F, numpy as np, cv2
from saf_target import compute_saf_target
from data import YorkUrbanSubset, collate_variable_lines
from model import build_lr_saf

def main(epochs=20, batch=2, lr=5e-5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda'
    ds = YorkUrbanSubset(in_res=320)
    rng = np.random.RandomState(seed); perm = rng.permutation(len(ds))
    train_idx = perm[:80].tolist()
    train_set = torch.utils.data.Subset(ds, train_idx)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True,
                                          collate_fn=collate_variable_lines, num_workers=2)
    model = build_lr_saf(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    print(f"training LR-SAF NO-TNNR (lam_tnnr=0), {epochs} epochs")
    for ep in range(epochs):
        model.train(); t0 = time.time(); ep_L = []
        for batch_d in loader:
            optim.zero_grad()
            imgs = batch_d['image'].to(device); B, _, H, W = imgs.shape
            with torch.no_grad():
                tgt_list = []
                for b in range(B):
                    n = int(batch_d['lines_mask'][b].sum().item())
                    if n == 0:
                        tgt_list.append(None); continue
                    o = compute_saf_target(batch_d['lines'][b, :n].to(device), H, W,
                                            sigma=2.0, K=3, bounded='afm', device=device)
                    tgt_list.append(o)
            a_tgt = torch.zeros(B, 2, H, W, device=device)
            t_tgt = torch.zeros(B, 1, H, W, device=device)
            j_tgt = torch.zeros(B, 1, H, W, device=device)
            s_tgt = torch.zeros(B, H, W, device=device)
            for b, o in enumerate(tgt_list):
                if o is None: continue
                a_tgt[b, 0] = o['a_x']; a_tgt[b, 1] = o['a_y']
                t_tgt[b, 0] = o['t_star']; j_tgt[b, 0] = o['junc']; s_tgt[b] = o['support']
            out = model(imgs)
            a_enc = out['a']; t_pred = out['t_star']; junc_pred = out['junc']
            diff = (a_enc - a_tgt).abs().sum(dim=1)
            w = s_tgt * 10.0 + (1 - s_tgt) * 1.0
            L_rec = (w * diff).mean()
            L_junc = F.binary_cross_entropy(junc_pred.clamp(1e-6, 1-1e-6), j_tgt)
            L_t = (((t_pred.squeeze(1) - t_tgt.squeeze(1)).abs() * s_tgt).sum() / s_tgt.sum().clamp(min=1))
            # NO TNNR loss here
            L = 1.0*L_rec + 1.0*L_junc + 0.5*L_t
            if not torch.isfinite(L): continue
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step(); ep_L.append(L.item())
        sched.step()
        print(f"  ep{ep:02d}: L={np.mean(ep_L):.4f} ({time.time()-t0:.1f}s)")
    save = ROOT + '/checkpoints/lr_saf_no_tnnr.pth'
    torch.save({'model': model.state_dict()}, save)
    print(f"saved {save}")

if __name__ == '__main__':
    main()

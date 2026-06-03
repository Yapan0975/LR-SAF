"""
Train confidence head WITH semantic features.

Pipeline:
  1. Frozen LR-SAF main net -> AFM + junc (9 geometric features per segment)
  2. Frozen DeepLabV3 -> 21-channel softmax (semantic feature per pixel)
  3. Sample 16 points along each candidate segment, mean-pool semantic -> 21-d
  4. Concat 9 geom + 21 semantic = 30-d feature
  5. MLP -> 1-d confidence -> BCE against GT match label

Compare against geometry-only baseline (train_confidence.py).
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from data import YorkUrbanSubset
from model import build_lr_saf
from confidence_head import compute_segment_features, match_to_gt
from semantic_features import SemanticExtractor, sample_along_line
from metrics import s_ap
from lib.squeeze_to_lsg import lsgenerator


class SemanticConfMLP(nn.Module):
    def __init__(self, geom_dim=9, sem_dim=21, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geom_dim + sem_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, geom_feats, sem_feats):
        x = torch.cat([geom_feats, sem_feats], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


def run_squeeze(model, img, in_res=320):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (in_res, in_res)).astype(np.float32) / 255.0
    x_norm = (x - MEAN) / STD
    x_t = torch.from_numpy(x_norm).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x_t)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), offset, junc, x_t, (H_o, W_o)


def precompute(model, sem_ext, items, ds, label='train'):
    """Run frozen pipeline + semantic on each image; return per-image dict
    with geom feats, sem feats, GT match labels, raw kept segments."""
    out_list = []
    n_pos_total, n_total = 0, 0
    for item in items:
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        lines, offset, junc, x_norm_t, (H_o, W_o) = run_squeeze(model, img)
        if len(lines) == 0:
            continue
        # Geom features (computed in 320 coords)
        geom = compute_segment_features(lines, offset, junc, H=320, W=320)

        # Semantic feats in 320 coords (sample on per-pixel softmax)
        with torch.no_grad():
            sem_map = sem_ext(x_norm_t)[0]                  # [21, 320, 320]
        sem = sample_along_line(sem_map, lines[:, :4], n_samples=16)

        # Scale segments back to original image for GT matching
        sx, sy = W_o / 320.0, H_o / 320.0
        kept = lines[:, :5].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0
        tgt = match_to_gt(torch.from_numpy(kept[:, :4]).float(),
                          torch.from_numpy(gt).float(), H=H_o, W=W_o)

        out_list.append({
            'name': item['name'],
            'geom': geom.cuda(),                            # [N, 9]
            'sem':  sem.cuda(),                             # [N, 21]
            'tgt':  tgt.cuda(),                             # [N]
            'kept': kept,                                   # [N, 5+] in orig coords
            'gt':   gt, 'H_o': H_o, 'W_o': W_o,
        })
        n_pos_total += int(tgt.sum().item())
        n_total += len(tgt)
    print(f"  {label}: {len(out_list)} images, {n_total} candidates, "
          f"{n_pos_total} positives ({100*n_pos_total/max(n_total,1):.1f}%)")
    return out_list


def main(epochs=20, lr=1e-3, hidden=64, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    # Train/val split (same as elsewhere)
    ds = YorkUrbanSubset(in_res=320)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds))
    train_idx, val_idx = perm[:80].tolist(), perm[80:].tolist()
    items_train = [ds[i] for i in train_idx]
    items_val = [ds[i] for i in val_idx]

    # Frozen LR-SAF main
    main_model = build_lr_saf(device='cuda').eval()
    ckpt = torch.load(ROOT + '/checkpoints/lr_saf_best.pth',
                       map_location='cuda', weights_only=False)
    main_model.load_state_dict(ckpt['model'], strict=True)
    for p in main_model.parameters(): p.requires_grad_(False)
    print(f"loaded LR-SAF (epoch {ckpt['epoch']}, val_F={ckpt['val_F']:.4f})")

    # Frozen semantic extractor
    print("loading semantic extractor...")
    sem_ext = SemanticExtractor().cuda().eval()
    print("  done")

    print("\n=== precomputing ===")
    train_data = precompute(main_model, sem_ext, items_train, ds, label='train')
    val_data = precompute(main_model, sem_ext, items_val, ds, label='val')

    # Pos-weight
    pos = sum(d['tgt'].sum().item() for d in train_data)
    neg = sum((d['tgt'] == 0).sum().item() for d in train_data)
    pos_w = max(1.0, neg / max(pos, 1))
    print(f"\npos_weight: {pos_w:.2f}")

    # Head
    head = SemanticConfMLP(geom_dim=9, sem_dim=21, hidden=hidden).cuda()
    optim = torch.optim.Adam(head.parameters(), lr=lr)

    print(f"\n=== training (geom + semantic, {epochs} epochs) ===")
    best_ap10 = -1.0
    best_state = None
    for ep in range(epochs):
        head.train()
        order = np.random.permutation(len(train_data))
        ep_L = []
        for idx in order:
            d = train_data[idx]
            scores = head(d['geom'], d['sem']).clamp(1e-6, 1 - 1e-6)
            tgt = d['tgt']
            w = tgt * pos_w + (1 - tgt)
            loss = (- w * (tgt * torch.log(scores) + (1 - tgt) * torch.log(1 - scores))).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            ep_L.append(loss.item())

        head.eval()
        with torch.no_grad():
            ap5, ap10, ap15 = [], [], []
            for d in val_data:
                scores = head(d['geom'], d['sem']).cpu().numpy()
                aps = s_ap(d['kept'][:, :4], scores, d['gt'],
                           d['H_o'], d['W_o'], thresholds=(5, 10, 15))
                ap5.append(aps[5]); ap10.append(aps[10]); ap15.append(aps[15])
            ap5_m = float(np.mean(ap5))
            ap10_m = float(np.mean(ap10))
            ap15_m = float(np.mean(ap15))
        print(f"  ep{ep:02d}: L={np.mean(ep_L):.4f} | val sAP5={ap5_m:.4f} sAP10={ap10_m:.4f} sAP15={ap15_m:.4f}")
        if ap10_m > best_ap10:
            best_ap10 = ap10_m
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_eps = (ap5_m, ap10_m, ap15_m, ep)

    # Final evaluation with best checkpoint
    head.load_state_dict(best_state)
    head.eval()

    # Compare to geometry-only baseline (1/aspect score)
    print("\n=== COMPARISON (val 22) ===")
    ap_g_only = []
    for d in val_data:
        scores = 1.0 / (d['kept'][:, 4] + 1e-3)
        a = s_ap(d['kept'][:, :4], scores, d['gt'], d['H_o'], d['W_o'], (10,))
        ap_g_only.append(a[10])
    print(f"  1/aspect heuristic                           sAP10 = {np.mean(ap_g_only):.4f}")

    # Geometry-only confidence head (load if exists)
    g_ckpt = ROOT + '/checkpoints/lr_saf_conf_head.pth'
    if os.path.exists(g_ckpt):
        from confidence_head import ConfidenceMLP
        g_head = ConfidenceMLP(in_dim=9, hidden=64).cuda()
        g_head.load_state_dict(torch.load(g_ckpt, weights_only=False)['head'])
        g_head.eval()
        ap_g = []
        with torch.no_grad():
            for d in val_data:
                s = g_head(d['geom']).cpu().numpy()
                a = s_ap(d['kept'][:, :4], s, d['gt'], d['H_o'], d['W_o'], (10,))
                ap_g.append(a[10])
        print(f"  geometry-only learned MLP (9-d)             sAP10 = {np.mean(ap_g):.4f}")

    print(f"  geometry + semantic learned MLP (9+21=30-d) sAP5={best_eps[0]:.4f} "
          f"sAP10={best_eps[1]:.4f} sAP15={best_eps[2]:.4f} (best epoch {best_eps[3]})")

    # Save
    out_ckpt = ROOT + '/checkpoints/lr_saf_conf_head_sem.pth'
    torch.save({'head': best_state, 'best_metrics': best_eps,
                 'config': {'geom_dim': 9, 'sem_dim': 21, 'hidden': hidden}},
                out_ckpt)
    print(f"saved {out_ckpt}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    main(epochs=args.epochs, lr=args.lr)

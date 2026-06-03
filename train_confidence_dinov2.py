"""
Train confidence head with DINOv2 self-supervised features.

Comparison harness:
  Head A: 1/aspect (no learned)
  Head B: geom-only MLP (9-d)              ← saved as lr_saf_conf_head.pth
  Head C: geom + VOC-21 MLP (30-d)         ← saved as lr_saf_conf_head_sem.pth
  Head D: geom + DINOv2-384 MLP (393-d)    ← NEW, saved as lr_saf_conf_head_dinov2.pth

Runs Head D and prints A/B/C/D side-by-side.
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

import torch
import torch.nn as nn
import numpy as np
import cv2

from data import YorkUrbanSubset
from model import build_lr_saf
from confidence_head import ConfidenceMLP, compute_segment_features, match_to_gt
from train_confidence_semantic import SemanticConfMLP
from semantic_features import SemanticExtractor, sample_along_line
from dinov2_features import DINOv2Extractor
from metrics import s_ap
from lib.squeeze_to_lsg import lsgenerator


class DINOConfMLP(nn.Module):
    def __init__(self, geom_dim=9, dino_dim=384, proj=16, hidden=32, dropout=0.5):
        super().__init__()
        # Heavy compression to prevent overfit; DINOv2 features are very high-dim
        self.proj = nn.Sequential(
            nn.Linear(dino_dim, proj), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.net = nn.Sequential(
            nn.Linear(geom_dim + proj, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, geom, dino):
        d = self.proj(dino)
        x = torch.cat([geom, d], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


def run_squeeze(model, img, in_res=320):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (in_res, in_res)).astype(np.float32) / 255.0
    x_n = (x - MEAN) / STD
    x_t = torch.from_numpy(x_n).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x_t)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), offset, junc, x_t, (H_o, W_o)


def precompute(main_model, dino_ext, items, ds, label='train'):
    out_list = []
    n_pos_total, n_total = 0, 0
    for item in items:
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        lines, offset, junc, x_norm_t, (H_o, W_o) = run_squeeze(main_model, img)
        if len(lines) == 0:
            continue
        geom = compute_segment_features(lines, offset, junc, H=320, W=320)
        with torch.no_grad():
            dino_map = dino_ext(x_norm_t)[0]
        dino = sample_along_line(dino_map, lines[:, :4], n_samples=16)
        sx, sy = W_o / 320.0, H_o / 320.0
        kept = lines[:, :5].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0
        tgt = match_to_gt(torch.from_numpy(kept[:, :4]).float(),
                          torch.from_numpy(gt).float(), H=H_o, W=W_o)
        out_list.append({
            'name': item['name'], 'geom': geom.cuda(), 'dino': dino.cuda(),
            'tgt': tgt.cuda(), 'kept': kept, 'gt': gt,
            'H_o': H_o, 'W_o': W_o,
        })
        n_pos_total += int(tgt.sum().item())
        n_total += len(tgt)
    print(f"  {label}: {len(out_list)} images, {n_total} candidates, "
          f"{n_pos_total} positives ({100*n_pos_total/max(n_total,1):.1f}%)")
    return out_list


def main(epochs=30, lr=1e-3, hidden=128, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda'

    ds = YorkUrbanSubset(in_res=320)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds))
    train_idx, val_idx = perm[:80].tolist(), perm[80:].tolist()
    items_train = [ds[i] for i in train_idx]
    items_val = [ds[i] for i in val_idx]

    main_model = build_lr_saf(device=device).eval()
    ckpt = torch.load(ROOT + '/checkpoints/lr_saf_best.pth',
                       map_location=device, weights_only=False)
    main_model.load_state_dict(ckpt['model'], strict=True)
    for p in main_model.parameters(): p.requires_grad_(False)

    dino_ext = DINOv2Extractor().cuda().eval()
    print("DINOv2 extractor ready")

    print("\n=== precompute features ===")
    train_data = precompute(main_model, dino_ext, items_train, ds, 'train')
    val_data = precompute(main_model, dino_ext, items_val, ds, 'val')

    pos = sum(d['tgt'].sum().item() for d in train_data)
    neg = sum((d['tgt'] == 0).sum().item() for d in train_data)
    pos_w = max(1.0, neg / max(pos, 1))
    print(f"pos_weight: {pos_w:.2f}")

    head = DINOConfMLP(geom_dim=9, dino_dim=384, proj=16, hidden=32, dropout=0.5).cuda()
    optim = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-3)

    print(f"\n=== training (geom + DINOv2, {epochs} epochs) ===")
    best = {'sAP10': -1.0}
    best_state = None
    for ep in range(epochs):
        head.train()
        order = np.random.permutation(len(train_data))
        ep_L = []
        for idx in order:
            d = train_data[idx]
            scores = head(d['geom'], d['dino']).clamp(1e-6, 1 - 1e-6)
            tgt = d['tgt']
            w = tgt * pos_w + (1 - tgt)
            loss = (- w * (tgt * torch.log(scores)
                           + (1 - tgt) * torch.log(1 - scores))).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            ep_L.append(loss.item())

        head.eval()
        with torch.no_grad():
            ap5, ap10, ap15 = [], [], []
            for d in val_data:
                scores = head(d['geom'], d['dino']).cpu().numpy()
                aps = s_ap(d['kept'][:, :4], scores, d['gt'],
                           d['H_o'], d['W_o'], thresholds=(5, 10, 15))
                ap5.append(aps[5]); ap10.append(aps[10]); ap15.append(aps[15])
            ap5_m, ap10_m, ap15_m = (float(np.mean(ap5)), float(np.mean(ap10)),
                                     float(np.mean(ap15)))
        print(f"  ep{ep:02d}: L={np.mean(ep_L):.4f} | "
              f"sAP5={ap5_m:.4f} sAP10={ap10_m:.4f} sAP15={ap15_m:.4f}")
        if ap10_m > best['sAP10']:
            best = {'sAP5': ap5_m, 'sAP10': ap10_m, 'sAP15': ap15_m, 'ep': ep}
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Compare to all baselines at val
    print("\n=== ALL HEADS COMPARISON (val 22, sAP-10) ===")
    head.load_state_dict(best_state); head.eval()

    # A: 1/aspect heuristic
    ap10_aspect = []
    for d in val_data:
        scores = 1.0 / (d['kept'][:, 4] + 1e-3)
        a = s_ap(d['kept'][:, :4], scores, d['gt'], d['H_o'], d['W_o'], (10,))
        ap10_aspect.append(a[10])

    # B: geom-only learned
    from confidence_head import ConfidenceMLP
    geom_head = ConfidenceMLP(in_dim=9, hidden=64).cuda().eval()
    g_ckpt = torch.load(ROOT + '/checkpoints/lr_saf_conf_head.pth',
                         weights_only=False)
    geom_head.load_state_dict(g_ckpt['head'])
    ap10_geom = []
    with torch.no_grad():
        for d in val_data:
            s = geom_head(d['geom']).cpu().numpy()
            a = s_ap(d['kept'][:, :4], s, d['gt'], d['H_o'], d['W_o'], (10,))
            ap10_geom.append(a[10])

    # C: geom + VOC-21 semantic
    sem_ext = SemanticExtractor().cuda().eval()
    sem_head = SemanticConfMLP(geom_dim=9, sem_dim=21, hidden=64).cuda().eval()
    sem_head.load_state_dict(torch.load(
        ROOT + '/checkpoints/lr_saf_conf_head_sem.pth',
        weights_only=False)['head'])
    ap10_voc = []
    with torch.no_grad():
        for d in val_data:
            img = cv2.imread(os.path.join(ds.root, d['name'], f"{d['name']}.jpg"))
            # Re-extract VOC features (we didn't cache them earlier in this script)
            MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
            x = (x - MEAN) / STD
            x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
            sem_map = sem_ext(x_t)[0]
            # We need lines back in 320 coords for sampling. Reverse-scale kept:
            kept_320 = d['kept'][:, :4].copy()
            kept_320[:, 0::2] *= 320.0 / d['W_o']
            kept_320[:, 1::2] *= 320.0 / d['H_o']
            sem = sample_along_line(sem_map, kept_320, n_samples=16).cuda()
            scr = sem_head(d['geom'], sem).cpu().numpy()
            a = s_ap(d['kept'][:, :4], scr, d['gt'], d['H_o'], d['W_o'], (10,))
            ap10_voc.append(a[10])

    print(f"  A. 1/aspect heuristic            : sAP10 = {np.mean(ap10_aspect):.4f}")
    print(f"  B. geom-only MLP (9-d)            : sAP10 = {np.mean(ap10_geom):.4f}")
    print(f"  C. geom + VOC-21 MLP (30-d)       : sAP10 = {np.mean(ap10_voc):.4f}")
    print(f"  D. geom + DINOv2-384 MLP (9+64-d) : sAP10 = {best['sAP10']:.4f}  "
          f"(sAP5={best['sAP5']:.4f}, sAP15={best['sAP15']:.4f}, ep {best['ep']})")
    print(f"  AFM baseline (paper ref): sAP10 = 0.1638")

    out = ROOT + '/checkpoints/lr_saf_conf_head_dinov2.pth'
    torch.save({'head': best_state, 'best': best,
                'config': {'geom_dim': 9, 'dino_dim': 384, 'hidden': hidden}}, out)
    print(f"saved {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    main(epochs=args.epochs, lr=args.lr)

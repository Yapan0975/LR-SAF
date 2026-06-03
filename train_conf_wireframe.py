"""
Train confidence head on Wireframe-trained LR-SAF.

Uses lr_saf_wireframe_best.pth as frozen backbone.
Trains both: (B) geom-only and (C) geom+VOC semantic heads.
Reports sAP-10 on Wireframe test (462) and YorkUrban test (102, cross-domain).
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

from data_hawp import wireframe_train, wireframe_test, york_test
from model import build_lr_saf
from confidence_head import ConfidenceMLP, compute_segment_features, match_to_gt
from train_confidence_semantic import SemanticConfMLP
from semantic_features import SemanticExtractor, sample_along_line
from metrics import s_ap, f_measure
from lib.squeeze_to_lsg import lsgenerator

DATA_ROOT = ROOT + '/code/hawp_baseline/data'
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_squeeze(model, img_norm_t):
    with torch.no_grad():
        out = model(img_norm_t)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), offset, junc


def precompute(main_model, sem_ext, ds, label='train', limit=None, with_sem=True):
    out_list = []
    n_pos_total, n_total = 0, 0
    indices = list(range(min(limit, len(ds)) if limit else len(ds)))
    t0 = time.time()
    for ii in indices:
        item = ds[ii]
        H_o, W_o = item['H_orig'], item['W_orig']
        x_t = item['image'].unsqueeze(0).cuda()
        lines, offset, junc = run_squeeze(main_model, x_t)
        if len(lines) == 0:
            continue
        geom = compute_segment_features(lines, offset, junc, H=320, W=320)
        if with_sem and sem_ext is not None:
            with torch.no_grad():
                sem_map = sem_ext(x_t)[0]
            sem = sample_along_line(sem_map, lines[:, :4], n_samples=16).cuda()
        else:
            sem = None
        sx, sy = W_o / 320.0, H_o / 320.0
        kept = lines[:, :5].copy() if lines.shape[1] >= 5 else lines[:, :4].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0
        tgt = match_to_gt(torch.from_numpy(kept[:, :4]).float(),
                          torch.from_numpy(gt).float(), H=H_o, W=W_o)
        out_list.append({
            'name': item['name'], 'geom': geom.cuda(), 'sem': sem,
            'tgt': tgt.cuda(), 'kept': kept, 'gt': gt,
            'H_o': H_o, 'W_o': W_o,
        })
        n_pos_total += int(tgt.sum().item())
        n_total += len(tgt)
        if (ii + 1) % 500 == 0:
            print(f"  [{label}] {ii+1}/{len(indices)} ({time.time()-t0:.1f}s)")
    print(f"  [{label}] {len(out_list)} valid images, {n_total} candidates, "
          f"{n_pos_total} pos ({100*n_pos_total/max(n_total,1):.1f}%)  "
          f"[{time.time()-t0:.1f}s]")
    return out_list


def eval_head(head, val_data, with_sem):
    head.eval()
    ap10s = []
    ap5s = []
    with torch.no_grad():
        for d in val_data:
            if with_sem:
                s = head(d['geom'], d['sem']).cpu().numpy()
            else:
                s = head(d['geom']).cpu().numpy()
            aps = s_ap(d['kept'][:, :4], s, d['gt'], d['H_o'], d['W_o'], (5, 10))
            ap5s.append(aps[5]); ap10s.append(aps[10])
    return float(np.mean(ap5s)), float(np.mean(ap10s))


def main(epochs=15, lr=1e-3, hidden=64, seed=42, train_limit=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda'

    # Datasets
    tr_ds = wireframe_train(DATA_ROOT, in_res=320, augment=False)   # eval mode for prediction
    wf_test = wireframe_test(DATA_ROOT, in_res=320)
    york_t = york_test(DATA_ROOT, in_res=320)
    print(f"Wireframe train: {len(tr_ds)}, test: {len(wf_test)}, York test: {len(york_t)}")

    # Frozen LR-SAF main
    main_model = build_lr_saf(device=device).eval()
    ckpt = torch.load(ROOT + '/checkpoints/lr_saf_wireframe_best.pth',
                       map_location=device, weights_only=False)
    main_model.load_state_dict(ckpt['model'], strict=True)
    for p in main_model.parameters(): p.requires_grad_(False)
    print(f"loaded LR-SAF wireframe best: ep={ckpt['epoch']}, WF F={ckpt['wf_F']:.4f}")

    # Frozen semantic extractor
    sem_ext = SemanticExtractor().cuda().eval()

    # Precompute features for train (using main model + sem ext)
    print("\n=== precompute (this is the slow step) ===")
    train_data = precompute(main_model, sem_ext, tr_ds, 'train', limit=train_limit)
    wf_val_data = precompute(main_model, sem_ext, wf_test, 'wf_test')
    yk_val_data = precompute(main_model, sem_ext, york_t, 'york_test')

    pos = sum(d['tgt'].sum().item() for d in train_data)
    neg = sum((d['tgt'] == 0).sum().item() for d in train_data)
    pos_w = max(1.0, neg / max(pos, 1))
    print(f"pos_weight: {pos_w:.2f}")

    results = {}

    # ============================================================
    # Head B: geom-only
    # ============================================================
    print("\n=== training (B) geom-only confidence head ===")
    head_b = ConfidenceMLP(in_dim=9, hidden=hidden).cuda()
    opt_b = torch.optim.Adam(head_b.parameters(), lr=lr)
    best_b = {'wf_sAP10': -1, 'epoch': -1}
    for ep in range(epochs):
        head_b.train()
        order = np.random.permutation(len(train_data))
        ep_L = []
        for idx in order:
            d = train_data[idx]
            s = head_b(d['geom']).clamp(1e-6, 1-1e-6)
            tgt = d['tgt']
            w = tgt * pos_w + (1 - tgt)
            loss = (-w * (tgt * torch.log(s) + (1 - tgt) * torch.log(1 - s))).mean()
            opt_b.zero_grad(); loss.backward(); opt_b.step()
            ep_L.append(loss.item())
        wf_ap5, wf_ap10 = eval_head(head_b, wf_val_data, with_sem=False)
        y_ap5, y_ap10 = eval_head(head_b, yk_val_data, with_sem=False)
        print(f"  B ep{ep:02d}: L={np.mean(ep_L):.4f} | "
              f"WF sAP5={wf_ap5:.4f} sAP10={wf_ap10:.4f} | "
              f"York sAP5={y_ap5:.4f} sAP10={y_ap10:.4f}")
        if wf_ap10 > best_b['wf_sAP10']:
            best_b = {'wf_sAP5': wf_ap5, 'wf_sAP10': wf_ap10,
                      'y_sAP5': y_ap5, 'y_sAP10': y_ap10, 'epoch': ep,
                      'state': {k: v.clone() for k, v in head_b.state_dict().items()}}
    results['geom_only'] = {k: v for k, v in best_b.items() if k != 'state'}
    torch.save({'head': best_b['state']}, ROOT + '/checkpoints/lr_saf_wf_conf_geom.pth')

    # ============================================================
    # Head C: geom + VOC semantic
    # ============================================================
    print("\n=== training (C) geom + VOC semantic confidence head ===")
    head_c = SemanticConfMLP(geom_dim=9, sem_dim=21, hidden=hidden).cuda()
    opt_c = torch.optim.Adam(head_c.parameters(), lr=lr)
    best_c = {'wf_sAP10': -1, 'epoch': -1}
    for ep in range(epochs):
        head_c.train()
        order = np.random.permutation(len(train_data))
        ep_L = []
        for idx in order:
            d = train_data[idx]
            s = head_c(d['geom'], d['sem']).clamp(1e-6, 1-1e-6)
            tgt = d['tgt']
            w = tgt * pos_w + (1 - tgt)
            loss = (-w * (tgt * torch.log(s) + (1 - tgt) * torch.log(1 - s))).mean()
            opt_c.zero_grad(); loss.backward(); opt_c.step()
            ep_L.append(loss.item())
        wf_ap5, wf_ap10 = eval_head(head_c, wf_val_data, with_sem=True)
        y_ap5, y_ap10 = eval_head(head_c, yk_val_data, with_sem=True)
        print(f"  C ep{ep:02d}: L={np.mean(ep_L):.4f} | "
              f"WF sAP5={wf_ap5:.4f} sAP10={wf_ap10:.4f} | "
              f"York sAP5={y_ap5:.4f} sAP10={y_ap10:.4f}")
        if wf_ap10 > best_c['wf_sAP10']:
            best_c = {'wf_sAP5': wf_ap5, 'wf_sAP10': wf_ap10,
                      'y_sAP5': y_ap5, 'y_sAP10': y_ap10, 'epoch': ep,
                      'state': {k: v.clone() for k, v in head_c.state_dict().items()}}
    results['geom_voc_sem'] = {k: v for k, v in best_c.items() if k != 'state'}
    torch.save({'head': best_c['state']}, ROOT + '/checkpoints/lr_saf_wf_conf_sem.pth')

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Head':<20} {'WF sAP5':>10} {'WF sAP10':>10} {'York sAP5':>10} {'York sAP10':>10} {'epoch':>6}")
    print("-" * 80)
    for k, r in results.items():
        print(f"{k:<20} {r['wf_sAP5']:>10.4f} {r['wf_sAP10']:>10.4f} "
              f"{r['y_sAP5']:>10.4f} {r['y_sAP10']:>10.4f} {r['epoch']:>6}")
    print("=" * 80)
    print(f"\nHAWPv2 (reference) WF sAP10 = 0.697; York sAP10 = 0.314")

    with open(ROOT + '/logs/wireframe_conf_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--train_limit', type=int, default=None,
                    help='use only first N train images (for quick test)')
    args = ap.parse_args()
    main(epochs=args.epochs, train_limit=args.train_limit)

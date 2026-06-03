"""
Full-dataset AFM baseline evaluation on YorkUrban (102 images).
Reports F-measure (AFM-style) and sAP-5/10/15 (LCNN-style).

This is our reproduction zero-point for all subsequent LR-SAF comparisons.

Run: python3 eval_afm_full.py
"""
import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')
os.chdir(ROOT + '/code/afm_baseline')

import torch
import numpy as np
import cv2

from config import cfg
cfg.merge_from_file('experiments/afm_atrous.yaml')
from modeling.net import build_network
from lib.squeeze_to_lsg import lsgenerator

from data import YorkUrbanSubset                     # noqa: E402
from metrics import f_measure, s_ap                  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def infer(model, img_bgr, in_res=320):
    H_o, W_o = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (in_res, in_res)).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    afm = out[0] if isinstance(out, (list, tuple)) else out
    offset = afm[0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)              # returns (lines, xx, yy)
    lines = np.asarray(lines)                       # [N, 5+]: (x1,y1,x2,y2,aspect_ratio)
    return lines, H_o, W_o


def main(aspect_thr=None, limit=None, out_json='eval_afm_yorkurban.json'):
    """aspect_thr: drop candidate lines with 5th-column value above this threshold.
       Note: squeeze module's 5th column is NOT aspect ratio in [0,1] as docs claim;
       observed range is [1, 27] for YorkUrban. Treating it as a quality metric
       (smaller = better) for sAP scoring. Filtering disabled by default."""
    print(f"=== AFM baseline eval on YorkUrban ===")
    print(f"aspect_thr filter: {aspect_thr}")

    # Build model + load pretrained
    model = build_network(cfg).cuda().eval()
    ckpt = torch.load(ROOT + '/checkpoints/atrous/weight/model_final.pth.tar',
                      map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt, strict=True)

    # Dataset
    ds = YorkUrbanSubset(root=ROOT + '/data/YorkUrbanDB', in_res=320, limit=limit)
    print(f"images to evaluate: {len(ds)}")

    rec = []
    aps_5, aps_10, aps_15, f_list = [], [], [], []
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        name, H_o, W_o = item['name'], item['H_orig'], item['W_orig']
        img = cv2.imread(os.path.join(ds.root, name, f"{name}.jpg"))

        lines_pred, _, _ = infer(model, img)
        # 5th column observed range [1, 27]; treat as quality metric (lower=better).
        # By default keep all; optionally filter by threshold.
        if lines_pred.shape[1] >= 5:
            metric = lines_pred[:, 4]
            keep_idx = (metric <= aspect_thr) if aspect_thr is not None \
                       else np.ones(len(lines_pred), dtype=bool)
            kept = lines_pred[keep_idx][:, :4]
            scores = 1.0 / (lines_pred[keep_idx][:, 4] + 1e-3)
        else:
            kept = lines_pred[:, :4]
            scores = np.ones(len(kept))

        # Scale predictions back to original size
        sx, sy = W_o / 320.0, H_o / 320.0
        kept_scaled = kept.copy()
        kept_scaled[:, 0::2] *= sx
        kept_scaled[:, 1::2] *= sy

        # GT lines in original-image coords
        gt = item['lines'].numpy()                  # currently in 320x320 coords
        gt_orig = gt.copy()
        gt_orig[:, 0::2] /= ds.in_res / W_o
        gt_orig[:, 1::2] /= ds.in_res / H_o

        # F-measure (pixel-level on a low-res 100x100 raster for speed)
        # We rasterize at original resolution which dominates cost; downsample first.
        DS = 4
        H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
        kept_ds = kept_scaled.copy(); kept_ds[:, 0::2] /= DS; kept_ds[:, 1::2] /= DS
        gt_ds = gt_orig.copy(); gt_ds[:, 0::2] /= DS; gt_ds[:, 1::2] /= DS
        fm = f_measure(kept_ds, gt_ds, H_ds, W_ds, tol_frac=0.01)

        # sAP at original resolution (sAP normalizes to 128x128 internally)
        aps = s_ap(kept_scaled, scores, gt_orig, H_o, W_o, thresholds=(5, 10, 15))

        f_list.append(fm['F'])
        aps_5.append(aps[5]); aps_10.append(aps[10]); aps_15.append(aps[15])
        rec.append({'name': name, 'F': fm['F'], 'P': fm['P'], 'R': fm['R'],
                    'n_pred': int(len(kept)), 'n_gt': int(len(gt_orig)),
                    'sAP_5': aps[5], 'sAP_10': aps[10], 'sAP_15': aps[15]})

        if (i + 1) % 10 == 0 or i + 1 == len(ds):
            elapsed = time.time() - t0
            print(f"  [{i+1:3d}/{len(ds)}] {name}: F={fm['F']:.3f} "
                  f"sAP10={aps[10]:.3f} (running mean F={np.mean(f_list):.3f}, "
                  f"sAP10={np.mean(aps_10):.3f}) elapsed {elapsed:.1f}s")

    # Aggregate
    res = {
        'aspect_thr': aspect_thr,
        'n_images': len(ds),
        'F_mean': float(np.mean(f_list)),
        'F_median': float(np.median(f_list)),
        'sAP_5_mean': float(np.mean(aps_5)),
        'sAP_10_mean': float(np.mean(aps_10)),
        'sAP_15_mean': float(np.mean(aps_15)),
        'msAP': float((np.mean(aps_5) + np.mean(aps_10) + np.mean(aps_15)) / 3),
        'per_image': rec,
    }

    os.makedirs(ROOT + '/logs', exist_ok=True)
    out_path = os.path.join(ROOT, 'logs', out_json)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved {out_path}")

    print("\n=== AGGREGATE ===")
    print(f"  F-measure  : {res['F_mean']:.4f}  (median {res['F_median']:.4f})")
    print(f"  sAP-5      : {res['sAP_5_mean']:.4f}")
    print(f"  sAP-10     : {res['sAP_10_mean']:.4f}")
    print(f"  sAP-15     : {res['sAP_15_mean']:.4f}")
    print(f"  msAP       : {res['msAP']:.4f}")
    print(f"\nAFM paper YorkUrban F = 0.646 (a-trous variant), reference.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--aspect_thr', type=float, default=None,
                    help='filter threshold on 5th column; default: no filtering')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out_json', type=str, default='eval_afm_yorkurban.json')
    args = ap.parse_args()
    main(aspect_thr=args.aspect_thr, limit=args.limit, out_json=args.out_json)

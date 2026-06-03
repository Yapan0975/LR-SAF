"""
Fair comparison: AFM baseline vs LR-SAF on the SAME val 22-image subset.
Uses the same seed/split as train_full.py so the 22-image set is identical.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')
os.chdir(ROOT + '/code/afm_baseline')

import torch, numpy as np, cv2
from config import cfg
cfg.merge_from_file('experiments/afm_atrous.yaml')
from modeling.net import build_network
from lib.squeeze_to_lsg import lsgenerator

from data import YorkUrbanSubset
from model import build_lr_saf
from metrics import f_measure, s_ap

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEED = 42


def get_val_items():
    """Reproduce the same 22-image val subset as train_full.py."""
    ds = YorkUrbanSubset(in_res=320, limit=None)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ds))
    val_idx = perm[80:].tolist()
    items = []
    for vi in val_idx:
        item = ds[vi]
        gt_orig = item['lines'].numpy().copy()
        gt_orig[:, 0::2] *= item['W_orig'] / 320.0
        gt_orig[:, 1::2] *= item['H_orig'] / 320.0
        items.append({'name': item['name'], 'root': ds.root, 'gt_orig': gt_orig,
                      'H_orig': item['H_orig'], 'W_orig': item['W_orig']})
    return items


def infer_and_score(model, item, network_kind='afm'):
    name = item['name']; H_o = item['H_orig']; W_o = item['W_orig']
    img = cv2.imread(os.path.join(item['root'], name, f"{name}.jpg"))
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    if network_kind == 'afm':
        afm = out[0] if isinstance(out, (list, tuple)) else out
        offset = afm[0].cpu().numpy().astype(np.float32)
    elif network_kind == 'lr_saf':
        offset = out['a'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    lines = np.asarray(lines)
    if len(lines) == 0:
        return 0.0, 0.0, 0
    sx, sy = W_o / 320.0, H_o / 320.0
    kept = lines[:, :4].copy()
    kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
    scores = 1.0 / (lines[:, 4] + 1e-3) if lines.shape[1] >= 5 \
             else np.ones(len(kept))
    DS = 4
    H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
    kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
    gp = item['gt_orig'].copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
    fm = f_measure(kp, gp, H_ds, W_ds)
    aps = s_ap(kept, scores, item['gt_orig'], H_o, W_o, thresholds=(5, 10, 15))
    return fm['F'], aps, len(kept)


def main():
    val_items = get_val_items()
    print(f"val 22-image subset (seed={SEED})")

    # 1) AFM baseline (no fine-tuning)
    afm = build_network(cfg).cuda().eval()
    ckpt = torch.load(ROOT + '/checkpoints/atrous/weight/model_final.pth.tar',
                      map_location='cuda', weights_only=False)
    afm.load_state_dict(ckpt, strict=True)

    # 2) LR-SAF best
    lr_saf = build_lr_saf(device='cuda').eval()
    lr_ckpt = torch.load(ROOT + '/checkpoints/lr_saf_best.pth', map_location='cuda',
                         weights_only=False)
    lr_saf.load_state_dict(lr_ckpt['model'], strict=True)
    print(f"loaded LR-SAF best (epoch {lr_ckpt['epoch']}, val F was {lr_ckpt['val_F']:.4f})")

    results = {'afm_baseline': [], 'lr_saf': []}
    for item in val_items:
        F_a, ap_a, n_a = infer_and_score(afm, item, 'afm')
        F_l, ap_l, n_l = infer_and_score(lr_saf, item, 'lr_saf')
        results['afm_baseline'].append({'F': F_a, **{f'sAP_{k}': v for k, v in ap_a.items()}, 'n': n_a})
        results['lr_saf'].append({'F': F_l, **{f'sAP_{k}': v for k, v in ap_l.items()}, 'n': n_l})

    def agg(rs):
        return {
            'F_mean':     float(np.mean([r['F'] for r in rs])),
            'sAP_5':      float(np.mean([r['sAP_5'] for r in rs])),
            'sAP_10':     float(np.mean([r['sAP_10'] for r in rs])),
            'sAP_15':     float(np.mean([r['sAP_15'] for r in rs])),
            'n_pred':     float(np.mean([r['n'] for r in rs])),
        }

    afm_agg = agg(results['afm_baseline'])
    lr_agg = agg(results['lr_saf'])

    print("\n=== HEAD-TO-HEAD on 22 val images (same subset) ===")
    print(f"{'metric':<10} {'AFM base':>10} {'LR-SAF':>10} {'delta':>10}")
    for k in ['F_mean', 'sAP_5', 'sAP_10', 'sAP_15', 'n_pred']:
        d = lr_agg[k] - afm_agg[k]
        print(f"{k:<10} {afm_agg[k]:>10.4f} {lr_agg[k]:>10.4f} {d:>+10.4f}")

    out_json = ROOT + '/logs/eval_compare.json'
    with open(out_json, 'w') as f:
        json.dump({'afm': afm_agg, 'lr_saf': lr_agg, 'per_image': results},
                  f, indent=2)
    print(f"\nsaved {out_json}")


if __name__ == '__main__':
    main()

"""4-way robustness eval: AFM / LR-SAF-noTNNR / LR-SAF-full / LR-SAF+sem
Goal: isolate TNNR's contribution under degradation (peer-review demand).
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
from confidence_head import ConfidenceMLP, compute_segment_features
from train_confidence_semantic import SemanticConfMLP
from semantic_features import SemanticExtractor, sample_along_line
from metrics import s_ap, image_records, sap_dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_deg(img, kind, lvl):
    img = img.astype(np.float32)
    if kind == 'noise':
        if lvl == 0: return img.astype(np.uint8)
        return np.clip(img + np.random.randn(*img.shape).astype(np.float32) * lvl, 0, 255).astype(np.uint8)
    if kind == 'blur':
        if lvl == 0: return img.astype(np.uint8)
        k = int(lvl); ker = np.zeros((k, k), dtype=np.float32); ker[k // 2, :] = 1.0 / k
        return cv2.filter2D(img.astype(np.uint8), -1, ker)
    if kind == 'lowlight':
        return np.clip(img * lvl, 0, 255).astype(np.uint8)


def infer_afm(model, img):
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    afm = out[0] if isinstance(out, (list, tuple)) else out
    offset = afm[0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), H_o, W_o


def infer_lrsaf(model, sem_ext, img, use_sem=False):
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x_n = (x - MEAN) / STD
    x_t = torch.from_numpy(x_n).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x_t)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    lines = np.asarray(lines)
    if len(lines) == 0:
        return lines, None, None, H_o, W_o
    geom = compute_segment_features(lines, offset, junc, H=320, W=320).cuda()
    sem = None
    if use_sem and sem_ext is not None:
        with torch.no_grad():
            sem_map = sem_ext(x_t)[0]
            sem = sample_along_line(sem_map, lines[:, :4], n_samples=16).cuda()
    return lines, geom, sem, H_o, W_o


def rec(lines, scores, gt, H_o, W_o):
    """Per-image record (128-frame) for DATASET-LEVEL sAP (standard LCNN)."""
    if len(lines) == 0:
        return image_records(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                             gt, H_o, W_o)
    sx, sy = W_o / 320.0, H_o / 320.0
    kept = lines[:, :4].copy(); kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
    return image_records(kept, np.asarray(scores).reshape(-1), gt, H_o, W_o)


def main(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = YorkUrbanSubset(in_res=320)
    rng = np.random.RandomState(seed); perm = rng.permutation(len(ds))
    val_idx = perm[80:].tolist()
    val_items = []
    for vi in val_idx:
        item = ds[vi]
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= item['W_orig'] / 320.0; gt[:, 1::2] *= item['H_orig'] / 320.0
        val_items.append({'name': item['name'], 'root': ds.root, 'gt': gt,
                           'H_o': item['H_orig'], 'W_o': item['W_orig']})
    print(f"val 22 images")

    # Load 4 models
    afm = build_network(cfg).cuda().eval()
    afm.load_state_dict(torch.load(ROOT + '/checkpoints/atrous/weight/model_final.pth.tar',
                                    map_location='cuda', weights_only=False), strict=True)
    notnnr = build_lr_saf(device='cuda').eval()
    notnnr.load_state_dict(torch.load(ROOT + '/checkpoints/lr_saf_no_tnnr.pth',
                                       map_location='cuda', weights_only=False)['model'], strict=True)
    full = build_lr_saf(device='cuda').eval()
    full.load_state_dict(torch.load(ROOT + '/checkpoints/lr_saf_best.pth',
                                     map_location='cuda', weights_only=False)['model'], strict=True)
    sem_ext = SemanticExtractor().cuda().eval()
    geom_head = ConfidenceMLP(in_dim=9, hidden=64).cuda().eval()
    geom_head.load_state_dict(torch.load(ROOT + '/checkpoints/lr_saf_conf_head.pth',
                                          weights_only=False)['head'])
    sem_head = SemanticConfMLP(geom_dim=9, sem_dim=21, hidden=64).cuda().eval()
    sem_head.load_state_dict(torch.load(ROOT + '/checkpoints/lr_saf_conf_head_sem.pth',
                                         weights_only=False)['head'])
    print("models loaded")

    # Degradation grid (subset of original: 3 levels per axis to focus on the discriminative range)
    grid = {
        'clean':    [(None, 1.0)],
        'noise':    [('noise', 10), ('noise', 35)],
        'blur':     [('blur', 5), ('blur', 9), ('blur', 17)],
        'lowlight': [('lowlight', 0.6), ('lowlight', 0.15)],
    }
    results = []
    for category, conds in grid.items():
        for (kind, lvl) in conds:
            print(f"  {category} lvl={lvl}", end=' ', flush=True)
            agg = {'afm': [], 'notnnr_g': [], 'full_g': [], 'full_sem': []}
            for item in val_items:
                img = cv2.imread(os.path.join(item['root'], item['name'],
                                                f"{item['name']}.jpg"))
                img_d = apply_deg(img, kind, lvl) if kind else img
                # 1. AFM (1/aspect score)
                lines_a, H_o, W_o = infer_afm(afm, img_d)
                sc = (1.0 / (lines_a[:, 4] + 1e-3) if len(lines_a) and lines_a.shape[1] >= 5
                      else np.ones(len(lines_a)))
                agg['afm'].append(rec(lines_a, sc, item['gt'], H_o, W_o))
                # 2. LR-SAF no TNNR + geom head
                lines_n, geom_n, _, _, _ = infer_lrsaf(notnnr, None, img_d, use_sem=False)
                sc_n = (geom_head(geom_n).detach().cpu().numpy() if len(lines_n) else None)
                agg['notnnr_g'].append(rec(lines_n, sc_n, item['gt'], H_o, W_o))
                # 3. LR-SAF full + geom head
                lines_f, geom_f, _, _, _ = infer_lrsaf(full, None, img_d, use_sem=False)
                sc_f = (geom_head(geom_f).detach().cpu().numpy() if len(lines_f) else None)
                agg['full_g'].append(rec(lines_f, sc_f, item['gt'], H_o, W_o))
                # 4. LR-SAF full + sem head
                lines_s, geom_s, sem_s, _, _ = infer_lrsaf(full, sem_ext, img_d, use_sem=True)
                sc_s = (sem_head(geom_s, sem_s).detach().cpu().numpy() if len(lines_s) else None)
                agg['full_sem'].append(rec(lines_s, sc_s, item['gt'], H_o, W_o))
            row = {'category': category, 'level': lvl,
                   'afm':       sap_dataset(agg['afm'], (10,))[10],
                   'notnnr_g':  sap_dataset(agg['notnnr_g'], (10,))[10],
                   'full_g':    sap_dataset(agg['full_g'], (10,))[10],
                   'full_sem':  sap_dataset(agg['full_sem'], (10,))[10]}
            results.append(row)
            print(f"-> AFM={row['afm']:.3f} noTNNR={row['notnnr_g']:.3f} "
                  f"full={row['full_g']:.3f} +sem={row['full_sem']:.3f}")

    out = ROOT + '/logs/tnnr_causal_ablation.json'
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nsaved {out}")

    # Summary: TNNR delta = full_g - notnnr_g
    print("\n=== TNNR CAUSAL DELTA (full - noTNNR, both with geom head) ===")
    for r in results:
        d = r['full_g'] - r['notnnr_g']
        print(f"  {r['category']} lvl={r['level']}: Δ_TNNR = {d:+.4f}")


if __name__ == '__main__':
    main()

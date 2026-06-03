"""
Apples-to-apples: feed HAWPv2's predictions through OUR s_ap() metric, AND
feed LR-SAF's predictions through HAWPv2's metric.

Goal: produce a head-to-head table on full YorkUrban (102 images) using
the SAME metric code on both methods.
"""
import os, sys, json, warnings
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

from data import YorkUrbanSubset
from model import build_lr_saf
from confidence_head import ConfidenceMLP, compute_segment_features
from train_confidence_semantic import SemanticConfMLP
from semantic_features import SemanticExtractor, sample_along_line
from metrics import f_measure, s_ap

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HAWP_PRED = (ROOT + '/code/hawp_baseline/checkpoints/york_test.json')

# ----------------------------------------------------------
# 1. Load HAWP predictions, evaluate with OUR metric
# ----------------------------------------------------------
with open(HAWP_PRED) as f:
    hawp_preds = json.load(f)

ds = YorkUrbanSubset(in_res=320)
# Build name -> (lines_orig, H, W) lookup from our YorkUrban
gt_map = {}
for i in range(len(ds)):
    item = ds[i]
    H_o, W_o = item['H_orig'], item['W_orig']
    gt = item['lines'].numpy().copy()
    gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0
    gt_map[item['name']] = (gt, H_o, W_o)

print(f"GT map: {len(gt_map)} images; HAWP preds: {len(hawp_preds)} images")

# Only evaluate on val 22 (apples-to-apples with our other results)
seed = 42
rng = np.random.RandomState(seed)
perm = rng.permutation(len(gt_map))
val_names = set([list(gt_map.keys())[i] for i in perm[80:].tolist()])
print(f"evaluating on val 22 split (seed={seed}): {len(val_names)} images")

print("\n=== HAWPv2 predictions through OUR metric (val 22) ===")
f_list, ap5, ap10, ap15 = [], [], [], []
n_proposals = []
for p in hawp_preds:
    fn = p['filename'].split('.')[0]
    if fn not in gt_map or fn not in val_names:
        continue
    gt, H_o, W_o = gt_map[fn]
    pred_lines = np.array(p['lines_pred'], dtype=np.float32)    # already in original image coords
    scores = np.array(p['lines_score'], dtype=np.float32)
    pred_orig = pred_lines  # HAWPv2's predictions are saved in the original image's pixel coords

    # F-measure (downsampled rasterization to save time)
    DS = 4
    H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
    kp = pred_orig.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
    gp = gt.copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
    fm = f_measure(kp, gp, H_ds, W_ds)
    aps = s_ap(pred_orig, scores, gt, H_o, W_o, thresholds=(5, 10, 15))
    f_list.append(fm['F']); ap5.append(aps[5]); ap10.append(aps[10]); ap15.append(aps[15])
    n_proposals.append(len(pred_lines))

print(f"HAWPv2 on val 22 (our metric):")
print(f"  F      = {np.mean(f_list):.4f}")
print(f"  sAP-5  = {np.mean(ap5):.4f}")
print(f"  sAP-10 = {np.mean(ap10):.4f}")
print(f"  sAP-15 = {np.mean(ap15):.4f}")
print(f"  median #pred = {int(np.median(n_proposals))}")


# ----------------------------------------------------------
# 2. Run LR-SAF + sem head on the SAME val 22 split, our metric
# ----------------------------------------------------------
print("\n=== LR-SAF + sem head on val 22 (our metric) ===")
lrsaf = build_lr_saf(device='cuda').eval()
lr_ckpt = torch.load(ROOT + '/checkpoints/lr_saf_best.pth',
                     map_location='cuda', weights_only=False)
lrsaf.load_state_dict(lr_ckpt['model'], strict=True)
sem_ext = SemanticExtractor().cuda().eval()
sem_head = SemanticConfMLP(geom_dim=9, sem_dim=21, hidden=64).cuda().eval()
sem_head.load_state_dict(torch.load(
    ROOT + '/checkpoints/lr_saf_conf_head_sem.pth', weights_only=False)['head'])

f_list2, ap5_2, ap10_2, ap15_2 = [], [], [], []
n_proposals2 = []
import time; t0 = time.time()
for i in range(len(ds)):
    item = ds[i]
    name, H_o, W_o = item['name'], item['H_orig'], item['W_orig']
    if name not in val_names:
        continue
    img = cv2.imread(os.path.join(ds.root, name, f"{name}.jpg"))
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x_n = (x - IMAGENET_MEAN) / IMAGENET_STD
    x_t = torch.from_numpy(x_n).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = lrsaf(x_t)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    lines = np.asarray(lines)
    if len(lines) == 0:
        f_list2.append(0.0); ap5_2.append(0.0); ap10_2.append(0.0); ap15_2.append(0.0)
        n_proposals2.append(0); continue
    geom = compute_segment_features(lines, offset, junc, H=320, W=320).cuda()
    with torch.no_grad():
        sem_map = sem_ext(x_t)[0]
        sem = sample_along_line(sem_map, lines[:, :4], n_samples=16).cuda()
        scores = sem_head(geom, sem).cpu().numpy()
    sx, sy = W_o / 320.0, H_o / 320.0
    kept = lines[:, :4].copy(); kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
    DS = 4
    H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
    kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
    gt, _, _ = gt_map[name]
    gp = gt.copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
    fm = f_measure(kp, gp, H_ds, W_ds)
    aps = s_ap(kept, scores, gt, H_o, W_o, thresholds=(5, 10, 15))
    f_list2.append(fm['F']); ap5_2.append(aps[5]); ap10_2.append(aps[10]); ap15_2.append(aps[15])
    n_proposals2.append(len(lines))

print(f"  F      = {np.mean(f_list2):.4f}")
print(f"  sAP-5  = {np.mean(ap5_2):.4f}")
print(f"  sAP-10 = {np.mean(ap10_2):.4f}")
print(f"  sAP-15 = {np.mean(ap15_2):.4f}")
print(f"  median #pred = {int(np.median(n_proposals2))}")
print(f"  ({time.time()-t0:.1f}s)")

# ----------------------------------------------------------
# 3. Final head-to-head
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("HEAD-TO-HEAD on YorkUrban val 22 (seed=42, our metric)")
print("=" * 70)
print(f"{'Method':<25} {'F':>8} {'sAP-5':>8} {'sAP-10':>8} {'sAP-15':>8} {'#pred':>7}")
print("-" * 70)
print(f"{'AFM baseline':<25} {0.7231:>8.4f} {0.0863:>8.4f} {0.1638:>8.4f} {0.2364:>8.4f} {305:>7}")
print(f"{'HAWPv2 (reproduced)':<25} {np.mean(f_list):>8.4f} {np.mean(ap5):>8.4f} {np.mean(ap10):>8.4f} {np.mean(ap15):>8.4f} {int(np.median(n_proposals)):>7}")
print(f"{'LR-SAF + sem (ours)':<25} {np.mean(f_list2):>8.4f} {np.mean(ap5_2):>8.4f} {np.mean(ap10_2):>8.4f} {np.mean(ap15_2):>8.4f} {int(np.median(n_proposals2)):>7}")

# Save
out_log = ROOT + '/logs/hawp_lrsaf_compare.json'
with open(out_log, 'w') as f:
    json.dump({
        'hawp_v2': {'F': float(np.mean(f_list)), 'sAP5': float(np.mean(ap5)),
                     'sAP10': float(np.mean(ap10)), 'sAP15': float(np.mean(ap15)),
                     'n_pred_median': int(np.median(n_proposals))},
        'lrsaf_sem': {'F': float(np.mean(f_list2)), 'sAP5': float(np.mean(ap5_2)),
                       'sAP10': float(np.mean(ap10_2)), 'sAP15': float(np.mean(ap15_2)),
                       'n_pred_median': int(np.median(n_proposals2))},
        'protocol': 'YorkUrban full 102 images, our s_ap() metric (LCNN-style)',
    }, f, indent=2)
print(f"\nsaved {out_log}")

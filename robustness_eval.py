"""
Cross-domain robustness evaluation.

For each (degradation, level), run all three configurations:
  - AFM baseline (no fine-tune)
  - LR-SAF (geometry-only confidence head)
  - LR-SAF + Semantic (geom+VOC confidence head)
Reports F + sAP-10 averaged over val 22.

Degradations:
  - Gaussian noise:   sigma in {0, 10, 20, 35, 50}   (pixel scale 0-255)
  - Motion blur:      kernel size in {0, 5, 9, 13, 17}
  - Low light:        brightness multiplier in {1.0, 0.6, 0.3, 0.15, 0.05}

Output:
  - JSON with all curves
  - Plots saved to logs/ (matplotlib)
"""
import os, sys, json, time, warnings
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
from confidence_head import ConfidenceMLP, compute_segment_features, match_to_gt
from train_confidence_semantic import SemanticConfMLP
from semantic_features import SemanticExtractor, sample_along_line
from metrics import f_measure, s_ap, image_records, sap_dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_degradation(img_bgr, kind, level):
    img = img_bgr.astype(np.float32)
    if kind == 'noise':
        if level == 0:
            return img_bgr
        noise = np.random.randn(*img.shape).astype(np.float32) * level
        return np.clip(img + noise, 0, 255).astype(np.uint8)
    elif kind == 'blur':
        if level == 0:
            return img_bgr
        # Motion blur: horizontal line kernel
        k = int(level)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img_bgr, -1, kernel)
    elif kind == 'lowlight':
        return np.clip(img * level, 0, 255).astype(np.uint8)
    raise ValueError(kind)


def infer_afm(model, img):
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    afm = out[0] if isinstance(out, (list, tuple)) else out
    offset = afm[0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), H_o, W_o


def infer_lrsaf(model, sem_ext, img):
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x_n = (x - IMAGENET_MEAN) / IMAGENET_STD
    x_t = torch.from_numpy(x_n).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x_t)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    lines = np.asarray(lines)
    if len(lines) == 0:
        return lines, None, None, H_o, W_o
    geom = compute_segment_features(lines, offset, junc, H=320, W=320)
    sem = None
    if sem_ext is not None:
        with torch.no_grad():
            sem_map = sem_ext(x_t)[0]
        sem = sample_along_line(sem_map, lines[:, :4], n_samples=16)
    return lines, geom.cuda(), (sem.cuda() if sem is not None else None), H_o, W_o


def score_and_eval(lines, scores, gt, H_o, W_o, ds_in_res=320):
    """Return (F-measure, per-image record). F stays per-image (its convention);
    sAP is accumulated as records and scored DATASET-LEVEL by the caller."""
    if len(lines) == 0:
        return 0.0, image_records(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), gt, H_o, W_o)
    sx, sy = W_o / ds_in_res, H_o / ds_in_res
    kept = lines[:, :4].copy()
    kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
    DS = 4
    H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
    kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
    gp = gt.copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
    fm = f_measure(kp, gp, H_ds, W_ds)
    rec = image_records(kept, np.asarray(scores).reshape(-1), gt, H_o, W_o)
    return fm['F'], rec


def main(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    # Val items (same as elsewhere)
    ds = YorkUrbanSubset(in_res=320)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds))
    val_idx = perm[80:].tolist()
    val_items = []
    for vi in val_idx:
        item = ds[vi]
        gt_orig = item['lines'].numpy().copy()
        gt_orig[:, 0::2] *= item['W_orig'] / 320.0
        gt_orig[:, 1::2] *= item['H_orig'] / 320.0
        val_items.append({'name': item['name'], 'root': ds.root,
                          'gt_orig': gt_orig,
                          'H_orig': item['H_orig'], 'W_orig': item['W_orig']})
    print(f"val 22 images, robustness eval seed={seed}")

    # === Load all three models ===
    print("\nLoading models...")
    # 1. AFM baseline
    afm_model = build_network(cfg).cuda().eval()
    ckpt = torch.load(ROOT + '/checkpoints/atrous/weight/model_final.pth.tar',
                      map_location='cuda', weights_only=False)
    afm_model.load_state_dict(ckpt, strict=True)

    # 2. LR-SAF main + geom-only confidence
    lrsaf_model = build_lr_saf(device='cuda').eval()
    lc = torch.load(ROOT + '/checkpoints/lr_saf_best.pth',
                    map_location='cuda', weights_only=False)
    lrsaf_model.load_state_dict(lc['model'], strict=True)
    geom_head = ConfidenceMLP(in_dim=9, hidden=64).cuda().eval()
    geom_head.load_state_dict(torch.load(
        ROOT + '/checkpoints/lr_saf_conf_head.pth',
        weights_only=False)['head'])

    # 3. LR-SAF + semantic head
    sem_ext = SemanticExtractor().cuda().eval()
    sem_head = SemanticConfMLP(geom_dim=9, sem_dim=21, hidden=64).cuda().eval()
    s_ckpt = torch.load(ROOT + '/checkpoints/lr_saf_conf_head_sem.pth',
                        weights_only=False)
    sem_head.load_state_dict(s_ckpt['head'])
    print("  models loaded\n")

    # === Degradations grid ===
    grid = {
        'noise':    [0, 10, 20, 35, 50],
        'blur':     [0, 5, 9, 13, 17],
        'lowlight': [1.0, 0.6, 0.3, 0.15, 0.05],
    }
    results = {}

    for kind, levels in grid.items():
        print(f"=== {kind} ===")
        results[kind] = []
        for lvl in levels:
            print(f"  level={lvl}", end=' ', flush=True)
            f_afm, rec_afm, f_lrs, rec_lrs, f_sem, rec_sem = [], [], [], [], [], []
            t0 = time.time()
            for item in val_items:
                img_orig = cv2.imread(os.path.join(item['root'], item['name'],
                                                    f"{item['name']}.jpg"))
                img_d = apply_degradation(img_orig, kind, lvl)
                # 1. AFM baseline (1/aspect score)
                lines_a, H_o, W_o = infer_afm(afm_model, img_d)
                scores_a = (1.0 / (lines_a[:, 4] + 1e-3)
                            if len(lines_a) and lines_a.shape[1] >= 5 else np.ones(len(lines_a)))
                f1, r1 = score_and_eval(lines_a, scores_a, item['gt_orig'], H_o, W_o)
                f_afm.append(f1); rec_afm.append(r1)

                # 2. LR-SAF + geom-only confidence
                lines_l, geom, sem, H_o, W_o = infer_lrsaf(lrsaf_model, sem_ext, img_d)
                sc_g = geom_head(geom).detach().cpu().numpy() if len(lines_l) else np.zeros(0)
                f2, r2 = score_and_eval(lines_l, sc_g, item['gt_orig'], H_o, W_o)
                f_lrs.append(f2); rec_lrs.append(r2)

                # 3. LR-SAF + semantic
                sc_s = sem_head(geom, sem).detach().cpu().numpy() if len(lines_l) else np.zeros(0)
                f3, r3 = score_and_eval(lines_l, sc_s, item['gt_orig'], H_o, W_o)
                f_sem.append(f3); rec_sem.append(r3)

            row = {
                'level': lvl,
                'F_afm':     float(np.mean(f_afm)),
                'F_lrsaf':   float(np.mean(f_lrs)),
                'F_lrsaf_s': float(np.mean(f_sem)),
                'sAP10_afm':     sap_dataset(rec_afm, (10,))[10],
                'sAP10_lrsaf':   sap_dataset(rec_lrs, (10,))[10],
                'sAP10_lrsaf_s': sap_dataset(rec_sem, (10,))[10],
            }
            results[kind].append(row)
            print(f"-> F: AFM={row['F_afm']:.3f} LR={row['F_lrsaf']:.3f} "
                  f"LR+sem={row['F_lrsaf_s']:.3f} | "
                  f"sAP10: {row['sAP10_afm']:.3f} {row['sAP10_lrsaf']:.3f} "
                  f"{row['sAP10_lrsaf_s']:.3f}  ({time.time()-t0:.1f}s)")
        print()

    # Save
    out_path = ROOT + '/logs/robustness.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")

    # Plot curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ci, kind in enumerate(['noise', 'blur', 'lowlight']):
            xs = [r['level'] for r in results[kind]]
            for ri, metric in enumerate(['F', 'sAP10']):
                ax = axes[ri, ci]
                ax.plot(xs, [r[f'{metric}_afm']     for r in results[kind]],
                         'o-', label='AFM baseline', color='gray')
                ax.plot(xs, [r[f'{metric}_lrsaf']   for r in results[kind]],
                         's-', label='LR-SAF (geom)', color='C0')
                ax.plot(xs, [r[f'{metric}_lrsaf_s'] for r in results[kind]],
                         '^-', label='LR-SAF + semantic', color='C2')
                ax.set_title(f'{kind} - {metric}')
                ax.set_xlabel('degradation level')
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                if ri == 0 and ci == 0:
                    ax.legend()
        plt.tight_layout()
        plot_path = ROOT + '/logs/robustness_curves.png'
        plt.savefig(plot_path, dpi=120)
        print(f"saved {plot_path}")
    except Exception as e:
        print(f"plotting failed: {e}")


if __name__ == '__main__':
    main()

"""
Run DeepLSD inference on YorkUrban val 22, compute our F + sAP metrics.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/deeplsd_baseline')

import torch
import numpy as np
import cv2
from omegaconf import OmegaConf

from deeplsd.models.deeplsd_inference import DeepLSD
from data import YorkUrbanSubset
from metrics import f_measure, s_ap


DEEPLSD_CKPT = ROOT + '/code/deeplsd_baseline/weights/deeplsd_wireframe.tar'


def main(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    # Build DeepLSD
    conf = {
        'detect_lines': True,
        'line_detection_params': {
            'merge': False,
            'filtering': True,
            'grad_nfa': True,
            'grad_thresh': 3,
        },
    }
    model = DeepLSD(conf).cuda().eval()
    ckpt = torch.load(DEEPLSD_CKPT, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    print("DeepLSD model loaded")

    # YorkUrban val 22 (same seed/split as everywhere)
    ds = YorkUrbanSubset(in_res=320)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds))
    val_idx = perm[80:].tolist()
    print(f"val 22 images")

    f_list, ap5, ap10, ap15 = [], [], [], []
    n_pred = []
    t0 = time.time()
    for vi in val_idx:
        item = ds[vi]
        H_o, W_o = item['H_orig'], item['W_orig']
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        # DeepLSD uses grayscale tensor
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model({'image': t})
        # out['lines'] is list of [N, 2, 2] for each batch
        lines = out['lines'][0]                              # [N, 2, 2]
        if isinstance(lines, torch.Tensor):
            lines = lines.cpu().numpy()
        else:
            lines = np.asarray(lines)
        if len(lines) == 0:
            f_list.append(0.0); ap5.append(0.0); ap10.append(0.0); ap15.append(0.0)
            n_pred.append(0); continue
        # Flatten endpoints: [N, 4]
        pred_lines = lines.reshape(-1, 4)
        # DeepLSD outputs in original image coords
        scores = (out.get('line_scores', [np.ones(len(pred_lines))])[0]
                  if 'line_scores' in out else np.ones(len(pred_lines)))
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        # GT in original coords
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0
        gt[:, 1::2] *= H_o / 320.0

        DS = 4
        H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
        kp = pred_lines.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
        gp = gt.copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
        fm = f_measure(kp, gp, H_ds, W_ds)
        aps = s_ap(pred_lines, scores, gt, H_o, W_o, thresholds=(5, 10, 15))
        f_list.append(fm['F']); ap5.append(aps[5]); ap10.append(aps[10]); ap15.append(aps[15])
        n_pred.append(len(pred_lines))

    elapsed = time.time() - t0
    print(f"\n=== DeepLSD on YorkUrban val 22 (our metric) ===")
    print(f"  F     = {np.mean(f_list):.4f}")
    print(f"  sAP-5 = {np.mean(ap5):.4f}")
    print(f"  sAP-10= {np.mean(ap10):.4f}")
    print(f"  sAP-15= {np.mean(ap15):.4f}")
    print(f"  median #pred = {int(np.median(n_pred))}")
    print(f"  total time = {elapsed:.1f}s ({22/elapsed*60:.1f} fpm)")

    out_json = ROOT + '/logs/deeplsd_yorkurban.json'
    with open(out_json, 'w') as f:
        json.dump({
            'F': float(np.mean(f_list)), 'sAP5': float(np.mean(ap5)),
            'sAP10': float(np.mean(ap10)), 'sAP15': float(np.mean(ap15)),
            'n_pred_median': int(np.median(n_pred)),
        }, f, indent=2)
    print(f"saved {out_json}")


if __name__ == '__main__':
    main()

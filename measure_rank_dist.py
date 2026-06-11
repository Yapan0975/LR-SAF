"""
Measure the EMPIRICAL distribution of the adaptive truncation rank r(W)
over real windows, using the deployed LR-SAF backbone's predicted junction
map (exactly the signal tnn_loss.py uses at training time).

Reviewer concern: with K_max=3, r(W)=1+floor((K_max-1)*s(W))=1+floor(2*s),
and the window-mean junction strength s rarely reaches 0.5 (r=2) or 1.0
(r=3), so the advertised three-level adaptive rank may be degenerate.
This script reports the actual r=1/2/3 frequencies and the s distribution.

Run on the server. Deterministic.
"""
import os, sys, json, argparse
import numpy as np
import torch
import cv2

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

from data import YorkUrbanSubset
from model import build_lr_saf

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main(window=9, stride=8, K_max=3, ckpt=None, out_json=None):
    ds = YorkUrbanSubset(in_res=320)
    model = build_lr_saf(device='cuda').eval()
    c = torch.load(ckpt or (ROOT + '/checkpoints/lr_saf_best.pth'),
                   map_location='cuda', weights_only=False)
    model.load_state_dict(c['model'] if isinstance(c, dict) and 'model' in c else c,
                          strict=True)

    counts = np.zeros(K_max + 1, dtype=np.int64)   # counts[r]
    s_all = []
    n_img = 0
    for i in range(len(ds)):
        item = ds[i]
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(xt)
        junc = out['junc'][0, 0].cpu().numpy()      # [320,320] in [0,1]
        H, W = junc.shape
        for r0 in range(0, H - window + 1, stride):
            for c0 in range(0, W - window + 1, stride):
                s = float(junc[r0:r0 + window, c0:c0 + window].mean())
                s = min(max(s, 0.0), 1.0)
                r = 1 + int(np.floor((K_max - 1) * s))
                r = min(r, window - 1)
                counts[r] += 1
                s_all.append(s)
        n_img += 1

    total = int(counts.sum())
    s_all = np.array(s_all)
    print(f"images={n_img}, windows={total}, K_max={K_max}, window={window}, stride={stride}")
    print(f"s (window-mean junction): mean={s_all.mean():.4f} median={np.median(s_all):.4f} "
          f"max={s_all.max():.4f} p99={np.percentile(s_all,99):.4f}")
    print(f"frac(s>=0.5) [r>=2] = {float((s_all>=0.5).mean()):.4f}; "
          f"frac(s>=1.0) [r=3]  = {float((s_all>=1.0).mean()):.4f}")
    for r in range(1, K_max + 1):
        print(f"  r={r}: {int(counts[r])} ({100*counts[r]/total:.2f}%)")

    if out_json:
        rec = {'images': n_img, 'windows': total, 'K_max': K_max,
               'window': window, 'stride': stride,
               's_mean': float(s_all.mean()), 's_median': float(np.median(s_all)),
               's_max': float(s_all.max()),
               'frac_s_ge_0.5': float((s_all >= 0.5).mean()),
               'frac_s_ge_1.0': float((s_all >= 1.0).mean()),
               'dist': {f'r={r}': {'count': int(counts[r]),
                                   'frac': float(counts[r] / total)}
                        for r in range(1, K_max + 1)}}
        with open(out_json, 'w') as fh:
            json.dump(rec, fh, indent=2)
        print(f"saved {out_json}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_json', type=str, default=None)
    args = ap.parse_args()
    main(out_json=args.out_json)

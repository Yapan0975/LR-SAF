"""
Dump per-image records (predictions + 1/aspect score + GT, in the 128 frame)
for a backbone-only model, and report dataset-level sAP under the corrected
LCNN metric. Used to recompute backbone-only sAP for the CV folds.
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
from metrics import image_records, sap_dataset
from lib.squeeze_to_lsg import lsgenerator

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cv_test_idx(n, fold, nfolds):
    parts = np.array_split(np.random.RandomState(0).permutation(n), nfolds)
    return parts[fold].tolist()


def load_backbone(kind, ckpt):
    if kind == 'lrsaf':
        from model import build_lr_saf
        m = build_lr_saf(device='cuda').eval()
        c = torch.load(ckpt, map_location='cuda', weights_only=False)
        m.load_state_dict(c['model'] if isinstance(c, dict) and 'model' in c else c, strict=True)
    else:
        _cwd = os.getcwd(); os.chdir(ROOT + '/code/afm_baseline')
        from config import cfg
        cfg.merge_from_file('experiments/afm_atrous.yaml')
        from modeling.net import build_network
        os.chdir(_cwd)
        m = build_network(cfg).cuda().eval()
        c = torch.load(ckpt, map_location='cuda', weights_only=False)
        m.load_state_dict(c['model'] if isinstance(c, dict) and 'model' in c else c, strict=True)
    return m


def infer(m, kind, img):
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = m(xt)
    if kind == 'lrsaf':
        offset = out['a'][0].cpu().numpy().astype(np.float32)
    else:
        afm = out[0] if isinstance(out, (list, tuple)) else out
        offset = afm[0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), H_o, W_o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['lrsaf', 'afm'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--fold', type=int, default=-1)
    ap.add_argument('--nfolds', type=int, default=5)
    ap.add_argument('--out_json', required=True)
    a = ap.parse_args()

    ds = YorkUrbanSubset(in_res=320)
    if a.fold >= 0:
        val_idx = cv_test_idx(len(ds), a.fold, a.nfolds)
    else:
        val_idx = np.random.RandomState(42).permutation(len(ds))[80:].tolist()

    m = load_backbone(a.kind, a.ckpt)
    recs = []
    for vi in val_idx:
        item = ds[vi]
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        lines, H_o, W_o = infer(m, a.kind, img)
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0
        if len(lines) == 0:
            r = image_records(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), gt, H_o, W_o)
        else:
            kept = lines[:, :5].copy()
            kept[:, 0::2] *= W_o / 320.0; kept[:, 1::2] *= H_o / 320.0
            score = 1.0 / (kept[:, 4] + 1e-3) if kept.shape[1] >= 5 else np.ones(len(kept))
            r = image_records(kept[:, :4], score, gt, H_o, W_o)
        recs.append({'name': item['name'], 'pred': r['pred'].tolist(),
                     'score': r['score'].tolist(), 'gt': r['gt'].tolist()})

    def asrec(r):
        return {'pred': np.array(r['pred'], np.float32).reshape(-1, 4) if r['pred'] else np.zeros((0, 4), np.float32),
                'score': np.array(r['score'], np.float32),
                'gt': np.array(r['gt'], np.float32).reshape(-1, 4) if r['gt'] else np.zeros((0, 4), np.float32)}
    sap = sap_dataset([asrec(r) for r in recs], thresholds=(5, 10, 15))
    json.dump({'kind': a.kind, 'fold': a.fold, 'sAP': sap, 'records': recs},
              open(a.out_json, 'w'))
    print(f"{a.kind} fold{a.fold}: dataset sAP {{5:{sap[5]:.4f}, 10:{sap[10]:.4f}, 15:{sap[15]:.4f}}}")


if __name__ == '__main__':
    main()

"""Frozen-model external test on the Wireframe TEST set.

Reviewer #2 request: at least one frozen-model external test on a public
line-segment dataset. We take models DEVELOPED ON YorkUrban (CV fold-0
canonical backbones), FREEZE them (no training on Wireframe whatsoever),
and evaluate on the Wireframe test set under the project's STANDARD
dataset-level LCNN sAP (metrics.image_records + metrics.sap_dataset).

Four configs (all YorkUrban-only, frozen at eval):
  1. afm_bb     - AFM backbone (YorkUrban fold-0), native ranking 1/(quality+1e-3)
  2. lrsaf_bb   - LR-SAF backbone (YorkUrban fold-0), native ranking
  3. afm_head   - AFM backbone + 9-d geom head (head trained ONLY on York fold-0 train)
  4. lrsaf_head - LR-SAF backbone + 9-d geom head (head trained ONLY on York fold-0 train)

Scoring is byte-for-byte the project standard: predictions are scaled to the
ORIGINAL image frame, image_records() normalizes to the 128 frame internally,
sap_dataset() pools globally with one-to-one greedy matching on summed squared
endpoint distance, thresholds (5,10,15). F-measure is computed exactly as
main_table_sap.py's fmeas() (DS=4 downscaled raster F).

NO Wireframe data is used in head training. Heads are loaded frozen.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

import torch
import numpy as np
import cv2

from data_hawp import HAWPJsonDataset
from confidence_head import ConfidenceMLP, compute_segment_features
from metrics import f_measure, image_records, sap_dataset
from lib.squeeze_to_lsg import lsgenerator

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THS = (5, 10, 15)

WF_ANN = ROOT + '/code/hawp_baseline/data/wireframe/test.json'
WF_IMG = ROOT + '/code/hawp_baseline/data/wireframe/images'

CK = ROOT + '/checkpoints/revision/cv'
AFM_BB   = CK + '/afm_fold0.pth'
LRSAF_BB = CK + '/lrsaf_fold0.pth'
AFM_HEAD   = ROOT + '/checkpoints/revision/heads/afm_head_fold0.pth'
LRSAF_HEAD = ROOT + '/checkpoints/revision/lr_saf_conf_head_seed42.pth'


def fmeas(pred, gt, H, W):
    """Identical to main_table_sap.py fmeas(): DS=4 downscaled raster F."""
    DS = 4
    Hd = max(H // DS, 32); Wd = max(W // DS, 32)
    kp = pred.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
    gp = gt.copy();   gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
    return f_measure(kp, gp, Hd, Wd)['F']


def load_afm():
    _cwd = os.getcwd(); os.chdir(ROOT + '/code/afm_baseline')
    from config import cfg
    cfg.merge_from_file('experiments/afm_atrous.yaml')
    from modeling.net import build_network
    os.chdir(_cwd)
    m = build_network(cfg).cuda().eval()
    c = torch.load(AFM_BB, map_location='cuda', weights_only=False)
    m.load_state_dict(c['model'] if isinstance(c, dict) and 'model' in c else c, strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_lrsaf():
    from model import build_lr_saf
    m = build_lr_saf(device='cuda').eval()
    c = torch.load(LRSAF_BB, map_location='cuda', weights_only=False)
    m.load_state_dict(c['model'] if isinstance(c, dict) and 'model' in c else c, strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def afm_infer(m, img):
    """AFM offset map; junc stubbed as zeros (matches train_conf_on_afm.py)."""
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = m(xt)
    afm = out[0] if isinstance(out, (list, tuple)) else out
    offset = afm[0].cpu().numpy().astype(np.float32)
    junc = np.zeros((1, 320, 320), dtype=np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), offset, junc


def lrsaf_infer(m, img):
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = m(xt)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    return np.asarray(lines), offset, junc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_json', default=ROOT + '/logs/wireframe_external_frozen.json')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()

    ds = HAWPJsonDataset(ann_file=WF_ANN, img_dir=WF_IMG, in_res=320, limit=a.limit)
    n = len(ds)
    print(f'Wireframe test: {n} images  (ann={WF_ANN})')

    # Precompute per-image GT in ORIGINAL frame (image_records normalizes to 128).
    # HAWPJsonDataset returns lines in the 320 frame; undo to original exactly
    # as main_table_sap.py does (gt[:,0::2]*=W_o/320).
    meta = []
    for i in range(n):
        it = ds[i]
        H_o, W_o = it['H_orig'], it['W_orig']
        gt = it['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0
        gt[:, 1::2] *= H_o / 320.0
        meta.append({'name': it['name'], 'filename': ds.items[i]['filename'],
                     'gt': gt, 'H': H_o, 'W': W_o})

    head_afm = ConfidenceMLP(in_dim=9, hidden=64).cuda().eval()
    head_afm.load_state_dict(torch.load(AFM_HEAD, weights_only=False)['head'])
    head_lr = ConfidenceMLP(in_dim=9, hidden=64).cuda().eval()
    head_lr.load_state_dict(torch.load(LRSAF_HEAD, weights_only=False)['head'])

    res = {}

    def run_backbone(kind, infer_fn, head):
        recs_bb, fs_bb = [], []
        recs_hd, fs_hd = [], []
        for x in meta:
            img = cv2.imread(os.path.join(WF_IMG, x['filename']))
            if img is None:
                raise FileNotFoundError(os.path.join(WF_IMG, x['filename']))
            lines, offset, junc = infer_fn(img)
            if len(lines) == 0:
                empty = image_records(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                                      x['gt'], x['H'], x['W'])
                recs_bb.append(empty); fs_bb.append(0.0)
                recs_hd.append(empty); fs_hd.append(0.0)
                continue
            kept = lines[:, :5].copy()
            kept[:, 0::2] *= x['W'] / 320.0
            kept[:, 1::2] *= x['H'] / 320.0
            # --- backbone-only native ranking (matches dump_backbone_records.py) ---
            sc_bb = 1.0 / (kept[:, 4] + 1e-3) if kept.shape[1] >= 5 else np.ones(len(kept))
            recs_bb.append(image_records(kept[:, :4], sc_bb, x['gt'], x['H'], x['W']))
            fs_bb.append(fmeas(kept[:, :4], x['gt'], x['H'], x['W']))
            # --- head ranking (features in 320 space, matches training) ---
            feats = compute_segment_features(lines[:, :5].copy(), offset, junc, H=320, W=320).cuda()
            with torch.no_grad():
                sc_hd = head(feats).detach().cpu().numpy().reshape(-1)
            recs_hd.append(image_records(kept[:, :4], sc_hd, x['gt'], x['H'], x['W']))
            fs_hd.append(fmeas(kept[:, :4], x['gt'], x['H'], x['W']))
        ap_bb = sap_dataset(recs_bb, THS)
        ap_hd = sap_dataset(recs_hd, THS)
        res[f'{kind}_bb'] = {'F': float(np.mean(fs_bb)),
                             'sAP5': ap_bb[5], 'sAP10': ap_bb[10], 'sAP15': ap_bb[15],
                             'n_images': len(recs_bb)}
        res[f'{kind}_head'] = {'F': float(np.mean(fs_hd)),
                               'sAP5': ap_hd[5], 'sAP10': ap_hd[10], 'sAP15': ap_hd[15],
                               'n_images': len(recs_hd)}
        print(f'{kind}_bb  :', res[f'{kind}_bb'])
        print(f'{kind}_head:', res[f'{kind}_head'])

    print('--- AFM backbone + AFM head ---')
    afm = load_afm()
    run_backbone('afm', lambda im: afm_infer(afm, im), head_afm)
    del afm; torch.cuda.empty_cache()

    print('--- LR-SAF backbone + LR-SAF head ---')
    lr = load_lrsaf()
    run_backbone('lrsaf', lambda im: lrsaf_infer(lr, im), head_lr)
    del lr; torch.cuda.empty_cache()

    out = {
        'dataset': 'Wireframe test (frozen YorkUrban models, cross-dataset)',
        'n_images': n,
        'ann_file': WF_ANN,
        'img_dir': WF_IMG,
        'checkpoints': {'afm_bb': AFM_BB, 'lrsaf_bb': LRSAF_BB,
                        'afm_head': AFM_HEAD, 'lrsaf_head': LRSAF_HEAD},
        'metric': 'standard dataset-level LCNN sAP (image_records + sap_dataset), thresholds (5,10,15)',
        'results': res,
        'yorkurban_indomain_cv_reference': {'afm_bb': 0.043, 'lrsaf_bb': 0.010,
                                            'afm_head': 0.122, 'lrsaf_head': 0.039},
    }
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(out, open(a.out_json, 'w'), indent=2)
    print('SAVED', a.out_json)
    # ordering check
    r = res
    print('\nORDERING CHECK (out-of-domain Wireframe):')
    print('  afm_head sAP10 >= lrsaf_head sAP10 ?',
          r['afm_head']['sAP10'] >= r['lrsaf_head']['sAP10'],
          f"({r['afm_head']['sAP10']:.4f} vs {r['lrsaf_head']['sAP10']:.4f})")
    print('  backbones low ? afm_bb sAP10=%.4f  lrsaf_bb sAP10=%.4f'
          % (r['afm_bb']['sAP10'], r['lrsaf_bb']['sAP10']))


if __name__ == '__main__':
    main()

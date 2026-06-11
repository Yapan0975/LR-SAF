"""
Train the confidence head on top of a frozen LR-SAF model.

For each training image:
  1. Run frozen LR-SAF -> AFM prediction
  2. Run squeeze -> candidate line segments
  3. Match each candidate to GT -> binary label
  4. Compute features, predict score, BCE loss
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from data import YorkUrbanSubset
from model import build_lr_saf
from confidence_head import ConfidenceMLP, compute_segment_features, match_to_gt
from metrics import s_ap, f_measure, image_records, sap_dataset
from lib.squeeze_to_lsg import lsgenerator


def run_squeeze(model, img, in_res=320):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (in_res, in_res)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    offset = out['a'][0].cpu().numpy().astype(np.float32)
    junc = out['junc'][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    lines = np.asarray(lines)
    return lines, offset, junc, (H_o, W_o)


def main(epochs=10, lr=1e-3, seed=42, ckpt=None, out_json=None, fold=-1, nfolds=5):
    torch.manual_seed(seed); np.random.seed(seed)

    ds = YorkUrbanSubset(in_res=320)
    if fold is not None and fold >= 0:
        # K-fold CV: held-out test = fold; head trained on the rest. Matches the
        # backbone's CV partition (cv-seed 0) so test images are unseen by both.
        cvperm = np.random.RandomState(0).permutation(len(ds))
        parts = np.array_split(cvperm, nfolds)
        val_idx = parts[fold].tolist()
        train_idx = np.concatenate([parts[j] for j in range(nfolds) if j != fold]).tolist()
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(ds))
        train_idx, val_idx = perm[:80].tolist(), perm[80:].tolist()
    items_train = [ds[i] for i in train_idx]
    items_val = [ds[i] for i in val_idx]

    # Frozen LR-SAF main model (use per-seed backbone if provided)
    main_model = build_lr_saf(device='cuda').eval()
    main_ckpt = torch.load(ckpt or (ROOT + '/checkpoints/lr_saf_best.pth'),
                            map_location='cuda', weights_only=False)
    main_model.load_state_dict(main_ckpt['model'], strict=True)
    for p in main_model.parameters():
        p.requires_grad_(False)
    print(f"loaded main model: epoch={main_ckpt['epoch']}, val_F={main_ckpt['val_F']:.4f}")

    # Confidence head
    head = ConfidenceMLP(in_dim=9, hidden=64).cuda()
    optim = torch.optim.Adam(head.parameters(), lr=lr)

    print("\n=== precomputing candidate segments + features for train set ===")
    train_data = []
    for item in items_train:
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        lines, offset, junc, (H_o, W_o) = run_squeeze(main_model, img)
        if len(lines) == 0:
            continue
        sx, sy = W_o / 320.0, H_o / 320.0
        kept = lines[:, :5].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0

        feats = compute_segment_features(lines[:, :5].copy(), offset, junc, H=320, W=320)  # BUGFIX: features in 320-space (lines), not original-coord kept
        # Note: features computed in 320-coord space; pass kept (scaled) for matching
        target = match_to_gt(torch.from_numpy(kept[:, :4]).float(),
                              torch.from_numpy(gt).float(), H=H_o, W=W_o)
        train_data.append({'feats': feats.cuda(), 'tgt': target.cuda(),
                            'name': item['name']})
    n_pos = sum(t['tgt'].sum().item() for t in train_data)
    n_total = sum(len(t['tgt']) for t in train_data)
    print(f"train: {len(train_data)} images, {n_total} candidates, "
          f"{int(n_pos)} positives ({100*n_pos/max(n_total,1):.1f}%)")

    print("\n=== precomputing for val set ===")
    val_data = []
    for item in items_val:
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        lines, offset, junc, (H_o, W_o) = run_squeeze(main_model, img)
        if len(lines) == 0:
            continue
        sx, sy = W_o / 320.0, H_o / 320.0
        kept = lines[:, :5].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0
        feats = compute_segment_features(lines[:, :5].copy(), offset, junc, H=320, W=320)  # BUGFIX: features in 320-space (lines), not original-coord kept
        target = match_to_gt(torch.from_numpy(kept[:, :4]).float(),
                              torch.from_numpy(gt).float(), H=H_o, W=W_o)
        val_data.append({'feats': feats.cuda(), 'tgt': target.cuda(),
                         'kept': kept, 'gt': gt,
                         'H_o': H_o, 'W_o': W_o, 'name': item['name']})

    print("\n=== training confidence head ===")
    # Pos weight to balance
    pos_w = max(1.0, (n_total - n_pos) / max(n_pos, 1))
    print(f"pos_weight: {pos_w:.2f}")

    for ep in range(epochs):
        head.train()
        ep_L = []
        order = np.random.permutation(len(train_data))
        for idx in order:
            d = train_data[idx]
            scores = head(d['feats']).clamp(1e-6, 1-1e-6)
            tgt = d['tgt']
            # Weighted BCE
            w = tgt * pos_w + (1 - tgt)
            loss = (- w * (tgt * torch.log(scores) + (1 - tgt) * torch.log(1 - scores))).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            ep_L.append(loss.item())

        # Eval: re-rank predictions by learned score; DATASET-LEVEL sAP
        # (pool all images, one global PR curve) -- the standard LCNN protocol.
        head.eval()
        with torch.no_grad():
            recs = []
            for d in val_data:
                sc = head(d['feats']).cpu().numpy().reshape(-1)
                r = image_records(d['kept'][:, :4], sc, d['gt'], d['H_o'], d['W_o'])
                r['name'] = d['name']
                recs.append(r)
            aps = sap_dataset(recs, thresholds=(5, 10, 15))
            ap5_m, ap10_m = aps[5], aps[10]
        print(f"  ep{ep:02d}: L={np.mean(ep_L):.4f} | val(dataset) sAP5={ap5_m:.4f} sAP10={ap10_m:.4f}")

    # final-epoch records for dataset-level scoring + image-resample bootstrap
    if ckpt is None:
        save = ROOT + '/checkpoints/lr_saf_conf_head.pth'
    else:
        save = ROOT + f'/checkpoints/revision/lr_saf_conf_head_seed{seed}.pth'
        os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save({'head': head.state_dict()}, save)
    print(f"saved {save}; final dataset sAP-10={ap10_m:.4f}")

    if out_json:
        rec = {'seed': seed, 'ckpt': ckpt or 'lr_saf_best.pth',
               'final_sAP10': ap10_m, 'final_sAP5': ap5_m,
               'records': [{'name': r['name'],
                            'pred': r['pred'].tolist(),
                            'score': r['score'].tolist(),
                            'gt': r['gt'].tolist()} for r in recs]}
        with open(out_json, 'w') as fh:
            json.dump(rec, fh)
        print(f"saved {out_json}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--out_json', type=str, default=None)
    ap.add_argument('--fold', type=int, default=-1)
    ap.add_argument('--nfolds', type=int, default=5)
    args = ap.parse_args()
    main(epochs=args.epochs, seed=args.seed, ckpt=args.ckpt, out_json=args.out_json,
         fold=args.fold, nfolds=args.nfolds)

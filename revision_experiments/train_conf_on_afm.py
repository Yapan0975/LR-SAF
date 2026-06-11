"""Train the same 9-d geometric confidence head on top of a frozen
AFM-finetuned backbone (the matched-protocol AFM of Table XIII).

Replicates train_confidence.py's protocol but swaps the backbone for
AFM, so the conf-head's contribution can be isolated from the
LR-SAF backbone.

Output: best val sAP-10 on YorkUrban-22.
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
from confidence_head import ConfidenceMLP, compute_segment_features, match_to_gt
from metrics import s_ap, f_measure, image_records, sap_dataset
from lib.squeeze_to_lsg import lsgenerator

# AFM backbone (chdir to afm_baseline so relative cfg path resolves)
_orig_cwd = os.getcwd()
os.chdir(ROOT + '/code/afm_baseline')
from config import cfg
cfg.merge_from_file('experiments/afm_atrous.yaml')
from modeling.net import build_network
os.chdir(_orig_cwd)


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_squeeze_afm(model, img, in_res=320):
    """Adapter: AFM network output -> (lines, offset, junc, (H,W))."""
    H_o, W_o = img.shape[:2]
    x = cv2.resize(img, (in_res, in_res)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    # AFM model output: tuple/list, first element is the offset (B, 2, H, W)
    afm_out = out[0] if isinstance(out, (list, tuple)) else out
    offset = afm_out[0].cpu().numpy().astype(np.float32)
    # AFM has no junction head; stub a zero map for compute_segment_features
    junc = np.zeros((1, in_res, in_res), dtype=np.float32)
    lines, _, _ = lsgenerator(offset)
    lines = np.asarray(lines)
    return lines, offset, junc, (H_o, W_o)


def main(epochs=15, lr=1e-3, seed=42, afm_ckpt=None, out_head=None, out_json=None,
         fold=-1, nfolds=5):
    torch.manual_seed(seed); np.random.seed(seed)

    ds = YorkUrbanSubset(in_res=320)
    if fold is not None and fold >= 0:
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

    # Frozen AFM backbone
    main_model = build_network(cfg).cuda().eval()
    ckpt = torch.load(afm_ckpt, map_location='cuda', weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    main_model.load_state_dict(state, strict=True)
    for p in main_model.parameters():
        p.requires_grad_(False)
    print(f"loaded AFM-finetuned backbone from {afm_ckpt}")

    head = ConfidenceMLP(in_dim=9, hidden=64).cuda()
    optim = torch.optim.Adam(head.parameters(), lr=lr)

    print("\n=== train precompute ===")
    train_data = []
    for item in items_train:
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        lines, offset, junc, (H_o, W_o) = run_squeeze_afm(main_model, img)
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
        train_data.append({'feats': feats.cuda(), 'tgt': target.cuda()})
    n_pos = sum(t['tgt'].sum().item() for t in train_data)
    n_total = sum(len(t['tgt']) for t in train_data)
    print(f"train: {len(train_data)} imgs, {n_total} cands, {int(n_pos)} pos "
          f"({100*n_pos/max(n_total,1):.1f}%)")
    pos_w = max(1.0, (n_total - n_pos) / max(n_pos, 1))   # match train_confidence.py weighting

    print("\n=== val precompute ===")
    val_data = []
    for item in items_val:
        img = cv2.imread(os.path.join(ds.root, item['name'], f"{item['name']}.jpg"))
        lines, offset, junc, (H_o, W_o) = run_squeeze_afm(main_model, img)
        if len(lines) == 0:
            val_data.append(None); continue
        sx, sy = W_o / 320.0, H_o / 320.0
        kept = lines[:, :5].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / 320.0; gt[:, 1::2] *= H_o / 320.0
        feats = compute_segment_features(lines[:, :5].copy(), offset, junc, H=320, W=320)  # BUGFIX: features in 320-space (lines), not original-coord kept
        val_data.append({'feats': feats.cuda(), 'kept': kept, 'gt': gt,
                         'H_orig': H_o, 'W_orig': W_o, 'name': item['name']})

    def eval_val():
        # DATASET-LEVEL sAP (standard LCNN protocol): pool all images.
        head.eval()
        recs = []
        for v in val_data:
            if v is None:
                continue
            with torch.no_grad():
                sc = head(v['feats']).cpu().numpy().reshape(-1)
            r = image_records(v['kept'][:, :4], sc, v['gt'], v['H_orig'], v['W_orig'])
            r['name'] = v['name']
            recs.append(r)
        head.train()
        return sap_dataset(recs, thresholds=(10,))[10], recs

    print(f"\n=== training, {epochs} epochs ===")
    log = {'train_loss': [], 'val_sAP10': []}
    best_sap, final_sap = -1.0, float('nan')
    for ep in range(epochs):
        ep_losses = []
        for batch in train_data:
            optim.zero_grad()
            # BUGFIX: ConfidenceMLP.forward already applies sigmoid, so use a
            # weighted BCE on the probability (identical to train_confidence.py).
            # The previous binary_cross_entropy_with_logits applied a SECOND
            # sigmoid, clamping outputs to (0.5, 0.73) and crippling training.
            scores = head(batch['feats']).squeeze(-1).clamp(1e-6, 1 - 1e-6)
            tgt = batch['tgt']
            w = tgt * pos_w + (1 - tgt)
            loss = (-w * (tgt * torch.log(scores)
                          + (1 - tgt) * torch.log(1 - scores))).mean()
            loss.backward(); optim.step()
            ep_losses.append(loss.item())
        L = float(np.mean(ep_losses)) if ep_losses else float('nan')
        sap10, recs = eval_val()
        log['train_loss'].append(L); log['val_sAP10'].append(sap10)
        print(f"ep {ep:02d}: loss={L:.4f}  val(dataset) sAP-10={sap10:.4f}")
        final_sap = sap10
        best_sap = max(best_sap, sap10)
    # BUGFIX: report the FINAL-epoch sAP-10 (no test-set epoch selection),
    # matching train_confidence.py so the AFM-head and LR-SAF-head arms are
    # comparable. best_sAP10 is kept in the json for reference only.
    if out_head:
        os.makedirs(os.path.dirname(out_head) or '.', exist_ok=True)
        torch.save({'head': head.state_dict(), 'ep': epochs - 1,
                    'val_sAP10': final_sap}, out_head)
    print(f"\nfinal val(dataset) sAP-10 (AFM+geom-head): {final_sap:.4f}")
    if out_json:
        with open(out_json, 'w') as fh:
            json.dump({'seed': seed, 'final_sAP10': final_sap,
                       'records': [{'name': r['name'],
                                    'pred': r['pred'].tolist(),
                                    'score': r['score'].tolist(),
                                    'gt': r['gt'].tolist()} for r in recs],
                       'afm_ckpt': afm_ckpt}, fh)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--afm_ckpt', type=str, required=True)
    ap.add_argument('--out_head', type=str, default=None)
    ap.add_argument('--out_json', type=str, default=None)
    ap.add_argument('--fold', type=int, default=-1)
    ap.add_argument('--nfolds', type=int, default=5)
    args = ap.parse_args()
    main(epochs=args.epochs, seed=args.seed,
         afm_ckpt=args.afm_ckpt, out_head=args.out_head, out_json=args.out_json,
         fold=args.fold, nfolds=args.nfolds)

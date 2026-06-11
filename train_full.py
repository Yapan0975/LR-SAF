"""
LR-SAF full training on YorkUrban (80 train / 22 val split, deterministic).

This is a methodological validation: starting from AFM pretrained, fine-tune
with LR-SAF losses (SAF + TNNR + endpoint + junction) and compare against
the AFM baseline on a held-out val set.

For production training on Wireframe, manually download pointlines.zip from
https://github.com/huangkuns/wireframe and adjust the data loader.

Run: python3 train_full.py --epochs 20
"""
import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

ROOT = '/home/server/Documents/yping/LR-SAF-LSD'
sys.path.insert(0, ROOT + '/code/lr_saf')
sys.path.insert(0, ROOT + '/code/afm_baseline')
sys.path.insert(0, ROOT + '/code/afm_baseline/lib')

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from saf_target import compute_saf_target
from tnn_loss import tnn_loss
from bounded import encode_bounded, decode_bounded
from data import YorkUrbanSubset, collate_variable_lines
from model import build_lr_saf
from metrics import f_measure, s_ap


def make_targets(lines_batch, mask_batch, H, W, device, D_max=80.0, sigma=2.0, K=3,
                 encoding='afm'):
    """encoding: 'afm' for log encoding compatible with squeeze module;
                  'tanh' for our bounded encoding (paper Eq 4.2, NOT squeeze-compatible)."""
    B = lines_batch.shape[0]
    target = {'a': torch.zeros(B, 2, H, W, device=device),
              't_star': torch.zeros(B, 1, H, W, device=device),
              'junc':   torch.zeros(B, 1, H, W, device=device),
              'support': torch.zeros(B, H, W, device=device)}
    for b in range(B):
        n = int(mask_batch[b].sum().item())
        if n == 0:
            continue
        out = compute_saf_target(lines_batch[b, :n].to(device), H, W,
                                 sigma=sigma, K=K, D_max=D_max,
                                 bounded=encoding, device=device)
        target['a'][b, 0] = out['a_x']
        target['a'][b, 1] = out['a_y']
        target['t_star'][b, 0] = out['t_star']
        target['junc'][b, 0] = out['junc']
        target['support'][b] = out['support']
    return target


def weighted_l1(pred, gt, support, w=10.0):
    diff = (pred - gt).abs().sum(dim=1)
    weight = support * w + (1 - support) * 1.0
    return (weight * diff).mean()


@torch.no_grad()
def evaluate(model, val_items, D_max=80.0, in_res=320):
    """Run on val set, decode predictions, compute F + sAP."""
    model.eval()
    from lib.squeeze_to_lsg import lsgenerator
    f_list, ap10_list = [], []
    for item in val_items:
        name = item['name']
        img = cv2.imread(os.path.join(item['root'], name, f"{name}.jpg"))
        H_o, W_o = img.shape[:2]
        x = cv2.resize(img, (in_res, in_res)).astype(np.float32) / 255.0
        x[..., 0] = (x[..., 0] - 0.485) / 0.229
        x[..., 1] = (x[..., 1] - 0.456) / 0.224
        x[..., 2] = (x[..., 2] - 0.406) / 0.225
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()

        out = model(x)
        # Network predicts AFM-encoded values directly (squeeze-compatible)
        afm_enc = out['a']
        offset = afm_enc[0].cpu().numpy().astype(np.float32)
        lines_pred, _, _ = lsgenerator(offset)
        lines_pred = np.asarray(lines_pred)
        if len(lines_pred) == 0:
            f_list.append(0.0); ap10_list.append(0.0); continue
        sx, sy = W_o / in_res, H_o / in_res
        kept = lines_pred[:, :4].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        scores = 1.0 / (lines_pred[:, 4] + 1e-3) if lines_pred.shape[1] >= 5 \
                 else np.ones(len(kept))

        gt = item['gt_orig']
        # F at downsampled res
        DS = 4
        H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
        kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
        gp = gt.copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
        fm = f_measure(kp, gp, H_ds, W_ds)
        aps = s_ap(kept, scores, gt, H_o, W_o, thresholds=(10,))
        f_list.append(fm['F'])
        ap10_list.append(aps[10])
    model.train()
    return float(np.mean(f_list)), float(np.mean(ap10_list))


def main(epochs=20, batch=2, lr=5e-5, save_every=5,
         lam_rec=1.0, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5,
         tnn_warmup_epochs=3, seed=42, out_ckpt=None, out_log=None,
         K=3, encoding='afm', regularizer='tnnr', lam_tv=0.0,
         fold=-1, nfolds=5):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda'

    full_ds = YorkUrbanSubset(in_res=320, limit=None)
    print(f"YorkUrban total: {len(full_ds)} images")
    if fold is not None and fold >= 0:
        # K-fold CV: held-out test = fold (never used for training/selection),
        # train = the other folds. Fixed cv-seed=0 partition (independent of run seed).
        cvperm = np.random.RandomState(0).permutation(len(full_ds))
        parts = np.array_split(cvperm, nfolds)
        val_idx = parts[fold].tolist()
        train_idx = np.concatenate([parts[j] for j in range(nfolds) if j != fold]).tolist()
        print(f"[CV] fold {fold}/{nfolds}: test={len(val_idx)} train={len(train_idx)}")
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(full_ds))
        n_train = 80
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train:].tolist()
    print(f"train: {len(train_idx)}, val/test: {len(val_idx)}")

    train_set = torch.utils.data.Subset(full_ds, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True,
        collate_fn=collate_variable_lines, num_workers=2)

    # Prepare val items (img path + GT in original coords)
    val_items = []
    for vi in val_idx:
        item = full_ds[vi]
        H_o, W_o = item['H_orig'], item['W_orig']
        gt = item['lines'].numpy()    # in 320x320
        gt_orig = gt.copy()
        gt_orig[:, 0::2] *= W_o / 320.0
        gt_orig[:, 1::2] *= H_o / 320.0
        val_items.append({'name': item['name'], 'root': full_ds.root,
                          'gt_orig': gt_orig, 'H_orig': H_o, 'W_orig': W_o})

    # Model
    model = build_lr_saf(device=device)
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    D_MAX = 80.0

    log = {'train_loss': [], 'val_F': [], 'val_sAP10': [], 'epochs': []}
    print(f"\n=== LR-SAF fine-tuning, {epochs} epochs ===")
    best_F, best_path = -1.0, None
    for ep in range(epochs):
        ep_t = time.time()
        model.train()
        # TNNR warm-up
        eff_lam_tnnr = 0.0 if ep < tnn_warmup_epochs else \
                      lam_tnnr * min((ep - tnn_warmup_epochs + 1) / 3.0, 1.0)
        ep_losses = []
        for it, batch_d in enumerate(train_loader):
            optim.zero_grad()
            imgs = batch_d['image'].to(device)
            B, _, H, W = imgs.shape
            with torch.no_grad():
                tgt = make_targets(batch_d['lines'], batch_d['lines_mask'],
                                   H, W, device, D_max=D_MAX, K=K, encoding=encoding)
            out = model(imgs)
            # Network output is in AFM log-encoded space; compare directly to target
            a_enc = out['a']
            t_pred, junc_pred = out['t_star'], out['junc']

            L_rec = weighted_l1(a_enc, tgt['a'], tgt['support'])
            L_junc = F.binary_cross_entropy(junc_pred.clamp(1e-6, 1-1e-6),
                                            tgt['junc'])
            L_t = (((t_pred.squeeze(1) - tgt['t_star'].squeeze(1)).abs()
                    * tgt['support']).sum() /
                    tgt['support'].sum().clamp(min=1))
            if regularizer == 'tnnr':
                L_reg = tnn_loss(a_enc, junc_pred=junc_pred,
                                 window=9, stride=8, K_max=3, normalize=True)
                lam_reg = eff_lam_tnnr
            elif regularizer == 'tv':
                # Total-Variation regularizer on predicted AFM channels.
                # |a[i+1,j] - a[i,j]| + |a[i,j+1] - a[i,j]|, support-masked.
                dx = (a_enc[..., 1:, :] - a_enc[..., :-1, :]).abs().sum(dim=1)
                dy = (a_enc[..., :, 1:] - a_enc[..., :, :-1]).abs().sum(dim=1)
                L_reg = dx.mean() + dy.mean()
                lam_reg = eff_lam_tnnr  # reuse the same scheduled weight
            else:  # 'none'
                L_reg = torch.tensor(0.0, device=device)
                lam_reg = 0.0
            L = (lam_rec * L_rec + lam_reg * L_reg
                 + lam_junc * L_junc + lam_t * L_t)

            if not torch.isfinite(L):
                print(f"  ep{ep} it{it}: NaN loss, skip"); continue
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            ep_losses.append(L.item())

        sched.step()
        mean_L = float(np.mean(ep_losses)) if ep_losses else float('nan')

        # Eval every epoch
        val_F, val_sAP10 = evaluate(model, val_items, D_max=D_MAX)
        log['train_loss'].append(mean_L)
        log['val_F'].append(val_F)
        log['val_sAP10'].append(val_sAP10)
        log['epochs'].append(ep)
        elapsed = time.time() - ep_t
        print(f"ep {ep:02d}: train_L={mean_L:.4f} | val F={val_F:.4f} sAP10={val_sAP10:.4f} "
              f"| lam_tnnr_eff={eff_lam_tnnr:.4f} | {elapsed:.1f}s")

        # Save best (legacy 80/22 protocol only; never select on a CV test fold)
        if (fold is None or fold < 0) and val_F > best_F:
            best_F = val_F
            best_path = out_ckpt or (ROOT + '/checkpoints/lr_saf_best.pth')
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save({'model': model.state_dict(), 'epoch': ep,
                        'val_F': val_F, 'val_sAP10': val_sAP10}, best_path)

    if fold is not None and fold >= 0:
        # CV: save and report the FINAL-epoch model (no test-fold selection)
        best_path = out_ckpt or (ROOT + '/checkpoints/lr_saf_best.pth')
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        torch.save({'model': model.state_dict(), 'epoch': ep,
                    'val_F': val_F, 'val_sAP10': val_sAP10}, best_path)
        log['final_F'] = val_F; log['final_sAP10'] = val_sAP10

    # Save log
    log_path = out_log or (ROOT + '/logs/lr_saf_train_log.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nlogs -> {log_path}")
    print(f"best ckpt -> {best_path}  (F={best_F:.4f})")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"AFM baseline (paper):       F=0.646")
    print(f"AFM baseline (our YorkUrban full eval): F=0.7351, sAP10=0.1608")
    print(f"LR-SAF on val (80/22):     best F={best_F:.4f}, final sAP10={log['val_sAP10'][-1]:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lam_tnnr', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=None,
                    help='Override best-checkpoint path (revision experiments).')
    ap.add_argument('--log', type=str, default=None,
                    help='Override training-log path (revision experiments).')
    ap.add_argument('--K', type=int, default=3,
                    help='Soft top-K assignment (1 = hard partition).')
    ap.add_argument('--encoding', type=str, default='afm',
                    choices=['afm', 'tanh'],
                    help='AFM target encoding (afm=log, tanh=bounded).')
    ap.add_argument('--reg', type=str, default='tnnr',
                    choices=['tnnr', 'tv', 'none'],
                    help='Regularizer applied to AFM windows.')
    ap.add_argument('--fold', type=int, default=-1,
                    help='K-fold CV: held-out test fold index (>=0 enables CV).')
    ap.add_argument('--nfolds', type=int, default=5)
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr,
         lam_tnnr=args.lam_tnnr, seed=args.seed, fold=args.fold, nfolds=args.nfolds,
         out_ckpt=args.out, out_log=args.log,
         K=args.K, encoding=args.encoding, regularizer=args.reg)

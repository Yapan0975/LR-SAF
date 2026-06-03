"""
Full LR-SAF training on the Wireframe dataset (HAWP-format JSON).

Trains from AFM pretrained, evaluates on Wireframe test (462) AND YorkUrban
test (102, cross-dataset eval per AFM/HAWP convention).

Run: python3 train_wireframe.py --epochs 50 --batch 4
"""
import os, sys, time, json, argparse, warnings
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
from data_hawp import (wireframe_train, wireframe_test, york_test,
                        collate_variable_lines)
from model import build_lr_saf
from metrics import f_measure, s_ap
from lib.squeeze_to_lsg import lsgenerator

DATA_ROOT = ROOT + '/code/hawp_baseline/data'

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_targets(lines_batch, mask_batch, H, W, device, K=3, sigma=2.0):
    B = lines_batch.shape[0]
    tgt = {'a': torch.zeros(B, 2, H, W, device=device),
           't_star': torch.zeros(B, 1, H, W, device=device),
           'junc': torch.zeros(B, 1, H, W, device=device),
           'support': torch.zeros(B, H, W, device=device)}
    for b in range(B):
        n = int(mask_batch[b].sum().item())
        if n == 0:
            continue
        o = compute_saf_target(lines_batch[b, :n].to(device), H, W,
                               sigma=sigma, K=K, bounded='afm', device=device)
        tgt['a'][b, 0] = o['a_x']; tgt['a'][b, 1] = o['a_y']
        tgt['t_star'][b, 0] = o['t_star']
        tgt['junc'][b, 0] = o['junc']
        tgt['support'][b] = o['support']
    return tgt


def weighted_l1(pred, gt, support, w=10.0):
    diff = (pred - gt).abs().sum(dim=1)
    weight = support * w + (1 - support) * 1.0
    return (weight * diff).mean()


@torch.no_grad()
def evaluate_dataset(model, ds, name='val', limit=None):
    model.eval()
    f_list, ap10_list = [], []
    if limit is not None:
        indices = list(range(min(limit, len(ds))))
    else:
        indices = list(range(len(ds)))

    for idx in indices:
        item = ds[idx]
        H_o, W_o = item['H_orig'], item['W_orig']
        # Recover original image to feed network at in_res
        x = item['image'].unsqueeze(0).cuda()
        out = model(x)
        offset = out['a'][0].cpu().numpy().astype(np.float32)
        lines, _, _ = lsgenerator(offset)
        lines = np.asarray(lines)
        if len(lines) == 0:
            f_list.append(0.0); ap10_list.append(0.0); continue
        sx, sy = W_o / ds.in_res, H_o / ds.in_res
        kept = lines[:, :4].copy(); kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        scores = (1.0 / (lines[:, 4] + 1e-3) if lines.shape[1] >= 5
                  else np.ones(len(kept)))
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= W_o / ds.in_res
        gt[:, 1::2] *= H_o / ds.in_res
        # Quick eval at downsampled res
        DS = 4
        H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
        kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
        gp = gt.copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
        fm = f_measure(kp, gp, H_ds, W_ds)
        aps = s_ap(kept, scores, gt, H_o, W_o, thresholds=(10,))
        f_list.append(fm['F']); ap10_list.append(aps[10])
    model.train()
    return float(np.mean(f_list)), float(np.mean(ap10_list))


def main(epochs=50, batch=4, lr=5e-5,
         lam_rec=1.0, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5,
         tnn_warmup_epochs=3, eval_every=5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda'

    train_ds = wireframe_train(DATA_ROOT, in_res=320, augment=True)
    wf_test = wireframe_test(DATA_ROOT, in_res=320)
    york_t = york_test(DATA_ROOT, in_res=320)
    print(f"Wireframe train: {len(train_ds)}, test: {len(wf_test)}")
    print(f"YorkUrban test:  {len(york_t)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        collate_fn=collate_variable_lines,
        num_workers=4, pin_memory=True)

    model = build_lr_saf(device=device)
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    log = {'epochs': [], 'train_loss': [],
           'wf_F': [], 'wf_sAP10': [],
           'york_F': [], 'york_sAP10': []}
    best = {'wf_F': -1.0, 'epoch': -1}
    best_path = ROOT + '/checkpoints/lr_saf_wireframe_best.pth'

    print(f"\n=== LR-SAF training on Wireframe, {epochs} epochs ===")
    for ep in range(epochs):
        ep_t = time.time()
        eff_lam_tnnr = (0.0 if ep < tnn_warmup_epochs
                         else lam_tnnr * min((ep - tnn_warmup_epochs + 1) / 3.0, 1.0))
        ep_losses = []
        model.train()
        for it, batch_d in enumerate(train_loader):
            optim.zero_grad()
            imgs = batch_d['image'].to(device, non_blocking=True)
            B, _, H, W = imgs.shape
            with torch.no_grad():
                tgt = make_targets(batch_d['lines'], batch_d['lines_mask'],
                                   H, W, device)
            out = model(imgs)
            a_enc = out['a']; t_pred = out['t_star']; junc_pred = out['junc']

            L_rec = weighted_l1(a_enc, tgt['a'], tgt['support'])
            L_junc = F.binary_cross_entropy(junc_pred.clamp(1e-6, 1-1e-6),
                                            tgt['junc'])
            L_t = (((t_pred.squeeze(1) - tgt['t_star'].squeeze(1)).abs()
                    * tgt['support']).sum() / tgt['support'].sum().clamp(min=1))
            L_tnnr = tnn_loss(a_enc, junc_pred=junc_pred,
                              window=9, stride=8, K_max=3, normalize=True)
            L = (lam_rec * L_rec + eff_lam_tnnr * L_tnnr
                 + lam_junc * L_junc + lam_t * L_t)
            if not torch.isfinite(L):
                print(f"  ep{ep} it{it}: NaN, skip"); continue
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            ep_losses.append(L.item())

        sched.step()
        mean_L = float(np.mean(ep_losses)) if ep_losses else float('nan')

        # Evaluate periodically (eval is expensive)
        if (ep + 1) % eval_every == 0 or ep == 0 or ep == epochs - 1:
            wf_F, wf_ap10 = evaluate_dataset(model, wf_test, name='wf_test')
            y_F, y_ap10 = evaluate_dataset(model, york_t, name='york_test')
        else:
            wf_F = wf_ap10 = y_F = y_ap10 = float('nan')

        log['epochs'].append(ep)
        log['train_loss'].append(mean_L)
        log['wf_F'].append(wf_F); log['wf_sAP10'].append(wf_ap10)
        log['york_F'].append(y_F); log['york_sAP10'].append(y_ap10)
        elapsed = time.time() - ep_t
        print(f"ep {ep:02d}: L={mean_L:.4f} | "
              f"WF F={wf_F:.4f} sAP10={wf_ap10:.4f} | "
              f"York F={y_F:.4f} sAP10={y_ap10:.4f} | "
              f"lam_tnnr={eff_lam_tnnr:.4f} | {elapsed:.1f}s")

        if wf_F > best['wf_F'] and wf_F == wf_F:    # not NaN
            best = {'wf_F': wf_F, 'wf_sAP10': wf_ap10,
                    'york_F': y_F, 'york_sAP10': y_ap10, 'epoch': ep}
            torch.save({'model': model.state_dict(), **best}, best_path)

    log_path = ROOT + '/logs/lr_saf_wireframe_log.json'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nlog -> {log_path}")
    print(f"best -> {best_path}  (epoch {best['epoch']}, WF F={best['wf_F']:.4f}, "
          f"WF sAP10={best['wf_sAP10']:.4f}, York sAP10={best['york_sAP10']:.4f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lam_tnnr', type=float, default=0.05)
    ap.add_argument('--eval_every', type=int, default=5)
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr,
         lam_tnnr=args.lam_tnnr, eval_every=args.eval_every)

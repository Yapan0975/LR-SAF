"""
LR-SAF ablation runner.

Trains multiple variants on the same train/val split and reports val F + sAP.
Each variant takes ~6 min (20 epochs * 17 s with eval) on RTX 5090.

Variants (8 total):
  v0  baseline:        SAF (K=3) + TNNR + junction + endpoint, AFM encoding
  v1  no-SAF:          K=1 hard assignment (equivalent to AFM target)
  v2  no-TNNR:          lam_tnnr = 0
  v3  no-junction:     lam_junc = 0, fixed_r=2 (no adaptive)
  v4  K=5:             soft assignment with K=5
  v5  K=1+TNNR:         hard assignment + TNNR (control: TNNR alone vs SAF alone)
  v6  fixed_r=1:       TNNR with rank=1 (single-line only)
  v7  fixed_r=3:       TNNR with rank=3 (over-regularize)
  v8  large_lam_tnnr:   lam_tnnr = 0.20 (4x default)

Each variant ablates ONE component vs the v0 baseline.
"""
import os, sys, json, time, copy, argparse, warnings
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
from data import YorkUrbanSubset, collate_variable_lines
from model import build_lr_saf
from metrics import f_measure, s_ap


VARIANTS = {
    'v0_baseline':  dict(K=3, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5, fixed_r=None,
                          desc='full LR-SAF: SAF(K=3) + TNNR + junc + t*'),
    'v1_no_SAF':    dict(K=1, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5, fixed_r=None,
                          desc='K=1 hard assignment (no soft)'),
    'v2_no_TNN':    dict(K=3, lam_tnnr=0.0,  lam_junc=1.0, lam_t=0.5, fixed_r=None,
                          desc='SAF without TNNR regularizer'),
    'v3_no_junc':   dict(K=3, lam_tnnr=0.05, lam_junc=0.0, lam_t=0.5, fixed_r=2,
                          desc='SAF + TNNR(r=2 fixed), no junction head'),
    'v4_K5':        dict(K=5, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5, fixed_r=None,
                          desc='soft assignment K=5'),
    'v5_K1_TNN':    dict(K=1, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5, fixed_r=None,
                          desc='hard assignment + TNNR (TNNR alone vs SAF alone)'),
    'v6_r1_fixed':  dict(K=3, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5, fixed_r=1,
                          desc='TNNR with rank=1 only'),
    'v7_r3_fixed':  dict(K=3, lam_tnnr=0.05, lam_junc=1.0, lam_t=0.5, fixed_r=3,
                          desc='TNNR with rank=3 (likely over-reg)'),
    'v8_large_tnn': dict(K=3, lam_tnnr=0.20, lam_junc=1.0, lam_t=0.5, fixed_r=None,
                          desc='lam_tnnr = 0.20 (4x default)'),
}


def make_targets(lines_batch, mask_batch, H, W, device, K=3, sigma=2.0):
    B = lines_batch.shape[0]
    tgt = {'a': torch.zeros(B, 2, H, W, device=device),
           't_star': torch.zeros(B, 1, H, W, device=device),
           'junc':   torch.zeros(B, 1, H, W, device=device),
           'support': torch.zeros(B, H, W, device=device)}
    for b in range(B):
        n = int(mask_batch[b].sum().item())
        if n == 0: continue
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
def evaluate(model, val_items, in_res=320):
    from lib.squeeze_to_lsg import lsgenerator
    model.eval()
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    f_list, ap5, ap10, ap15 = [], [], [], []
    for item in val_items:
        H_o, W_o = item['H_orig'], item['W_orig']
        img = cv2.imread(os.path.join(item['root'], item['name'],
                                       f"{item['name']}.jpg"))
        x = cv2.resize(img, (in_res, in_res)).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
        out = model(x)
        offset = out['a'][0].cpu().numpy().astype(np.float32)
        lines, _, _ = lsgenerator(offset)
        lines = np.asarray(lines)
        if len(lines) == 0:
            f_list.append(0.0); ap5.append(0.0); ap10.append(0.0); ap15.append(0.0); continue
        sx, sy = W_o / in_res, H_o / in_res
        kept = lines[:, :4].copy(); kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        scores = (1.0 / (lines[:, 4] + 1e-3)) if lines.shape[1] >= 5 else np.ones(len(kept))
        DS = 4
        H_ds = max(H_o // DS, 32); W_ds = max(W_o // DS, 32)
        kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
        gp = item['gt_orig'].copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
        fm = f_measure(kp, gp, H_ds, W_ds)
        aps = s_ap(kept, scores, item['gt_orig'], H_o, W_o, thresholds=(5, 10, 15))
        f_list.append(fm['F']); ap5.append(aps[5]); ap10.append(aps[10]); ap15.append(aps[15])
    return (float(np.mean(f_list)), float(np.mean(ap5)),
            float(np.mean(ap10)), float(np.mean(ap15)))


def run_variant(name, conf, train_loader, val_items, epochs, lr,
                tnn_warmup_epochs=3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda'
    model = build_lr_saf(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best = {'F': -1, 'epoch': -1, 'ap5': 0, 'ap10': 0, 'ap15': 0}
    history = []
    print(f"\n--- variant {name}: {conf['desc']} ---")
    for ep in range(epochs):
        eff_lam_tnnr = 0.0 if ep < tnn_warmup_epochs else \
                      conf['lam_tnnr'] * min((ep - tnn_warmup_epochs + 1) / 3.0, 1.0)
        model.train()
        ep_L = []
        for batch in train_loader:
            optim.zero_grad()
            imgs = batch['image'].to(device)
            B, _, H, W = imgs.shape
            with torch.no_grad():
                tgt = make_targets(batch['lines'], batch['lines_mask'], H, W, device,
                                   K=conf['K'])
            out = model(imgs)
            a_enc = out['a']; t_pred = out['t_star']; junc_pred = out['junc']

            L_rec = weighted_l1(a_enc, tgt['a'], tgt['support'])
            L_junc = F.binary_cross_entropy(junc_pred.clamp(1e-6, 1-1e-6), tgt['junc'])
            L_t = (((t_pred.squeeze(1) - tgt['t_star'].squeeze(1)).abs() * tgt['support']).sum()
                   / tgt['support'].sum().clamp(min=1))
            if eff_lam_tnnr > 0:
                L_tnnr = tnn_loss(a_enc, junc_pred=junc_pred,
                                 window=9, stride=8,
                                 K_max=3, fixed_r=conf['fixed_r'], normalize=True)
            else:
                L_tnnr = torch.tensor(0.0, device=device)
            L = L_rec + eff_lam_tnnr * L_tnnr + conf['lam_junc'] * L_junc + conf['lam_t'] * L_t
            if not torch.isfinite(L): continue
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            ep_L.append(L.item())
        sched.step()
        F_val, ap5, ap10, ap15 = evaluate(model, val_items)
        history.append({'ep': ep, 'L': float(np.mean(ep_L)),
                        'F': F_val, 'ap5': ap5, 'ap10': ap10, 'ap15': ap15,
                        'lam_tnnr_eff': eff_lam_tnnr})
        print(f"  ep{ep:02d}: L={np.mean(ep_L):.3f} F={F_val:.4f} "
              f"sAP10={ap10:.4f} (lam_tnnr={eff_lam_tnnr:.3f})")
        if F_val > best['F']:
            best = {'F': F_val, 'epoch': ep, 'ap5': ap5, 'ap10': ap10, 'ap15': ap15}
    print(f"  best: epoch {best['epoch']}, F={best['F']:.4f} sAP10={best['ap10']:.4f}")
    return {'name': name, 'conf': conf, 'best': best, 'history': history}


def main(epochs=15, only=None):
    seed = 42; torch.manual_seed(seed); np.random.seed(seed)
    ds = YorkUrbanSubset(in_res=320)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds))
    train_idx, val_idx = perm[:80].tolist(), perm[80:].tolist()
    train_set = torch.utils.data.Subset(ds, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=2, shuffle=True,
        collate_fn=collate_variable_lines, num_workers=2)
    val_items = []
    for vi in val_idx:
        item = ds[vi]
        gt = item['lines'].numpy().copy()
        gt[:, 0::2] *= item['W_orig'] / 320.0; gt[:, 1::2] *= item['H_orig'] / 320.0
        val_items.append({'name': item['name'], 'root': ds.root, 'gt_orig': gt,
                          'H_orig': item['H_orig'], 'W_orig': item['W_orig']})

    variants_to_run = VARIANTS if only is None else \
                      {k: VARIANTS[k] for k in only.split(',') if k in VARIANTS}
    print(f"=== ABLATION ({len(variants_to_run)} variants x {epochs} epochs) ===")

    results = []
    for name, conf in variants_to_run.items():
        res = run_variant(name, conf, train_loader, val_items, epochs, lr=5e-5)
        results.append(res)
        # Persist after each variant
        with open(ROOT + '/logs/ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'variant':<18} {'F':>8} {'sAP5':>8} {'sAP10':>8} {'sAP15':>8} {'epoch':>6}")
    print("-" * 75)
    for r in results:
        b = r['best']
        print(f"{r['name']:<18} {b['F']:>8.4f} {b['ap5']:>8.4f} "
              f"{b['ap10']:>8.4f} {b['ap15']:>8.4f} {b['epoch']:>6}")
    print("=" * 75)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--only', type=str, default=None,
                    help='comma-separated subset of variant names')
    args = ap.parse_args()
    main(epochs=args.epochs, only=args.only)

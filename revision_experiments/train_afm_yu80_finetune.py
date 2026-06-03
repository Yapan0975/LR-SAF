"""Matched-protocol fine-tune of vanilla AFM on YorkUrban-80.

Reviewer M5: the rev4 headline used LR-SAF fine-tuned on YU-80 vs.
AFM/DeepLSD/HAWPv2 evaluated cross-domain (Wireframe checkpoints).
To close this, we fine-tune AFM with the same training protocol
LR-SAF used: 20 epochs, YorkUrban-80/22 split, seed 42, AFM target
encoding, L1 reconstruction loss only (no TNNR, junction, t*).

This is the AFM analogue of train_full.py. DeepLSD and HAWPv2 require
their own codebases for a similar treatment; we add those later.
"""

from __future__ import annotations

import os, sys, json, time, warnings, argparse
warnings.filterwarnings("ignore")

ROOT = "/home/server/Documents/yping/LR-SAF-LSD"
sys.path.insert(0, ROOT + "/code/lr_saf")
sys.path.insert(0, ROOT + "/code/afm_baseline")
sys.path.insert(0, ROOT + "/code/afm_baseline/lib")
os.chdir(ROOT + "/code/afm_baseline")

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from config import cfg
cfg.merge_from_file("experiments/afm_atrous.yaml")
from modeling.net import build_network
from lib.squeeze_to_lsg import lsgenerator

from data import YorkUrbanSubset, collate_variable_lines
from saf_target import compute_saf_target
from metrics import f_measure, s_ap


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_afm_target(lines_batch, mask_batch, H, W, device, D_max=80.0):
    """AFM-style log-encoded attraction-field targets (vanilla, hard partition)."""
    B = lines_batch.shape[0]
    a = torch.zeros(B, 2, H, W, device=device)
    support = torch.zeros(B, H, W, device=device)
    for b in range(B):
        n = int(mask_batch[b].sum().item())
        if n == 0:
            continue
        out = compute_saf_target(
            lines_batch[b, :n].to(device), H, W,
            sigma=0.1, K=1, D_max=D_max, bounded="afm", device=device,
        )
        a[b, 0] = out["a_x"]
        a[b, 1] = out["a_y"]
        support[b] = out["support"]
    return a, support


def weighted_l1(pred, gt, support, w=10.0):
    diff = (pred - gt).abs().sum(dim=1)
    weight = support * w + (1 - support) * 1.0
    return (weight * diff).mean()


@torch.no_grad()
def eval_yu22(model, val_items, in_res=320):
    model.eval()
    F_list, ap10 = [], []
    for it in val_items:
        img = cv2.imread(os.path.join(it["root"], it["name"], f"{it['name']}.jpg"))
        H_o, W_o = img.shape[:2]
        x = cv2.resize(img, (in_res, in_res)).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
        out = model(x)
        afm = out[0] if isinstance(out, (list, tuple)) else out
        offset = afm[0].cpu().numpy().astype(np.float32)
        lines, _, _ = lsgenerator(offset)
        lines = np.asarray(lines)
        if len(lines) == 0:
            F_list.append(0.0); ap10.append(0.0); continue
        sx, sy = W_o / in_res, H_o / in_res
        kept = lines[:, :4].copy()
        kept[:, 0::2] *= sx; kept[:, 1::2] *= sy
        scores = 1.0 / (lines[:, 4] + 1e-3) if lines.shape[1] >= 5 else np.ones(len(kept))
        DS = 4
        H_ds, W_ds = max(H_o // DS, 32), max(W_o // DS, 32)
        kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
        gp = it["gt_orig"].copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
        fm = f_measure(kp, gp, H_ds, W_ds)
        aps = s_ap(kept, scores, it["gt_orig"], H_o, W_o, thresholds=(10,))
        F_list.append(fm["F"]); ap10.append(aps[10])
    return float(np.mean(F_list)), float(np.mean(ap10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"

    full_ds = YorkUrbanSubset(in_res=320, limit=None)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(full_ds))
    train_idx, val_idx = perm[:80].tolist(), perm[80:].tolist()

    train_set = torch.utils.data.Subset(full_ds, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        collate_fn=collate_variable_lines, num_workers=2,
    )

    val_items = []
    for vi in val_idx:
        item = full_ds[vi]
        gt = item["lines"].numpy().copy()
        gt[:, 0::2] *= item["W_orig"] / 320.0
        gt[:, 1::2] *= item["H_orig"] / 320.0
        val_items.append({"name": item["name"], "root": full_ds.root, "gt_orig": gt,
                          "H_orig": item["H_orig"], "W_orig": item["W_orig"]})

    model = build_network(cfg).cuda()
    ckpt = torch.load(ROOT + "/checkpoints/atrous/weight/model_final.pth.tar",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    print(f"AFM loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = {"train_loss": [], "val_F": [], "val_sAP10": [], "epochs": []}
    best_F = -1.0
    D_MAX = 80.0
    for ep in range(args.epochs):
        ep_t = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad()
            imgs = batch["image"].to(device)
            B, _, H, W = imgs.shape
            with torch.no_grad():
                tgt_a, support = make_afm_target(batch["lines"], batch["lines_mask"], H, W, device, D_MAX)
            out = model(imgs)
            afm = out[0] if isinstance(out, (list, tuple)) else out
            L = weighted_l1(afm, tgt_a, support)
            if not torch.isfinite(L):
                continue
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(L.item())
        sched.step()

        F_val, sAP10_val = eval_yu22(model, val_items)
        elapsed = time.time() - ep_t
        log["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
        log["val_F"].append(F_val); log["val_sAP10"].append(sAP10_val); log["epochs"].append(ep)
        print(f"ep {ep:02d}: train_L={log['train_loss'][-1]:.4f} | val F={F_val:.4f} sAP10={sAP10_val:.4f} | {elapsed:.1f}s")

        if F_val > best_F:
            best_F = F_val
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_F": F_val, "val_sAP10": sAP10_val}, args.out)

    with open(args.log, "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"\nbest F={best_F:.4f} (AFM fine-tuned on YU-80, seed={args.seed})")


if __name__ == "__main__":
    main()

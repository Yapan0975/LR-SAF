"""Evaluate an LR-SAF checkpoint on YorkUrban val 22 (seed-42 split).

Used for EXP-B (Wireframe -> YorkUrban zero-shot) of the Paper 1 Major Rev.
The Wireframe-trained checkpoint at lr_saf_wireframe_best.pth is loaded
without any YorkUrban fine-tuning, then evaluated on the same 22-image
val set used by train_full.py.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

ROOT = "/home/server/Documents/yping/LR-SAF-LSD"
sys.path.insert(0, ROOT + "/code/lr_saf")
sys.path.insert(0, ROOT + "/code/afm_baseline")
sys.path.insert(0, ROOT + "/code/afm_baseline/lib")
os.chdir(ROOT + "/code/afm_baseline")

import torch, numpy as np, cv2
from lib.squeeze_to_lsg import lsgenerator
from data import YorkUrbanSubset
from model import build_lr_saf
from metrics import f_measure, s_ap

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_val_items(seed: int):
    ds = YorkUrbanSubset(in_res=320, limit=None)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds))
    val_idx = perm[80:].tolist()
    items = []
    for vi in val_idx:
        item = ds[vi]
        gt_orig = item["lines"].numpy().copy()
        gt_orig[:, 0::2] *= item["W_orig"] / 320.0
        gt_orig[:, 1::2] *= item["H_orig"] / 320.0
        items.append({"name": item["name"], "root": ds.root, "gt_orig": gt_orig,
                      "H_orig": item["H_orig"], "W_orig": item["W_orig"]})
    return items


def infer_and_score(model, item):
    name, H_o, W_o = item["name"], item["H_orig"], item["W_orig"]
    img = cv2.imread(os.path.join(item["root"], name, f"{name}.jpg"))
    x = cv2.resize(img, (320, 320)).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(x)
    offset = out["a"][0].cpu().numpy().astype(np.float32)
    lines, _, _ = lsgenerator(offset)
    lines = np.asarray(lines)
    if len(lines) == 0:
        return 0.0, {5: 0.0, 10: 0.0, 15: 0.0}, 0
    sx, sy = W_o / 320.0, H_o / 320.0
    kept = lines[:, :4].copy()
    kept[:, 0::2] *= sx
    kept[:, 1::2] *= sy
    scores = 1.0 / (lines[:, 4] + 1e-3) if lines.shape[1] >= 5 else np.ones(len(kept))
    DS = 4
    H_ds, W_ds = max(H_o // DS, 32), max(W_o // DS, 32)
    kp = kept.copy(); kp[:, 0::2] /= DS; kp[:, 1::2] /= DS
    gp = item["gt_orig"].copy(); gp[:, 0::2] /= DS; gp[:, 1::2] /= DS
    fm = f_measure(kp, gp, H_ds, W_ds)
    aps = s_ap(kept, scores, item["gt_orig"], H_o, W_o, thresholds=(5, 10, 15))
    return fm["F"], aps, len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to LR-SAF checkpoint (.pth) to evaluate.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Split seed for YorkUrban-22 val (must match train_full.py).")
    ap.add_argument("--label", default="zero-shot")
    args = ap.parse_args()

    items = get_val_items(args.seed)
    model = build_lr_saf(device="cuda").eval()
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"loaded {args.ckpt}; evaluating {args.label} on {len(items)} val images")

    rows = []
    for it in items:
        F, ap, n = infer_and_score(model, it)
        rows.append({"name": it["name"], "F": F, "sAP_5": ap[5],
                     "sAP_10": ap[10], "sAP_15": ap[15], "n_pred": n})

    agg = {
        "label": args.label,
        "ckpt": args.ckpt,
        "n_images": len(rows),
        "F_mean": float(np.mean([r["F"] for r in rows])),
        "sAP_5_mean": float(np.mean([r["sAP_5"] for r in rows])),
        "sAP_10_mean": float(np.mean([r["sAP_10"] for r in rows])),
        "sAP_15_mean": float(np.mean([r["sAP_15"] for r in rows])),
        "F_std": float(np.std([r["F"] for r in rows])),
        "sAP_10_std": float(np.std([r["sAP_10"] for r in rows])),
        "per_image": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"{args.label:>20s}: F={agg['F_mean']:.4f}, sAP_10={agg['sAP_10_mean']:.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

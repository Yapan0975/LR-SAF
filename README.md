# LR-SAF: A Component-wise Analysis of AFM-Style Line Segment Detection

The paper evaluates four AFM-style components — soft top-K assignment, a 9-d
MLP ranking head, optional semantic features, and a truncated nuclear norm
(TNNR) regularizer — under a matched in-domain fine-tuning protocol on
YorkUrban. After re-scoring every condition with the standard dataset-level
LCNN sAP evaluator (the earlier draft used a more lenient per-image-averaged
metric), the audit is a diagnostic negative-result audit: under the corrected
metric the proposed components do not deliver a net gain over the AFM baseline.
The findings are:

1. Under 5-fold held-out CV (standard dataset-level LCNN sAP-10), the
   soft-assignment LR-SAF backbone scores *below* the AFM backbone
   (0.010 ± 0.003 vs. AFM 0.043 ± 0.014). Adding the 9-d MLP geometry head
   raises both, but LR-SAF+head (0.039 ± 0.006 fold-wise, 0.037 pooled) still
   trails AFM+head (0.122 ± 0.023 fold-wise, 0.115 pooled). The paired
   difference favours AFM by +0.078 sAP-10 (95% bootstrap CI
   [+0.064, +0.095]).
2. The 9-d MLP ranking head, not the backbone, accounts for the bulk of any
   sAP-10 movement; it helps the matched AFM backbone more than the LR-SAF
   one. On a frozen external Wireframe test the same ordering holds
   (AFM+head 0.185 vs. LR-SAF+head 0.055 sAP-10).
3. The TNNR regularizer has a negligible isolated effect: mean Δ = +0.002
   sAP-10 (range [−0.001, +0.006]) and is empirically indistinguishable from a
   total-variation regularizer at the same scheduled weight.

## Code organization

```
bounded.py                  bounded encoding (Eq. 5)
saf_target.py               soft attraction field target (Eq. 3-4)
tnn_loss.py                 truncated nuclear norm + DC decomposition
diff_squeeze.py / _v2.py    differentiable squeeze decoder
confidence_head.py          9-d MLP confidence head
semantic_features.py        VOC-21 semantic feature head
dinov2_features.py          DINOv2 self-supervised feature head

model.py                    a-trous Residual U-Net backbone
data.py / data_hawp.py      YorkUrban + Wireframe loaders
metrics.py                  F-measure + sAP-5/10/15

train_ablation.py           main training entry (9 ablation variants)
train_confidence.py         confidence-head training (geometry only)
train_confidence_dinov2.py  + DINOv2 features
train_conf_wireframe.py     full Wireframe training
eval_afm_full.py            AFM baseline reproduction
eval_compare.py             cross-method evaluation
eval_deeplsd.py             DeepLSD reproduction
robustness_eval.py          degradation sweep (noise / blur / low-light)
robustness_4way.py          4-way comparison driver
compare_hawp_lrsaf.py       HAWPv2 vs LR-SAF
bench_efficiency.py         FPS / params / FLOPs (Table 11)
plot_robustness.py          Figure 1 generator

revision_experiments/       reviewer-driven scripts (multi-seed, AFM matched
                            fine-tune, TV-regularizer comparison, etc.)
```

## Reproducing the standard-metric numbers

1. **Environment.** Python 3.10, PyTorch 2.2, CUDA 12.1. Install the Python
   dependencies with `pip install -r requirements.txt`.
2. **Data.** YorkUrban (102 images) and Wireframe (5000 images). The repo
   does not ship data; symlink the original releases into a local `data/`
   directory.
3. **AFM baseline + matched fine-tune.**
   ```bash
   python eval_afm_full.py --dataset yorkurban
   bash revision_experiments/afm_multiseed.sh
   ```
4. **LR-SAF backbone.**
   ```bash
   bash revision_experiments/exp_c_multiseed_yu80.sh
   python revision_experiments/aggregate_multiseed.py
   ```
5. **Confidence head on matched AFM (paired backbone-vs-head comparison).**
   ```bash
   bash revision_experiments/afm_head_multiseed.sh
   python revision_experiments/aggregate_afm_multiseed.py
   ```
6. **Robustness sweep + Figure 1.**
   ```bash
   python robustness_4way.py
   python plot_robustness.py
   ```
7. **TNNR causal ablation and TV comparison.**
   ```bash
   bash revision_experiments/rev9_experiments.sh
   python revision_experiments/rev9_aggregate.py
   ```

Cross-validation aggregation, the paired-difference and image-resample
bootstrap analysis, and the LaTeX table fillers live in
`revision_experiments/` and `scripts/`.

## Reproducibility artifacts

This release bundles the configurations, split files, and raw predictions
requested during review:

- **Corrected standard-metric evaluator.** `metrics.py` now computes the
  standard dataset-level LCNN sAP via `image_records` (per-image
  prediction / GT / score records) and `sap_dataset` (the dataset-level sAP
  aggregation). `main_table_sap.py`, `dump_backbone_records.py`,
  `eval_wireframe_external.py`, `train_confidence.py`, and
  `revision_experiments/train_conf_on_afm.py` are the matching corrected
  drivers. These supersede the earlier per-image-averaged metric.
- **CV runner scripts** in `scripts/`: `run_cv.sh`, `run_cv_fold.sh`,
  `run_cv_sap.sh`, `run_cv_sap_fold.sh`.
- **Split indices and protocol** in `configs/`:
  `cv_split.md` documents the exact 5-fold and 80/22 protocols, and
  `cv_fold_indices.json` lists the materialised 5-fold indices.
- **Raw predictions / results** in `predictions/`:
  - `predictions/cv_sap/` — 20 per-image record files (per-image
    predictions, GT, and scores) for the four conditions
    (`afm_bb`, `lrsaf_bb`, `afm_head`, `lrsaf_head`) across folds 0-4,
    behind the CV sAP tables.
  - `predictions/main_table_sap.json`,
    `predictions/wireframe_external_frozen.json`,
    `predictions/robustness_SAPFIX.json`,
    `predictions/tnnr_causal_SAPFIX.json` — the aggregate result JSONs.

### Split protocols (summary)

- **5-fold CV.** Folds are
  `numpy.array_split(numpy.random.RandomState(0).permutation(102), 5)`
  over the 102 YorkUrban images (fold sizes 21, 21, 20, 20, 20). One
  fine-tune per fold; the final-epoch model is scored on the held-out fold.
- **Main 80/22 split.** `numpy.random.RandomState(42).permutation(102)`;
  the first 80 indices are training, the last 22 are the held-out
  validation set.

See `configs/cv_split.md` for the full description and reproduction snippets.

## Notes

- All training was done on a single RTX 5090. Wireframe training took ~12 h
  per seed for the full 50 epochs.
- The DINOv2 feature head was evaluated as an alternative to VOC-21 semantics
  but did not improve on small-data conditions.
- TNNR (truncated nuclear norm with adaptive rank
  `r(W) = 1 + floor((K_max − 1) · s(W))`, `K_max = 3`, where `s(W)` is the
  per-window junction strength) is implemented in `tnn_loss.py`; the DC
  decomposition makes the term amenable to standard non-convex SGD/Adam.
  Note that in the trained model the predicted junction strength `s(W)` never
  crosses the 0.5 threshold, so the rule degenerates to `r = 1` for every
  window — the adaptive rank is effectively a fixed rank-1 nuclear-norm
  penalty in practice.

## Citation

If you use this code, please cite the manuscript (final reference will be
updated upon acceptance).


## License

MIT.

Corresponding author: Cai Jing <caijing@zjjcxy.cn>, Zhejiang Police College.

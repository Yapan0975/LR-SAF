# LR-SAF: A Component-Wise Audit of AFM-Style Line Segment Detection

Reference code for the manuscript

> **A Component-Wise Audit of AFM-Style Line Segment Detection**
> Yang Ping, Chen Zhang, Xu Ruoyi, Chen Dongjie, Cai Jing
> Submitted to *Pattern Recognition* (Elsevier), 2026.

The paper evaluates four AFM-style components — soft top-K assignment, a 9-d
MLP ranking head, optional semantic features, and a truncated nuclear norm
(TNNR) regularizer — under a matched in-domain fine-tuning protocol on
YorkUrban. The audit produces three findings:

1. The soft-assignment backbone gives a small seed-consistent F-measure gain
   over AFM (0.7783 ± 0.0086 vs. 0.7645 ± 0.0108) but does not improve sAP-10.
2. The 9-d MLP ranking head accounts for most of the reported sAP-10
   improvement; on a matched AFM backbone the same head reaches 0.356 ± 0.014.
3. TNNR contributes only +0.007 mean sAP-10 under degradation and is
   empirically indistinguishable from a total-variation regularizer at the
   same scheduled weight.

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

## Reproducing the headline numbers

1. **Environment.** Python 3.10, PyTorch 2.2, CUDA 12.1. See `requirements.txt`
   (TODO).
2. **Data.** YorkUrban (102 images) and Wireframe (5000 images). The repo
   does not ship data; symlink the original releases into a local `data/`
   directory.
3. **AFM baseline + matched fine-tune.**
   ```bash
   python eval_afm_full.py --dataset yorkurban
   bash revision_experiments/afm_multiseed.sh
   ```
4. **LR-SAF backbone (3-seed).**
   ```bash
   bash revision_experiments/exp_c_multiseed_yu80.sh
   python revision_experiments/aggregate_multiseed.py
   ```
5. **Confidence head on matched AFM (Table 13).**
   ```bash
   bash revision_experiments/afm_head_multiseed.sh
   python revision_experiments/aggregate_afm_multiseed.py
   ```
6. **Robustness sweep + Figure 1.**
   ```bash
   python robustness_4way.py
   python plot_robustness.py
   ```
7. **TNNR causal ablation (Table 6) + TV comparison (Table 9).**
   ```bash
   bash revision_experiments/rev9_experiments.sh
   python revision_experiments/rev9_aggregate.py
   ```

Multi-seed aggregation, Welch t-tests and the LaTeX table fillers live in
`revision_experiments/`.

## Notes

- All training was done on a single RTX 5090. Wireframe training took ~12 h
  per seed for the full 50 epochs.
- The DINOv2 feature head was evaluated as an alternative to VOC-21 semantics
  but did not improve on small-data conditions.
- TNNR (truncated nuclear norm with adaptive rank `r(W) = 2·K(W)`) is
  implemented in `tnn_loss.py`; the DC decomposition makes the term
  amenable to standard non-convex SGD/Adam.

## Citation

If you use this code, please cite the manuscript (final reference will be
updated upon acceptance).

```bibtex
@article{lrsaf2026,
  title={A Component-Wise Audit of AFM-Style Line Segment Detection},
  author={Yang, Ping and Chen, Zhang and Xu, Ruoyi and Chen, Dongjie and Cai, Jing},
  journal={Pattern Recognition (under review)},
  year={2026}
}
```

## License

MIT.

Corresponding author: Cai Jing <caijing@zjjcxy.cn>, Zhejiang Police College.

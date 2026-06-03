# Major Revision Experiments

These scripts respond to reviewer comment M5 (实验公平性). They are designed
to run on the remote `evidlife-server` (4×RTX 5090) in
`~/Documents/yping/LR-SAF-LSD/code/lr_saf/revision_experiments/`.

## Experiments to launch

### EXP-A: Baselines on YorkUrban-80 fine-tune
Reviewer concern: AFM/DeepLSD/HAWPv2 evaluated cross-domain (Wireframe→YorkUrban),
while LR-SAF was fine-tuned on YorkUrban-80. Re-train all baselines with the
same 80/22 split and seed 42 to get apples-to-apples comparison.

- `exp_a_finetune_afm_yu80.py` — fine-tune AFM checkpoint on YorkUrban-80, 20 ep
- `exp_a_finetune_deeplsd_yu80.py` — fine-tune DeepLSD on YorkUrban-80, 20 ep
- `exp_a_finetune_hawpv2_yu80.py` — fine-tune HAWPv2 on YorkUrban-80, 20 ep

Estimated wall-clock per script: 1.5–3 h on one RTX 5090.

### EXP-B: LR-SAF Wireframe→YorkUrban zero-shot
Train LR-SAF on Wireframe-5000 only (no YorkUrban fine-tune), evaluate on
YorkUrban val 22. Reports the "fair" cross-domain LR-SAF number.

- `exp_b_lrsaf_wireframe_only.py` — train LR-SAF on Wireframe full, 50 ep

Estimated wall-clock: 4–6 h.

### EXP-C: 3-seed mean ± std for LR-SAF main results
Replicate the LR-SAF main runs with seeds {42, 17, 2024} on both YorkUrban-80
fine-tune and Wireframe full. Report mean ± std on all metrics.

- `exp_c_multiseed_yu80.sh` — driver for 3-seed YorkUrban runs
- `exp_c_multiseed_wireframe.sh` — driver for 3-seed Wireframe runs

Estimated total wall-clock: 3 × (3 h + 6 h) ≈ 27 h on one RTX 5090,
parallelizable to ~7 h on the 4-GPU cluster.

## Reporting

After completion, aggregate into:
- `logs/exp_a_baseline_finetune.json`
- `logs/exp_b_lrsaf_zeroshot.json`
- `logs/exp_c_multiseed_lrsaf.json`

Update `D:\_7_sci\LSD\手稿草稿\theorems\results_log.md` and
`latex\sections\07_experiments.tex` Table III with the new rows.

## Status

- [ ] EXP-A AFM fine-tune
- [ ] EXP-A DeepLSD fine-tune
- [ ] EXP-A HAWPv2 fine-tune
- [ ] EXP-B LR-SAF Wireframe-only
- [ ] EXP-C multi-seed LR-SAF YorkUrban (×3)
- [ ] EXP-C multi-seed LR-SAF Wireframe (×3)
- [ ] Aggregate + table update + recompile

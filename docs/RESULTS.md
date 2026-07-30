# Audited K=4 results

The table below was reconstructed from each fold's explicit checkpoint_best.pth validation directory and the original-GT-only ET/TC/WT evaluation summaries.

| Fold | Cases | ET Dice | TC Dice | WT Dice | Mean | nnU-Net foreground Dice | best checkpoint current_epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 271 | 0.873858 | 0.915120 | 0.932792 | 0.907257 | 0.834997 | 496 |
| 1 | 270 | 0.870550 | 0.899720 | 0.914930 | 0.895066 | 0.835550 | 246 |
| 2 | 270 | 0.878646 | 0.926043 | 0.934984 | 0.913224 | 0.841428 | 443 |
| 3 | 270 | 0.857453 | 0.899162 | 0.923698 | 0.893438 | 0.814202 | 1 |
| 4 | 270 | 0.873947 | 0.921011 | 0.933427 | 0.909462 | 0.842952 | 332 |

Case-weighted five-fold means:

| Metric | Value |
|---|---:|
| ET Dice | 0.870893 |
| TC Dice | 0.912213 |
| WT Dice | 0.927970 |
| ET/TC/WT mean Dice | 0.903692 |

The checkpoint current_epoch field is nnU-Net metadata. For example, fold 0 checkpoint current_epoch=496 corresponds to the last best log event during zero-based log epoch 495. It should not be presented as a one-based publication epoch without explaining this convention.

## Gate audit

The preserved K-selection report characterized fold-0 K=4 as optional_diagnostic:

- 100 cases with gate logs;
- mean gates: 0.2555, 0.2537, 0.2372, 0.2536;
- gate entropy mean: 1.3858;
- gate max mean: 0.2555;
- dominant expert fraction: 1.0;
- collapse_warning: true;
- fold/source artifact warnings: false.

Near-uniform probabilities do not eliminate the dominant-expert collapse warning because the same expert won argmax for all audited cases. Do not claim that four specialized experts emerged from this evidence.

## Checkpoint averaging artifact

A post-training arithmetic mean over all 995 floating network tensors of the five checkpoint_best.pth files was preserved as checkpoint_average_best_5fold.pth. Its SHA-256 and the input hashes are listed in provenance/checkpoint_checksums.md.

Weight averaging is not the same operation as prediction-probability ensembling. The presence of this artifact does not establish that it is superior or that it was used for every submission. scripts/average_nnunet_checkpoints.py is provided only to reproduce the recorded arithmetic operation.

## Interpretation boundary

These are local original-GT-only cross-validation values. They are not official BraTS leaderboard scores, do not include NSD/HD95 unless recomputed, and are not evidence of clinical performance.

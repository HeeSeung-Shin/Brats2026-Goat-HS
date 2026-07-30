# BraTS 2026 GOAT Dataset007 ResEnc-M SoftMoE K=4

This repository contains code and configuration for the five-fold experiment:

| Field | Value |
|---|---|
| Dataset | Dataset007_Brats26_Goat_MLConsensusPseudo |
| Backbone/plans | nnUNetResEncUNetMPlans_D005Compat |
| Trainer | nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot |
| Configuration | 3d_fullres |
| Folds | 0, 1, 2, 3, 4 |
| Epochs | 500 |
| SoftMoE experts | 4 |
| Initialization | fold-matched Dataset005 ResEnc-M checkpoint_best.pth |

> Raw MRI, labels, pseudo-labels, case identifiers, split files, case weights, predictions, and checkpoints are not included. Obtain and use BraTS assets only under the applicable challenge/Synapse terms.

한국어 사용법은 [README_KO.md](README_KO.md)를 참고하십시오.

## Method

The standard nnU-Net ResEnc-M encoder/decoder is augmented with:

- a residual SoftMoE adapter at the deepest encoder feature;
- four lightweight 3D convolutional experts;
- case-level soft routing from global-average-pooled bottleneck features;
- temperature 1.0, balance-loss weight 0.001, and adapter scale initialized to 0.1;
- an ET/TC/WT auxiliary head used only during training;
- pseudo-label loss weighting with the ramp curriculum;
- fold-specific ET-aware case-sampling weights.

The main four-class nnU-Net segmentation head remains the inference output. No EDA cluster or hard routing label is supplied to the gate.

## Repository layout

    config/              dataset.json, ResEnc-M plans, and environment defaults
    docs/                data preparation, experiment configuration, and results
    private_assets/      local-only asset contract; contents are Git-ignored
    provenance/          source and checkpoint SHA-256 records
    requirements/        pinned Python dependencies
    scripts/data/        pseudo-label and Dataset007 preparation tools
    scripts/             setup, verification, training, validation, and evaluation
    src/nnunet_overlays/ custom trainer import closure

## Quick start

Create an isolated environment:

    conda env create -f environment.yml
    conda activate brats-goat-resencm-softmoe-k4
    export VENV_DIR="$CONDA_PREFIX"
    export PYTHON_BOOTSTRAP_BIN=python
    DRY_RUN=1 bash scripts/bootstrap.sh
    bash scripts/bootstrap.sh

The bootstrap clones pinned nnU-Net commit `f6d221d1b79cd2173650f78f97ecfee273e0cf86` and installs the five-file trainer overlay.

Set private paths in the shell:

    export nnUNet_raw=/absolute/private/path/nnUNet_raw
    export nnUNet_preprocessed=/absolute/private/path/nnUNet_preprocessed
    export nnUNet_results=/absolute/private/path/nnUNet_results_K4
    export PRIVATE_MANIFEST=/absolute/private/path/dataset007_case_manifest.csv
    export PRIVATE_SPLITS=/absolute/private/path/splits_final.json
    export CASE_WEIGHT_ROOT=/absolute/private/path/case_weights
    export D005_PRETRAINED_ROOT=/absolute/private/path/d005_resencm_experiment
    export D005_LABELS_DIR=/absolute/private/path/Dataset005/labelsTr

Verify and prepare:

    python scripts/verify_assets.py
    python scripts/verify_assets.py --hash
    DRY_RUN=1 bash scripts/prepare_preprocessing.sh
    bash scripts/prepare_preprocessing.sh

Train one fold:

    DRY_RUN=1 bash scripts/train_fold.sh 0
    bash scripts/train_fold.sh 0

Preflight all folds, then explicitly launch five sequential 500-epoch jobs:

    bash scripts/train_5fold.sh
    CONFIRM_FULL_TRAINING=YES bash scripts/train_5fold.sh --execute

Validate checkpoint_best.pth and compute original-GT-only ET/TC/WT Dice:

    bash scripts/validate_fold.sh 0 --dry-run
    bash scripts/validate_fold.sh 0
    bash scripts/evaluate_fold.sh 0 --dry-run
    bash scripts/evaluate_fold.sh 0

Use `--final` with validate_fold.sh only when checkpoint_final.pth is intended.

## Results and limitation

The five-fold case-weighted original-GT-only means were ET 0.8709, TC 0.9122, WT 0.9280, and mean Dice 0.9037. These are local validation values, not an official challenge leaderboard score.

The fold-0 gate analysis reported near-uniform gate probabilities but a dominant-expert fraction of 1.0 and `collapse_warning=True`. Treat K=4 as an experimental configuration, not evidence of clinically meaningful expert specialization. See [docs/RESULTS.md](docs/RESULTS.md).

## Reproducibility boundary

Exact reproduction requires the frozen manifest, five-fold split, fold-specific case weights, case metadata, and five fold-matched Dataset005 ResEnc-M initialization checkpoints. Their hashes are recorded, but the files are not distributed.

The run is not guaranteed bitwise deterministic because GPU kernels, AMP, augmentation workers, driver/toolchain differences, and scheduling may change numerical results.

## Documentation

- [Data preparation](docs/DATA_PREPARATION.md)
- [Exact experiment](docs/EXPERIMENT.md)
- [Results](docs/RESULTS.md)
- [Checkpoint hashes](provenance/checkpoint_checksums.md)

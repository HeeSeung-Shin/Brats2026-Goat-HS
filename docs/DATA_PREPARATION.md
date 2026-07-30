# Dataset007 data preparation

No BraTS image, label, pseudo-label, prediction, case list, or private metadata is distributed in this repository.

## Required supervised dataset

The Dataset007 experiment contains 2,334 supervised cases:

| Source | Count | Training role |
|---|---:|---|
| Dataset005 ground truth | 1,351 | original_labeled |
| Strict M/L consensus pseudo-label | 983 | pseudo_labeled_ml_strict |
| Total | 2,334 | imagesTr + labelsTr |

Another 155 cases remain under imagesUn and are not part of supervised training.

Expected layout:

    nnUNet_raw/
    ├── Dataset005_Brats26_Goat_With_GroundTruth/
    │   └── labelsTr/
    └── Dataset007_Brats26_Goat_MLConsensusPseudo/
        ├── dataset.json
        ├── dataset007_case_manifest.csv
        ├── imagesTr/
        ├── labelsTr/
        └── imagesUn/

Each case uses four nnU-Net channels:

| Channel | Modality |
|---|---|
| _0000 | t1n |
| _0001 | t1c |
| _0002 | t2w |
| _0003 | t2f |

Labels are 0 background, 1 NCR, 2 ED, and 3 ET.

## Frozen split

The exact splits_final.json must be placed in:

    nnUNet_preprocessed/Dataset007_Brats26_Goat_MLConsensusPseudo/splits_final.json

| Fold | Train | GT in train | Pseudo in train | Validation |
|---:|---:|---:|---:|---:|
| 0 | 2,063 | 1,080 | 983 | 271 |
| 1 | 2,064 | 1,081 | 983 | 270 |
| 2 | 2,064 | 1,081 | 983 | 270 |
| 3 | 2,064 | 1,081 | 983 | 270 |
| 4 | 2,064 | 1,081 | 983 | 270 |

Every validation case is from the original Dataset005 GT set; all 983 pseudo-labeled cases remain in training in every fold.

## Fold-specific sampling assets

The trainer requires:

    case_weights/
    ├── d007_case_region_cluster_meta.csv
    ├── case_weights_fold0.json
    ├── case_weights_fold1.json
    ├── case_weights_fold2.json
    ├── case_weights_fold3.json
    └── case_weights_fold4.json

The original trainer silently falls back to uniform sampling when these files are missing. The repository wrapper rejects that fallback for exact reproduction.

## D005 initialization

Each Dataset007 fold is initialized from the same-numbered standard ResEnc-M Dataset005 checkpoint:

    pretrained_d005_resencm/
    ├── fold_0/checkpoint_best.pth
    ├── fold_1/checkpoint_best.pth
    ├── fold_2/checkpoint_best.pth
    ├── fold_3/checkpoint_best.pth
    └── fold_4/checkpoint_best.pth

Exact checkpoint hashes are in [../provenance/checkpoint_checksums.md](../provenance/checkpoint_checksums.md).

## Optional pseudo-label reconstruction

The scripts/data directory preserves the preparation stages used for the ML-consensus dataset. Run every script with --help and verify all paths before execution.

A typical sequence is:

    python scripts/data/check_resenc_ml_inputs.py --help
    bash scripts/data/run_resencML_pseudolabel_inference.sh --help
    bash scripts/data/run_fuse_and_qc_ml_pseudolabels.sh --help
    python scripts/data/build_dataset007_mlconsensus.py --help
    python scripts/data/create_dataset007_gtval_splits.py --help
    python scripts/data/build_et_tc_case_meta.py --help
    python scripts/data/make_etaware_case_weights.py --help

Teacher inference and probability fusion are expensive and require separately authorized teacher checkpoints. They are not run by bootstrap.sh or prepare_preprocessing.sh.

## Install and verify private assets

Export paths without editing tracked config:

    export nnUNet_raw=/private/nnUNet_raw
    export nnUNet_preprocessed=/private/nnUNet_preprocessed
    export PRIVATE_MANIFEST=/private/dataset007_case_manifest.csv
    export PRIVATE_SPLITS=/private/splits_final.json
    export CASE_WEIGHT_ROOT=/private/case_weights
    export D005_PRETRAINED_ROOT=/private/pretrained_d005_resencm
    export D005_LABELS_DIR=/private/Dataset005/labelsTr

Run:

    python scripts/verify_assets.py
    python scripts/verify_assets.py --hash
    DRY_RUN=1 bash scripts/prepare_preprocessing.sh
    bash scripts/prepare_preprocessing.sh

The hash mode reads approximately 4 GB of initialization checkpoints and is intentionally slower.

# Private assets (not tracked)

Only this README belongs in Git. Place authorized assets here or point to equivalent external paths with environment variables.

Expected local layout:

    private_assets/
    ├── dataset007_case_manifest.csv
    ├── splits_final.json
    ├── dataset005_labelsTr/
    │   └── <case>.nii.gz
    ├── case_weights/
    │   ├── d007_case_region_cluster_meta.csv
    │   ├── case_weights_fold0.json
    │   ├── case_weights_fold1.json
    │   ├── case_weights_fold2.json
    │   ├── case_weights_fold3.json
    │   └── case_weights_fold4.json
    └── pretrained_d005_resencm/
        ├── fold_0/checkpoint_best.pth
        ├── fold_1/checkpoint_best.pth
        ├── fold_2/checkpoint_best.pth
        ├── fold_3/checkpoint_best.pth
        └── fold_4/checkpoint_best.pth

The full Dataset007 imagesTr/labelsTr/imagesUn directories normally remain under an external nnUNet_raw root, not here.

Environment overrides:

    export nnUNet_raw=/private/nnUNet_raw
    export nnUNet_preprocessed=/private/nnUNet_preprocessed
    export nnUNet_results=/private/nnUNet_results_K4
    export PRIVATE_MANIFEST=/private/dataset007_case_manifest.csv
    export PRIVATE_SPLITS=/private/splits_final.json
    export CASE_WEIGHT_ROOT=/private/case_weights
    export D005_PRETRAINED_ROOT=/private/pretrained_d005_resencm
    export D005_LABELS_DIR=/private/Dataset005/labelsTr

Run python scripts/verify_assets.py --hash before training. Never stage the private files above.

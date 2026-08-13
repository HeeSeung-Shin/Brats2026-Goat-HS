# Private inputs (not distributed)

Only this README belongs in Git. BraTS/Synapse data, teacher probabilities,
pseudo-labels, case manifests, sampling weights, and checkpoints must remain in
authorized external storage.

The final public workflow expects:

```text
private_assets/
├── splits_final.json
├── dataset005_labelsTr/<case>.nii.gz
├── pseudolabels_resencML_5fold_best/
│   ├── raw_resencM_5fold/<case>.npz
│   ├── raw_resencL_5fold/<case>.npz
│   ├── fused_labels/<case>.nii.gz
│   └── manifests/pseudo_labeled_ml_strict_cases.txt
├── case_weights/
│   └── case_weights_fold{0,1,2,3,4}.json
└── pretrained_d005_resencm/
    └── fold_{0,1,2,3,4}/checkpoint_best.pth
```

Each fold weight JSON is an externally supplied mapping from exactly that
fold's training case IDs to finite weights in `[1.1, 2.0]`. The public code
validates and normalizes these weights for replacement sampling; it does not
invent an unpublished generation formula.

Override locations with `PRIVATE_SPLITS`, `CASE_WEIGHT_ROOT`,
`D005_PRETRAINED_ROOT`, and `D005_LABELS_DIR` after sourcing
`config/experiment.env`. Never stage private files in this directory.

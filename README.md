# AIstatLab at BraTS-GoAT 2026 Task 3: Quality-Controlled Consensus Pseudo-Labeling and Dense Expert Adaptation

This repository implements the paper's final **ResEnc-M, K=4, five-fold**
system. It contains the consensus pseudo-label/QC path, mixed-supervision
student trainer, final regional submission inference, evaluation, and
lightweight tests.

The two five-fold ResEnc-M/L teacher families generate logits that are equally
averaged within each family before softmax. Their ET/TC/WT probabilities are
fused with ResEnc-M weights `0.70/0.50/0.40`, thresholded at `0.50`, corrected
to `ET ⊆ TC ⊆ WT`, and screened by the paper's confidence, uncertainty,
coverage, component, and inter-teacher agreement rules. The paper reports 983
retained cases from 1,138 unlabeled cases, used with 1,351 annotated cases.

The student is a six-stage ResEnc-M with channels
`[32, 64, 128, 256, 320, 320]`, blocks `[1, 3, 4, 6, 6, 6]`, a training-only
ET/TC/WT auxiliary head, and a dense K=4 adapter at the final 320-channel
encoder skip. All experts process every patch and are combined by soft routing.
`SoftMoE` is retained only as the internal class identifier for this dense
expert adapter. The auxiliary loss weights are ET `0.2`, TC `0.2`, WT `0.1`;
the adapter uses temperature `1`, residual scale initialization `0.1`, and
balance coefficient `0.001`.

## Installation

```bash
conda env create -f environment.yml
conda activate brats-goat-resencm-softmoe-k4
export VENV_DIR="${CONDA_PREFIX}" PYTHON_BOOTSTRAP_BIN=python
bash scripts/bootstrap.sh
source config/experiment.env
```

`requirements/base.txt` pins the paper runtime and `bootstrap.sh` installs the
pinned nnU-Net revision plus the repository overlay. Set `nnUNet_raw`,
`nnUNet_preprocessed`, and `nnUNet_results` before sourcing the configuration
when using other authorized storage roots.

## Required inputs

BraTS/Synapse terms prevent this repository from distributing MRI, ground
truth, derived pseudo-labels, or checkpoints. The workflow therefore requires:

```text
${nnUNet_raw}/Dataset005_Brats26_Goat_With_GroundTruth/
├── imagesTr/<case>_{0000,0001,0002,0003}.nii.gz
├── labelsTr/<case>.nii.gz
└── imagesUn/<case>_{0000,0001,0002,0003}.nii.gz

private_assets/
├── splits_final.json
├── case_weights/case_weights_fold{0,1,2,3,4}.json
└── pretrained_d005_resencm/fold_{0,1,2,3,4}/checkpoint_best.pth
```

The ResEnc-M/L teacher checkpoints and the five final student checkpoints must
also be supplied through authorized nnU-Net result storage. Fold-specific
ET-aware weight JSONs are required inputs but are not distributed with the
public repository. Their generation formula is not specified by the paper, so
the public code only validates exact case IDs, finite values in `[1.1, 2.0]`,
normalization for replacement sampling, and optionally the approximately 55.5%
annotated sampling mass. See `private_assets/README.md` for the full layout.

## Preprocessing

After pseudo-label construction and split preparation below:

```bash
bash scripts/prepare_preprocessing.sh --np 4
```

This installs the provided final 3D plan and annotated-only validation splits,
then runs nnU-Net fingerprinting and preprocessing. The final plan uses patch
size `128×160×112` and batch size 2.

## Pseudo-label generation and strict QC

Register five `checkpoint_best.pth` folds for both teacher plans in the
authorized nnU-Net result tree, then run:

```bash
bash scripts/data/run_resencML_pseudolabel_inference.sh --execute --gpu-id 0
bash scripts/data/run_fuse_and_qc_ml_pseudolabels.sh --overwrite

python scripts/data/build_dataset007_mlconsensus.py \
  --source-dataset-root "${nnUNet_raw}/Dataset005_Brats26_Goat_With_GroundTruth" \
  --pseudolabel-root private_assets/pseudolabels_resencML_5fold_best \
  --target-dataset-root "${nnUNet_raw}/Dataset007_Brats26_Goat_MLConsensusPseudo" \
  --overwrite true

python scripts/data/create_dataset007_gtval_splits.py \
  --dataset007-root "${nnUNet_raw}/Dataset007_Brats26_Goat_MLConsensusPseudo" \
  --dataset005-splits "${nnUNet_preprocessed}/Dataset005_Brats26_Goat_With_GroundTruth/splits_final.json" \
  --output-json private_assets/splits_final.json \
  --out-dir private_assets/split_reports
```

Teacher-specific agreement masks are derived from probabilities, not argmax
NIfTI labels: `ET=p_ET`, `TC=p_NCR+p_ET`, and
`WT=p_NCR+p_ED+p_ET`, followed by thresholding and hierarchy correction. Strict
agreement requires WT Dice ≥0.85, TC Dice ≥0.70, and ET Dice ≥0.50. Both-empty
ET passes; one-sided ET passes only when its present volume is below 5 mm³.

The fused prediction must have mean foreground confidence ≥0.70, normalized
entropy ≤0.35, margin ≥0.20, high-confidence foreground fraction ≥0.60, and
foreground coverage in `[10⁻⁵, 0.30]`. Small WT/TC/ET components use volume
thresholds `5/3/1 mm³` together with mean/max probability thresholds
`0.15/0.30`; ET suppression uses `1 mm³/0.35`. Final masks are reconstructed as
ET→3, TC\ET→1, WT\TC→2, and background→0.

Validate each externally supplied sampling file before training:

```bash
python scripts/data/validate_case_weights.py \
  --weights private_assets/case_weights/case_weights_fold0.json \
  --splits-json private_assets/splits_final.json \
  --fold 0 \
  --manifest "${nnUNet_raw}/Dataset007_Brats26_Goat_MLConsensusPseudo/dataset007_case_manifest.csv"
```

## Five-fold training

```bash
CONFIRM_FULL_TRAINING=YES bash scripts/train_5fold.sh --execute --gpu-id 0
```

The fixed final defaults are 500 epochs, 250/50 training/validation iterations
per epoch, batch size 2, foreground oversampling `0.5`, learning rate `5e-4`,
and gradient clipping `12`. Pseudo-label loss weight is `0.3` before epoch 50,
increases linearly to `0.7` at epoch 150, and remains `0.7`. Sampling is from
one pooled annotated/pseudo dataset with normalized replacement weights and no
fixed source quota. The auxiliary head is bypassed at inference; the dense
adapter remains active. The paper training used an NVIDIA GeForce RTX 5090.
All student parameters are jointly fine-tuned; the auxiliary head and dense
adapter are newly initialized.

## Final five-fold inference

Set `INPUT_DIR` and `OUTPUT_DIR` to user-provided input and output paths before
running this placeholder example.

```bash
python scripts/final_inference.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --plans-json config/nnUNetResEncUNetMPlans_D005Compat.json \
  --dataset-json config/dataset.json \
  --configuration 3d_fullres \
  --checkpoint-fold0 "${RESULT_ROOT}/fold_0/checkpoint_best.pth" \
  --checkpoint-fold1 "${RESULT_ROOT}/fold_1/checkpoint_best.pth" \
  --checkpoint-fold2 "${RESULT_ROOT}/fold_2/checkpoint_best.pth" \
  --checkpoint-fold3 "${RESULT_ROOT}/fold_3/checkpoint_best.pth" \
  --checkpoint-fold4 "${RESULT_ROOT}/fold_4/checkpoint_best.pth" \
  --device cuda
```

This entry point uses Gaussian-weighted sliding windows with patch
`128×160×112`, step size `0.5`, and all 8 combinations of three-axis mirroring.
It equally averages the five fold logits, applies softmax, restores four-class
probabilities to original image space, then performs the same `0.50` regional
mapping and hierarchy correction as pseudo-label generation. It applies no
connected-component, volume, or morphological post-processing.

For a functional model/checkpoint/forward check on available hardware:

```bash
python scripts/verify_environment.py --strict
python scripts/smoke_test_model.py --device cuda \
  --checkpoint "${RESULT_ROOT}/fold_0/checkpoint_best.pth"
```

Use `verify_environment.py --strict --exact-hardware` only when the paper's RTX
5090 model must be required.

## Evaluation

Set `PREDICTION_DIR` to the user-provided prediction path before running this
placeholder example.

```bash
python scripts/evaluate_original_gt.py \
  --prediction-dir "${PREDICTION_DIR}" \
  --dataset005-labels "${D005_LABELS_DIR}" \
  --splits-json "${PRIVATE_SPLITS}" \
  --fold all \
  --output-csv metrics.csv \
  --summary-csv metrics.summary.csv \
  --summary-json metrics.summary.json
```

Dice is computed for ET, TC, and WT. Both-empty prediction/reference regions
are `NaN` and excluded; one-sided empty regions are `0`. Regional valid-case
counts are recorded in CSV/JSON summaries, and case-level means use `nanmean`.

## Official validation-server results

The values below are **reported in Table 5 of the paper** and are not claimed
as outputs recomputed by this public repository.

| Metric | ET | TC | WT | Mean |
|---|---:|---:|---:|---:|
| DSC | 0.797 | 0.823 | 0.881 | 0.834 |
| NSD | 0.547 | 0.501 | 0.481 | 0.510 |

`Acalc = 0.6720`. These NSD values use the validation-server tolerance
`τ=0.5`; the final ranking used `τ=1`.

## Limitations

This repository targets only the final ResEnc-M K=4 system. Authorized private
inputs are required for the BraTS data, ResEnc-M/L teacher checkpoints, exact
Dataset005 split, pseudo-label assets, student checkpoints, and fold-specific
ET-aware sampling weights; because neither the weight files nor their
unpublished generation formula is distributed, a numerically identical
training run cannot be reproduced from scratch without those assets. Tables
1–4 analyses—including labeled-only, all-pseudo, auxiliary-off, K=2/3/5, and
paired statistical comparisons—are outside this repository's scope, as are the
paper's MedNeXt, BI-SegMamba, and heterogeneous-ensemble comparisons. The data
augmenter may run nondeterministically, so bitwise-identical results are not
guaranteed even in the same environment; RTX 4090 smoke testing is functional
verification only and does not guarantee numerical identity with the paper's
RTX 5090 run.

## Citation

```bibtex
@inproceedings{shin2026aistatlab,
  author    = {Heeseung Shin and Changwon Lim},
  title     = {AIstatLab at BraTS-GoAT 2026 Task 3: Quality-Controlled
               Consensus Pseudo-Labeling and Dense Expert Adaptation},
  booktitle = {BraTS-GoAT 2026},
  year      = {2026}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). Third-party
attributions and upstream license information are retained in
`THIRD_PARTY_NOTICES.md` and `third_party/licenses/Apache-2.0.txt`.

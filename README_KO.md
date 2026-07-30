# BraTS 2026 GOAT Dataset007 ResEnc-M SoftMoE K=4 학습 재현

이 저장소는 다음 실험을 GitHub에서 공유하기 위한 코드와 설정입니다.

| 항목 | 값 |
|---|---|
| 데이터셋 | Dataset007_Brats26_Goat_MLConsensusPseudo |
| Backbone/plans | nnUNetResEncUNetMPlans_D005Compat |
| Trainer | nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot |
| Configuration | 3d_fullres |
| Fold | 0, 1, 2, 3, 4 |
| Epoch | 500 |
| SoftMoE K | 4 |
| 초기화 | 동일 fold의 Dataset005 ResEnc-M checkpoint_best.pth |

> BraTS MRI, 정답, pseudo-label, case ID 목록, split, case weight, prediction, checkpoint는 포함하지 않았습니다. 공개 저장소에는 코드와 설정, 집계 결과, SHA-256만 올라갑니다.

## 모델 구성

ResEnc-M encoder의 가장 깊은 feature에 residual SoftMoE adapter를 추가했습니다.

- expert 수: 4
- expert: 경량 3D convolution block
- routing: bottleneck feature의 global average pooling 기반 softmax
- temperature: 1.0
- balance loss weight: 0.001
- adapter scale 초기값: 0.1
- reduction: 4
- ET/TC/WT auxiliary head: 학습에만 사용
- 최종 inference 출력: 기존 4-class nnU-Net segmentation head
- pseudo-label loss: ramp curriculum
- sampling: fold별 ET-aware case weight

EDA cluster를 gate의 정답이나 hard expert label로 사용하지 않습니다.

## 1. 저장소 구조

    config/              dataset.json, ResEnc-M plans, 환경 기본값
    docs/                데이터 준비, 실험 설정과 결과
    private_assets/      Git에 올라가지 않는 private 파일 배치 위치
    provenance/          코드와 checkpoint SHA-256
    requirements/        Python 의존성
    scripts/data/        pseudo-label 및 Dataset007 생성 과정
    scripts/             설치, 검사, 전처리, 학습, 검증, 평가
    src/nnunet_overlays/ custom trainer와 import dependency

## 2. 환경 준비

반드시 새 Conda 환경을 권장합니다.

    conda env create -f environment.yml
    conda activate brats-goat-resencm-softmoe-k4
    export VENV_DIR="$CONDA_PREFIX"
    export PYTHON_BOOTSTRAP_BIN=python

먼저 설치 예정 명령만 확인합니다.

    DRY_RUN=1 bash scripts/bootstrap.sh

확인 후 실제 설치합니다.

    bash scripts/bootstrap.sh

bootstrap은 고정된 nnU-Net commit을 clone하고 custom trainer overlay와 재현 환경을 설치합니다.

## 3. 필요한 private 데이터 구조

자세한 내용은 [DATA_PREPARATION.md](docs/DATA_PREPARATION.md)를 참고하십시오.

Dataset007은 다음 구성을 사용합니다.

- 원본 GT: 1,351 cases
- strict ML-consensus pseudo-label: 983 cases
- 총 supervised training pool: 2,334 cases
- 제외된 imagesUn: 155 cases
- channel 0000: t1n
- channel 0001: t1c
- channel 0002: t2w
- channel 0003: t2f
- label: 0 background, 1 NCR, 2 ED, 3 ET

로컬 경로는 tracked 파일에 쓰지 말고 shell에서 지정합니다.

    export nnUNet_raw=/absolute/private/path/nnUNet_raw
    export nnUNet_preprocessed=/absolute/private/path/nnUNet_preprocessed
    export nnUNet_results=/absolute/private/path/nnUNet_results_K4
    export PRIVATE_MANIFEST=/absolute/private/path/dataset007_case_manifest.csv
    export PRIVATE_SPLITS=/absolute/private/path/splits_final.json
    export CASE_WEIGHT_ROOT=/absolute/private/path/case_weights
    export D005_PRETRAINED_ROOT=/absolute/private/path/d005_resencm_experiment
    export D005_LABELS_DIR=/absolute/private/path/Dataset005/labelsTr

D005_PRETRAINED_ROOT는 다음 구조여야 합니다.

    d005_resencm_experiment/
    ├── fold_0/checkpoint_best.pth
    ├── fold_1/checkpoint_best.pth
    ├── fold_2/checkpoint_best.pth
    ├── fold_3/checkpoint_best.pth
    └── fold_4/checkpoint_best.pth

## 4. 데이터 및 checkpoint 확인

빠른 구조 검사:

    python scripts/verify_assets.py

느리지만 정확한 전체 SHA-256 검사:

    python scripts/verify_assets.py --hash

hash 검사를 통과하기 전에는 재현 학습을 시작하지 않는 것이 안전합니다.

## 5. 전처리

    DRY_RUN=1 bash scripts/prepare_preprocessing.sh
    bash scripts/prepare_preprocessing.sh

이 명령은 Dataset007을 다운로드하거나 pseudo-label을 새로 만들지 않습니다. 준비된 private manifest, split, case weight와 ResEnc-M plans를 검증하고 nnU-Net 전처리를 수행합니다.

## 6. 단일 fold 학습

먼저 fold 0 명령과 초기 checkpoint를 확인합니다.

    DRY_RUN=1 bash scripts/train_fold.sh 0

확인 후 학습합니다.

    bash scripts/train_fold.sh 0

fold 1–4도 반드시 같은 번호의 D005 checkpoint_best.pth로 초기화됩니다. 실제 학습 명령의 핵심은 다음과 같습니다.

    SOFTMOE_NUM_EXPERTS=4     SOFTMOE_MAX_EPOCHS=500     SOFTMOE_INIT_CHECKPOINT=/private/d005/fold_0/checkpoint_best.pth     nnUNetv2_train 7 3d_fullres 0       -tr nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot       -p nnUNetResEncUNetMPlans_D005Compat       -device cuda

## 7. 전체 5-fold

기본 명령은 preflight만 수행합니다.

    bash scripts/train_5fold.sh

실제 다섯 fold를 순차 실행하려면 이중 확인이 필요합니다.

    CONFIRM_FULL_TRAINING=YES bash scripts/train_5fold.sh --execute

## 8. 검증과 ET/TC/WT 평가

기본은 checkpoint_best.pth입니다.

    bash scripts/validate_fold.sh 0 --dry-run
    bash scripts/validate_fold.sh 0

원본 Dataset005 GT에 대해서만 ET/TC/WT Dice를 계산합니다.

    bash scripts/evaluate_fold.sh 0 --dry-run
    bash scripts/evaluate_fold.sh 0

checkpoint_final.pth를 검증하려면 다음을 사용합니다.

    bash scripts/validate_fold.sh 0 --final

## 9. 결과와 제한

5-fold case-weighted original-GT-only 평균:

- ET Dice: 0.8709
- TC Dice: 0.9122
- WT Dice: 0.9280
- ET/TC/WT mean Dice: 0.9037

이는 로컬 validation 결과이며 BraTS 공식 leaderboard 점수가 아닙니다.

중요한 제한으로 fold 0 gate 분석에서는 확률이 거의 균등했지만 dominant expert fraction이 1.0이었고 collapse_warning=True였습니다. 따라서 K=4를 의미 있는 expert specialization의 증거로 해석하면 안 됩니다. 정확한 fold별 값은 [RESULTS.md](docs/RESULTS.md)에 있습니다.

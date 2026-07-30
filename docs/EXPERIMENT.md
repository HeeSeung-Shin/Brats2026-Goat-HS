# Exact ResEnc-M SoftMoE K=4 experiment

## Identity

- Dataset: Dataset007_Brats26_Goat_MLConsensusPseudo
- Dataset CLI argument: 7
- Trainer: nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot
- Plans: nnUNetResEncUNetMPlans_D005Compat
- Configuration: 3d_fullres
- Folds: 0–4
- Device: CUDA, one process/GPU
- Epochs: 500
- Iterations per epoch: 250
- Validation iterations per epoch: 50
- Initial learning rate: 5e-4
- Weight decay: 3e-5
- Batch size: 2
- Patch size: 128 × 160 × 112
- Deep supervision: enabled
- Mirroring axes: 0, 1, 2
- nnUNet_compile: false

## Network modification

The underlying plans instantiate dynamic_network_architectures ResidualEncoderUNet. The trainer attaches BottleneckSoftMoEAdapter to the deepest encoder output and monkey-patches the network forward pass so that only the bottleneck skip is modified before decoding.

K=4 controls:

| Variable | Value |
|---|---:|
| SOFTMOE_NUM_EXPERTS | 4 |
| SOFTMOE_TEMPERATURE | 1.0 |
| SOFTMOE_BALANCE_WEIGHT | 0.001 |
| SOFTMOE_ADAPTER_SCALE_INIT | 0.1 |
| SOFTMOE_REDUCTION | 4 |
| SOFTMOE_MAX_EPOCHS | 500 |

Each expert is GroupNorm → 1×1×1 Conv → GELU → 3×3×3 Conv → GELU → 1×1×1 Conv. Routing uses global average pooling followed by an MLP and softmax. The adapter is residual: z + alpha × weighted_expert_correction.

## Loss and sampling

- Standard nnU-Net main segmentation loss.
- Pseudo-label loss uses the inherited ramp curriculum; fixed-weight fallback value is 0.5.
- ET/TC/WT region weights are ET 0.2, TC 0.2, WT 0.1.
- The train-only auxiliary head consumes highest-resolution four-class logits and predicts sigmoid ET/TC/WT membership.
- Foreground oversampling is 0.5.
- Fold-specific case sampling weights are mandatory for exact reproduction.
- SoftMoE balance loss is multiplied by 0.001.

No hard expert assignment, site/scanner label, EDA cluster label, or source label is fed to the gate.

## Initialization

For fresh fold f:

    SOFTMOE_INIT_CHECKPOINT=/private/d005_resencm/fold_f/checkpoint_best.pth

The loader copies 956 compatible tensors and intentionally leaves 39 new tensors under softmoe_adapter.* and region_aux_head.* initialized by the Dataset007 model.

Resume mode uses nnUNetv2_train --c and sets SOFTMOE_INIT_CHECKPOINT to an empty string.

## Audited command

For fold 0:

    SOFTMOE_NUM_EXPERTS=4     SOFTMOE_TEMPERATURE=1.0     SOFTMOE_BALANCE_WEIGHT=0.001     SOFTMOE_ADAPTER_SCALE_INIT=0.1     SOFTMOE_MAX_EPOCHS=500     SOFTMOE_INIT_CHECKPOINT=/private/d005_resencm/fold_0/checkpoint_best.pth     SOFTMOE_SKIP_ACTUAL_VALIDATION=1     CUDA_VISIBLE_DEVICES=0     nnUNetv2_train 7 3d_fullres 0       -tr nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot       -p nnUNetResEncUNetMPlans_D005Compat       -device cuda

The training launcher skipped nnU-Net's automatic final validation and then ran explicit best-checkpoint validation:

    SOFTMOE_INIT_CHECKPOINT=     SOFTMOE_SKIP_ACTUAL_VALIDATION=0     CUDA_VISIBLE_DEVICES=0     nnUNetv2_train 7 3d_fullres 0       -tr nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot       -p nnUNetResEncUNetMPlans_D005Compat       --val --val_best -device cuda

Repository wrappers reproduce these choices and add fail-fast hash checks.

## Software environment

- Python 3.11.15
- PyTorch 2.11.0+cu130
- CUDA runtime 13.0
- cuDNN 9.19
- nnU-Net 2.6.4, commit f6d221d1b79cd2173650f78f97ecfee273e0cf86
- dynamic-network-architectures 0.4.3
- batchgenerators 0.25.1
- batchgeneratorsv2 0.3.2
- NVIDIA GeForce RTX 5090, driver 580.126.20

BiSegMamba, Mamba, MONAI, and custom CUDA extensions are not part of this experiment.

## Determinism

The package preserves the configuration but does not promise bitwise equality. AMP, CUDA kernels, augmentation workers, scheduling, and dependency/driver differences can change results.

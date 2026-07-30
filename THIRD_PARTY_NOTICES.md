# Third-party notices

This file is an attribution record, not legal advice or a substitute for complete upstream license texts.

## nnU-Net

- Project: nnU-Net
- Upstream: https://github.com/MIC-DKFZ/nnUNet
- Audited commit: f6d221d1b79cd2173650f78f97ecfee273e0cf86
- Audited package version: 2.6.4
- License: Apache License 2.0
- Local license copy: third_party/licenses/Apache-2.0.txt
- Use: base training, preprocessing, validation, and inference framework

Suggested citation: Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature Methods 18, 203–211 (2021), https://doi.org/10.1038/s41592-020-01008-z

## Runtime dependencies

PyTorch, CUDA/cuDNN, dynamic-network-architectures, batchgenerators, NumPy, SciPy, SimpleITK, nibabel, pandas, and transitive packages retain their own licenses and binary redistribution terms. requirements/base.txt records versions but does not grant redistribution rights.

## BraTS 2026 GOAT

BraTS data and challenge materials are not software dependencies covered by the code licenses above. Data access, citation, derived-label handling, and redistribution remain controlled by the applicable challenge/Synapse terms.

This repository does not use or vendor BiSegMamba, SegMamba, Mamba, MONAI, or causal-conv1d.

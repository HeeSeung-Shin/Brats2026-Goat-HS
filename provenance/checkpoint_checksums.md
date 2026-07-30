# Checkpoint provenance

No checkpoint is distributed in this repository. The hashes below identify the exact private files used in the experiment.

## Dataset005 ResEnc-M initialization checkpoint_best.pth

| Fold | Size (bytes) | SHA-256 |
|---:|---:|---|
| 0 | 816,315,151 | 02ef1d4310e61e90a90cbf452ab9ff4a5ce62bd090759da6d1b03dbf1d5f2ad7 |
| 1 | 816,346,639 | 42cf485aadad58b7d7b47d5715b2d2a282da53534ad3ff619dfc7dc88e961a39 |
| 2 | 816,293,135 | a2a3992ad3de73684913713c2deff418fab33008b8248074a41e62b91187a1fb |
| 3 | 816,224,911 | 24372da3da6f65c4972b41158c0b954edc49ce45eb76bd608e94410b840ed9fc |
| 4 | 816,333,647 | d28917c3cbd4220f59a73eb0e48258a4e5c827723e6bff7aac3c4712050f7fe1 |

Source experiment directory name:

    nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres

## Dataset007 SoftMoE K=4 checkpoint_best.pth

| Fold | SHA-256 | checkpoint current_epoch | best EMA pseudo Dice |
|---:|---|---:|---:|
| 0 | 6162434c2368675f9461528fa5b0542580739e4737e80ee452b7ad199fe1b054 | 496 | 0.9023031 |
| 1 | a395c863e0bb04106b0d6364d5a30c378509f9ca0cda11a9cda509727c836b2a | 246 | 0.9031045 |
| 2 | 07c631df3fa0c108036c367e5600dcf4edb13264abdba12274e054687ca97283 | 443 | 0.9075049 |
| 3 | c31cab0495566966914b0a80b735c3626a0db8c24c88bf4f72b62772bbdc3acf | 1 | 0.9055718 |
| 4 | c623d06b680b05f7456fcbfe992a39896746244a4b9ee1c374f7f792b4f6abfc | 332 | 0.8981108 |

Target experiment directory name:

    nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot__nnUNetResEncUNetMPlans_D005Compat__3d_fullres

## Arithmetic five-fold weight average

- Input: the five Dataset007 checkpoint_best.pth files above.
- Method: arithmetic mean of 995 floating network_weights tensors.
- Output size: 1,536,717,183 bytes.
- SHA-256: 262169dd9274ce7ed56808a8ac084369444a40b274d529dd711cebe029cdf3b5

Weight averaging is not probability ensembling and is recorded only as a post-training artifact.

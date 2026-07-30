# Public GitHub checklist

Before publishing:

- [ ] Confirm authorship and choose a license for all custom trainer/data scripts.
- [ ] Replace CITATION.cff.example with confirmed author, title, year, and repository metadata.
- [ ] Confirm current BraTS 2026 GOAT citation and data-use terms.
- [ ] Do not include MRI, labels, pseudo-labels, predictions, case lists, split files, gate CSVs, or checkpoints.
- [ ] Confirm whether derived pseudo-labels and model weights may be redistributed.
- [ ] Keep only aggregate metrics; review them for accidental case identifiers or host paths.
- [ ] Run python scripts/verify_public_repo.py.
- [ ] Run sha256sum -c provenance/source_checksums.sha256.
- [ ] Run git diff --cached --check and inspect git diff --cached --stat.
- [ ] Confirm private_assets/README.md is the only tracked private_assets file.
- [ ] Confirm no file larger than 10 MiB is staged.
- [ ] State the fold-0 gate-collapse warning with any K=4 result.
- [ ] Do not describe local validation as an official leaderboard score.
- [ ] Cite nnU-Net and BraTS as required.

Suggested final review:

    git status --short
    git ls-files private_assets
    git diff --cached --check
    python scripts/verify_public_repo.py
    sha256sum -c provenance/source_checksums.sha256

No credential, personal access token, Synapse token, private URL, username/password, or workstation path should appear in a tracked file.

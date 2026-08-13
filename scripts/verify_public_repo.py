#!/usr/bin/env python3
"""Reject common accidental disclosures before publishing this repository."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


FORBIDDEN_EXTENSIONS = (
    ".nii",
    ".nii.gz",
    ".dcm",
    ".mha",
    ".mhd",
    ".nrrd",
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".pkl",
    ".pickle",
    ".joblib",
)
BACKUP_SUFFIXES = (".orig", ".rej")
SKIP_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a tree for files and strings unsafe for a public GitHub repository.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-size-mib", type=float, default=10.0)
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Scan ignored files too. By default a Git worktree scans tracked and unignored files.",
    )
    return parser.parse_args()


def git_visible_files(root: Path) -> list[Path] | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    paths = [root / os.fsdecode(item) for item in result.stdout.split(b"\0") if item]
    # The index still lists unstaged deletions. Scan the working tree that would
    # actually be published after those deletions are recorded.
    return [path for path in paths if path.exists() or path.is_symlink()]


def tree_files(root: Path) -> list[Path]:
    output: list[Path] = []
    for path in root.rglob("*"):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in SKIP_DIRECTORIES for part in relative_parts):
            continue
        if path.is_file() or path.is_symlink():
            output.append(path)
    return output


def candidate_files(root: Path, all_files: bool) -> list[Path]:
    if not all_files:
        visible = git_visible_files(root)
        if visible is not None:
            return sorted(set(visible))
    return sorted(tree_files(root))


def is_forbidden_extension(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(extension) for extension in FORBIDDEN_EXTENSIONS)


def secret_findings(text: str) -> list[str]:
    findings: list[str] = []
    local_prefix = "/" + "home" + "/" + "user"
    if local_prefix in text:
        findings.append("workstation-local path")

    private_key_prefix = "-----BEGIN "
    if any(
        private_key_prefix + kind + "PRIVATE KEY-----" in text
        for kind in ("", "RSA ", "EC ", "OPENSSH ")
    ):
        findings.append("private key material")

    high_confidence_patterns = (
        ("GitHub token", re.compile(("gh" + "p_") + r"[A-Za-z0-9]{20,}")),
        ("GitHub fine-grained token", re.compile(("github_" + "pat_") + r"[A-Za-z0-9_]{20,}")),
        ("AWS access key", re.compile(("AK" + "IA") + r"[0-9A-Z]{16}")),
        ("AWS temporary key", re.compile(("AS" + "IA") + r"[0-9A-Z]{16}")),
        ("Slack token", re.compile(("xox" + r"[baprs]-") + r"[A-Za-z0-9-]{20,}")),
        ("credential-bearing URL", re.compile(r"https?://[^\s/:@]+:[^\s/@]+@")),
    )
    for label, pattern in high_confidence_patterns:
        if pattern.search(text):
            findings.append(label)

    assignment = re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)\b"
        r"\s*[:=]\s*[\"']([^\"']{12,})[\"']"
    )
    placeholders = ("PLACEHOLDER", "CHANGEME", "REPLACE", "EXAMPLE", "YOUR_", "${", "<", ">")
    for match in assignment.finditer(text):
        value = match.group(1).upper()
        if not any(marker in value for marker in placeholders):
            findings.append("secret-like credential assignment")
            break
    return findings


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def main() -> int:
    args = parse_args()
    root = args.root.resolve(strict=False)
    if not root.is_dir():
        print(f"FAIL: repository root does not exist: {root}", file=sys.stderr)
        return 2
    max_bytes = int(args.max_size_mib * 1024 * 1024)
    failures: list[str] = []
    files = candidate_files(root, args.all_files)

    for path in files:
        try:
            relative = path.relative_to(root)
        except ValueError:
            failures.append(f"path escapes repository root: {path}")
            continue
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
                target.relative_to(root)
            except (OSError, ValueError):
                failures.append(f"external or broken symlink: {relative}")
                continue
        if any(path.name.lower().endswith(suffix) for suffix in BACKUP_SUFFIXES):
            failures.append(f"patch backup/reject file: {relative}")
        if is_forbidden_extension(path.name):
            failures.append(f"medical image, model weight, or binary data extension: {relative}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            failures.append(f"cannot stat {relative}: {exc}")
            continue
        if size > max_bytes:
            failures.append(f"file exceeds {args.max_size_mib:g} MiB ({size} bytes): {relative}")
            continue
        text = read_text(path)
        if text is None:
            continue
        for finding in secret_findings(text):
            failures.append(f"{finding}: {relative}")

    if failures:
        print(f"FAIL: {len(failures)} public-repository issue(s) found:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"PASS: scanned {len(files)} public-visible file(s); no blocked artifacts or secret-like strings found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

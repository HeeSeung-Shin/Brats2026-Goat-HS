from __future__ import annotations

from mlconsensus_common import copy_or_link


def test_copy_or_link_copy_mode(tmp_path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "nested" / "destination.txt"
    source.write_bytes(b"consensus-copy-regression")

    copy_or_link(source, destination, mode="copy")

    assert destination.read_bytes() == source.read_bytes()
    assert not destination.is_symlink()

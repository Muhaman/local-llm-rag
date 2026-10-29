from pathlib import Path

import pytest

from localrag.loaders import is_supported, load_file


def test_loads_markdown_and_text(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Title")
    (tmp_path / "b.TXT").write_text("plain")

    assert load_file(tmp_path / "a.md") == "# Title"
    assert load_file(tmp_path / "b.TXT") == "plain"


def test_unsupported_extensions_are_skipped(tmp_path: Path) -> None:
    assert not is_supported(tmp_path / "pic.png")
    assert is_supported(tmp_path / "doc.PDF")
    with pytest.raises(ValueError):
        load_file(tmp_path / "pic.png")

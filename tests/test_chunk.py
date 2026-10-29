import pytest

from localrag.chunk import chunk_text


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("hello world") == ["hello world"]


def test_blank_text_yields_no_chunks() -> None:
    assert chunk_text("  \n ") == []


def test_long_text_is_split_with_overlap() -> None:
    text = "".join(chr(97 + i % 26) for i in range(5000))

    chunks = chunk_text(text, max_chars=2000, overlap=200)

    assert len(chunks) == 3
    assert all(len(chunk) <= 2000 for chunk in chunks)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert previous[-200:] == following[:200]
    assert chunks[-1].endswith(text[-50:])


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=100, overlap=100)

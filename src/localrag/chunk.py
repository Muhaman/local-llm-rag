def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into fixed-size windows where consecutive chunks share `overlap` characters."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    if not text.strip():
        return []
    step = max_chars - overlap
    chunks: list[str] = []
    start = 0
    while True:
        chunks.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            return chunks
        start += step

import re
import uuid
from collections.abc import Callable
from math import sqrt


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")

    text = text.strip()
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + size].strip())
        if start + size >= len(text):
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        normalized = " ".join(paragraph.split())
        sentences.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        )
    return sentences


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Sentence embeddings must have matching dimensions")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 1.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return 1.0 - max(-1.0, min(1.0, similarity))


def semantic_chunk_sentences(
    sentences: list[str],
    embeddings: list[list[float]],
    size: int = 1000,
    overlap: int = 200,
    breakpoint_percentile: int = 90,
) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError("breakpoint_percentile must be between 0 and 100")
    if len(sentences) != len(embeddings):
        raise ValueError("Each sentence must have exactly one embedding")
    if not sentences:
        return []

    max_core_size = max(1, size - overlap - (1 if overlap else 0))
    units = []
    unit_embeddings = []
    starts_sentence = []
    for sentence, embedding in zip(sentences, embeddings, strict=True):
        pieces = chunk_text(sentence, max_core_size, 0)
        units.extend(pieces)
        unit_embeddings.extend([embedding] * len(pieces))
        starts_sentence.extend([True, *([False] * (len(pieces) - 1))])

    distances = [
        _cosine_distance(unit_embeddings[index - 1], unit_embeddings[index])
        for index in range(1, len(units))
    ]
    threshold = 0.0
    if distances:
        ordered = sorted(distances)
        position = (len(ordered) - 1) * breakpoint_percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        threshold = ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )

    cores = []
    current = ""
    for index, sentence in enumerate(units):
        separator = " " if starts_sentence[index] else ""
        next_text = f"{current}{separator}{sentence}"
        semantic_break = (
            index > 0
            and distances[index - 1] > 1e-8
            and distances[index - 1] >= threshold
        )
        if current and (semantic_break or len(next_text) > max_core_size):
            cores.append(current)
            current = sentence
        else:
            current = next_text
    if current:
        cores.append(current)

    chunks = []
    previous = ""
    for core in cores:
        prefix_limit = min(overlap, max(0, size - len(core) - 1))
        prefix = previous[-prefix_limit:].strip() if prefix_limit else ""
        chunk = f"{prefix} {core}".strip()
        chunks.append(chunk)
        previous = chunk
    return chunks


def extract_pdf(content: bytes) -> list[tuple[int, str]]:
    import pymupdf as fitz

    if b"%PDF-" not in content[:1024]:
        raise ValueError("File is not a PDF")
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            return [
                (number, page.get_text("text").strip()) for number, page in enumerate(document, 1)
            ]
    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError("PDF is invalid or unreadable") from error


def validate_pdf_upload(filename: str, content: bytes, max_bytes: int) -> list[tuple[int, str]]:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")
    if len(content) > max_bytes:
        raise ValueError(f"PDF exceeds the {max_bytes}-byte upload limit")
    return extract_pdf(content)


def build_chunks(
    document_id: str,
    filename: str,
    pages: list[tuple[int, str]],
    size: int = 1000,
    overlap: int = 200,
    chunker: Callable[[str], list[str]] | None = None,
) -> list[dict[str, str | int]]:
    chunks = []
    for page_number, text in pages:
        text_chunks = chunker(text) if chunker else chunk_text(text, size, overlap)
        for index, chunk in enumerate(text_chunks):
            chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page_number}:{index}")
            chunks.append(
                {
                    "id": str(chunk_id),
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "chunk_index": index,
                    "text": chunk,
                }
            )
    return chunks

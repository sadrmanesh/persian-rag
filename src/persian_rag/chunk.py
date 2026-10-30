"""Chunk Persian text by size, sentences, or recursive separators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from persian_rag.tokenize import split_sentence_spans

ChunkStrategy = Literal["fixed", "sentence", "recursive"]

_DEFAULT_SEPARATORS = (
    "\n\n",
    "\n",
    "۔ ",
    ". ",
    "؟ ",
    "! ",
    "؛ ",
    "، ",
    " ",
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous slice of a document."""

    id: str
    text: str
    start: int
    end: int
    index: int
    document_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "index": self.index,
            "document_id": self.document_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chunk:
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("chunk metadata must be an object")
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            start=int(data["start"]),
            end=int(data["end"]),
            index=int(data["index"]),
            document_id=str(data.get("document_id") or ""),
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Controls :func:`chunk_text`.

    ``overlap`` is a character overlap for the fixed and recursive strategies.
    ``sentence_overlap`` is the number of sentences repeated between neighboring
    sentence-based chunks. A single unit longer than ``chunk_size`` is split
    on character boundaries.
    """

    strategy: ChunkStrategy = "recursive"
    chunk_size: int = 800
    overlap: int = 120
    sentence_overlap: int = 1
    separators: tuple[str, ...] = _DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.strategy not in ("fixed", "sentence", "recursive"):
            raise ValueError("strategy must be 'fixed', 'sentence', or 'recursive'")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        if self.sentence_overlap < 0:
            raise ValueError("sentence_overlap must be >= 0")
        if isinstance(self.separators, list):
            object.__setattr__(self, "separators", tuple(self.separators))
        if not self.separators:
            raise ValueError("separators must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["separators"] = list(self.separators)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChunkingConfig:
        known = {item.name for item in fields(cls)}
        kwargs: dict[str, Any] = {key: data[key] for key in known if key in data}
        separators = kwargs.get("separators")
        if isinstance(separators, list):
            kwargs["separators"] = tuple(str(item) for item in separators)
        return cls(**kwargs)


def chunk_text(
    text: str,
    config: ChunkingConfig | None = None,
    *,
    document_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into chunks.

    Offsets refer to ``text`` as passed in. Normalize first when retrieval
    should see the canonical form; this function does not rewrite characters
    except that chunk text is the exact slice ``text[start:end]``.
    """

    options = config or ChunkingConfig()
    if not text or not text.strip():
        return []

    if options.strategy == "fixed":
        spans = _fixed_spans(text, 0, len(text), options.chunk_size, options.overlap)
    elif options.strategy == "sentence":
        spans = _sentence_spans(text, options.chunk_size, options.sentence_overlap)
    else:
        spans = _recursive_spans(text, options.chunk_size, options.overlap, options.separators)

    shared = dict(metadata or {})
    chunks: list[Chunk] = []
    for raw_start, raw_end in spans:
        trimmed = _trim_span(text, raw_start, raw_end)
        if trimmed is None:
            continue
        start, end = trimmed
        piece = text[start:end]
        index = len(chunks)
        prefix = document_id or "chunk"
        chunks.append(
            Chunk(
                id=f"{prefix}:{index}",
                text=piece,
                start=start,
                end=end,
                index=index,
                document_id=document_id,
                metadata=dict(shared),
            )
        )
    return chunks


def _fixed_spans(
    text: str,
    start: int,
    end: int,
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = start
    while position < end:
        stop = min(position + size, end)
        if stop < end:
            snap = text.rfind(" ", position + (size // 2), stop)
            if snap > position:
                stop = snap
        if stop <= position:
            stop = min(position + size, end)
        trimmed = _trim_span(text, position, stop)
        if trimmed is not None:
            spans.append(trimmed)
        if stop >= end:
            break
        nxt = stop - overlap
        if nxt <= position:
            nxt = stop
        position = nxt
    return spans


def _sentence_spans(text: str, size: int, sentence_overlap: int) -> list[tuple[int, int]]:
    sentences = split_sentence_spans(text)
    if not sentences:
        stripped = _trim_span(text, 0, len(text))
        return [stripped] if stripped else []

    chunks: list[tuple[int, int]] = []
    bucket: list[tuple[int, int]] = []
    for span in sentences:
        span_len = span[1] - span[0]
        if span_len > size:
            if bucket:
                chunks.append((bucket[0][0], bucket[-1][1]))
                bucket = []
            chunks.extend(_fixed_spans(text, span[0], span[1], size, overlap=0))
            continue
        if not bucket:
            bucket = [span]
            continue
        if span[1] - bucket[0][0] <= size:
            bucket.append(span)
            continue
        chunks.append((bucket[0][0], bucket[-1][1]))
        tail = bucket[-sentence_overlap:] if sentence_overlap > 0 else []
        bucket = tail if tail and span[1] - tail[0][0] <= size else []
        bucket.append(span)
    if bucket:
        chunks.append((bucket[0][0], bucket[-1][1]))
    return chunks


def _recursive_spans(
    text: str,
    size: int,
    overlap: int,
    separators: tuple[str, ...],
) -> list[tuple[int, int]]:
    start, end = _trim_span(text, 0, len(text)) or (0, 0)
    if start >= end:
        return []
    units = _atomic_units(text, start, end, separators, size)
    if not units:
        return []
    if len(units) == 1:
        return units
    return _merge_spans(units, size, overlap)


def _atomic_units(
    text: str,
    start: int,
    end: int,
    separators: tuple[str, ...],
    size: int,
) -> list[tuple[int, int]]:
    if start >= end or not text[start:end].strip():
        return []
    if end - start <= size:
        trimmed = _trim_span(text, start, end)
        return [trimmed] if trimmed else []

    chosen: str | None = None
    rest: tuple[str, ...] = ()
    segment = text[start:end]
    for index, separator in enumerate(separators):
        if separator and separator in segment:
            chosen = separator
            rest = separators[index + 1 :]
            break
    if chosen is None:
        return _fixed_spans(text, start, end, size, overlap=0)

    children = _split_keeping_separator(text, start, end, chosen)
    if len(children) == 1 and children[0] == (start, end):
        return _atomic_units(text, start, end, rest, size)

    units: list[tuple[int, int]] = []
    for child_start, child_end in children:
        units.extend(_atomic_units(text, child_start, child_end, rest, size))
    return units


def _split_keeping_separator(
    text: str,
    start: int,
    end: int,
    separator: str,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = start
    while position < end:
        found = text.find(separator, position, end)
        if found == -1:
            if text[position:end].strip():
                spans.append((position, end))
            break
        right = found + len(separator)
        if text[position:right].strip():
            spans.append((position, right))
        position = right
    return spans


def _merge_spans(
    units: list[tuple[int, int]],
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    bucket: list[tuple[int, int]] = []

    def span_len(parts: list[tuple[int, int]]) -> int:
        return parts[-1][1] - parts[0][0]

    for unit in units:
        if not bucket:
            bucket = [unit]
            continue
        if span_len([*bucket, unit]) <= size:
            bucket.append(unit)
            continue
        chunks.append((bucket[0][0], bucket[-1][1]))
        tail = _tail_units(bucket, overlap)
        if len(tail) >= len(bucket):
            tail = []
        bucket = [*tail, unit]
        if span_len(bucket) > size:
            bucket = [unit]
    if bucket:
        chunks.append((bucket[0][0], bucket[-1][1]))
    return chunks


def _tail_units(bucket: list[tuple[int, int]], overlap: int) -> list[tuple[int, int]]:
    if overlap <= 0 or not bucket:
        return []
    tail: list[tuple[int, int]] = []
    for unit in reversed(bucket):
        candidate = [unit, *tail]
        if tail and candidate[-1][1] - candidate[0][0] > overlap:
            break
        tail = candidate
    return tail


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end

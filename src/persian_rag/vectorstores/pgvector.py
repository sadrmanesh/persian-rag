"""PostgreSQL + pgvector store adapter.

Install with ``pip install 'persian-rag[pgvector]'`` (psycopg 3). The database
needs the ``vector`` extension; the store runs ``CREATE EXTENSION IF NOT
EXISTS vector`` when ``create_table`` is true and the role is allowed to.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from persian_rag.store import VectorHit, _check_parallel, _metadata_rows, _shared_dimension

PgDistance = Literal["cosine", "l2", "inner_product"]

_INSERT_BATCH = 500
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_OPERATORS: dict[str, tuple[str, str]] = {
    # distance operator, HNSW operator class
    "cosine": ("<=>", "vector_cosine_ops"),
    "l2": ("<->", "vector_l2_ops"),
    "inner_product": ("<#>", "vector_ip_ops"),
}


def vector_literal(vector: Sequence[float]) -> str:
    """Render a vector in pgvector's text format, for example ``[1.0,2.5]``."""

    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def _quote_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid table name {name!r}: use letters, digits, and underscores (max 63)"
        )
    return f'"{name}"'


def _connect(conninfo: str) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(
            "psycopg is not installed. Install it with: pip install 'persian-rag[pgvector]'"
        ) from exc
    return psycopg.connect(conninfo, autocommit=True)


class PgVectorStore:
    """Store chunks in a PostgreSQL table with a ``vector`` column.

    Pass either ``conninfo`` (a libpq connection string) or an open psycopg
    ``connection`` in autocommit mode. Scores are larger-is-closer: cosine
    similarity for ``cosine``, negative distance for ``l2``, and the inner
    product for ``inner_product``.
    """

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        dimension: int,
        connection: Any | None = None,
        table: str = "persian_rag_chunks",
        distance: PgDistance = "cosine",
        create_table: bool = True,
        create_index: bool = False,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if distance not in _OPERATORS:
            raise ValueError("distance must be 'cosine', 'l2', or 'inner_product'")
        if connection is None:
            if conninfo is None:
                raise ValueError("pass conninfo or connection")
            connection = _connect(conninfo)
        self.connection = connection
        self.dimension = dimension
        self.distance = distance
        self.table = table
        self._table_sql = _quote_identifier(table)
        if create_table:
            self.create_schema(create_index=create_index)

    def _execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
        return self.connection.execute(query, params)

    def create_schema(self, *, create_index: bool = False) -> None:
        """Create the extension, the table, and optionally an HNSW index."""

        self._execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._table_sql} ("
            "id TEXT PRIMARY KEY, "
            f"embedding vector({self.dimension}) NOT NULL, "
            "text TEXT NOT NULL, "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb)"
        )
        if create_index:
            _, opclass = _OPERATORS[self.distance]
            index = _quote_identifier(f"{self.table}_embedding_idx"[:63])
            self._execute(
                f"CREATE INDEX IF NOT EXISTS {index} ON {self._table_sql} "
                f"USING hnsw (embedding {opclass})"
            )

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        _check_parallel(ids, vectors, texts, metadatas)
        if not ids:
            return
        width = _shared_dimension(vectors)
        if width != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {width}")
        taken = self._execute(
            f"SELECT id FROM {self._table_sql} WHERE id = ANY(%s)", (list(ids),)
        ).fetchall()
        if taken:
            names = ", ".join(sorted(str(row[0]) for row in taken))
            raise ValueError(f"Duplicate vector id: {names}")
        metas = _metadata_rows(metadatas, len(ids))
        rows = [
            (doc_id, vector_literal(vector), text, json.dumps(meta, ensure_ascii=False))
            for doc_id, vector, text, meta in zip(ids, vectors, texts, metas, strict=True)
        ]
        for start in range(0, len(rows), _INSERT_BATCH):
            batch = rows[start : start + _INSERT_BATCH]
            placeholders = ", ".join(["(%s, %s::vector, %s, %s::jsonb)"] * len(batch))
            params = [value for row in batch for value in row]
            self._execute(
                f"INSERT INTO {self._table_sql} (id, embedding, text, metadata) "
                f"VALUES {placeholders}",
                params,
            )

    def search(self, vector: Sequence[float], k: int = 5) -> list[VectorHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if len(vector) != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {len(vector)}")
        operator, _ = _OPERATORS[self.distance]
        rows = self._execute(
            f"SELECT id, text, metadata, embedding {operator} %s::vector AS distance "
            f"FROM {self._table_sql} ORDER BY distance ASC, id ASC LIMIT %s",
            (vector_literal(vector), k),
        ).fetchall()
        return [
            VectorHit(
                id=str(doc_id),
                score=self._score(float(distance)),
                text=str(text),
                metadata=_decode_metadata(metadata),
            )
            for doc_id, text, metadata, distance in rows
        ]

    def _score(self, distance: float) -> float:
        if self.distance == "cosine":
            return 1.0 - distance
        # l2 distance and pgvector's negative inner product both sort ascending.
        return -distance

    def delete(self, ids: Sequence[str]) -> None:
        """Remove rows by id. Unknown ids are ignored."""

        if ids:
            self._execute(f"DELETE FROM {self._table_sql} WHERE id = ANY(%s)", (list(ids),))

    def __len__(self) -> int:
        row = self._execute(f"SELECT count(*) FROM {self._table_sql}").fetchone()
        return int(row[0]) if row else 0


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value) if isinstance(value, Mapping) else {}

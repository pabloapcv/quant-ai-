"""Passage search over stored documents: BM25 on paragraphs, point in time.

PRD §10.2 asks for hybrid BM25 + vector search; this is the BM25 half (no embedding model is
deployed). Each paragraph is a passage; long minutes paragraphs are split into windows of
about ``WINDOW`` words so one long paragraph cannot dominate. Queries only see documents
published at or before ``as_of``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from atlas.text.documents import Document

WINDOW = 160
K1, B = 1.5, 0.75
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "which",
        "while",
        "also",
        "their",
        "they",
        "been",
        "would",
        "could",
        "than",
        "other",
        "such",
        "these",
        "those",
        "into",
        "over",
        "more",
        "most",
        "some",
        "any",
        "not",
        "but",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Passage:
    doc: Document
    index: int  # passage number within the document
    text: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


def passages(doc: Document) -> list[Passage]:
    out: list[Passage] = []
    for para in doc.paragraphs:
        words = para.split()
        for i in range(0, max(len(words), 1), WINDOW):
            chunk = " ".join(words[i : i + WINDOW])
            if chunk:
                out.append(Passage(doc, len(out), chunk))
    return out


class BM25Index:
    def __init__(self, docs: Sequence[Document]) -> None:
        self.passages = [p for d in docs for p in passages(d)]
        self._tf = [Counter(tokenize(p.text)) for p in self.passages]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.passages)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        as_of: datetime | None = None,
        kind: str | None = None,
        per_doc: int = 2,
    ) -> list[Hit]:
        """Top ``k`` passages; at most ``per_doc`` from one document."""
        terms = tokenize(query)
        scored: list[Hit] = []
        for i, p in enumerate(self.passages):
            if as_of is not None and p.doc.published_at > as_of:
                continue
            if kind and p.doc.kind != kind:
                continue
            tf = self._tf[i]
            norm = K1 * (1 - B + B * self._len[i] / self._avg) if self._avg else K1
            s = sum(
                self._idf.get(t, 0.0) * tf[t] * (K1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if s > 0:
                scored.append(Hit(p, s))
        scored.sort(key=lambda h: -h.score)
        out: list[Hit] = []
        taken: Counter[str] = Counter()
        for h in scored:
            if taken[h.passage.doc.doc_id] < per_doc:
                out.append(h)
                taken[h.passage.doc.doc_id] += 1
            if len(out) == k:
                break
        return out

"""Text documents (PRD §6.6): central-bank statements and minutes, stored point in time.

A document is immutable once stored: ``published_at`` is when it became public (its
knowledge time), so as-of queries never see a statement before its release.
Layout: ``data/text/<source>/<kind>/<doc_id>.json``.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Kind = Literal["statement", "minutes", "press_conference"]
KINDS: tuple[Kind, ...] = ("statement", "minutes", "press_conference")
_SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z])")


class Document(BaseModel):
    doc_id: str  # e.g. "fomc-statement-2025-09-17", "ecb-press_conference-2025-12-18"
    source: str  # "fomc" | "ecb"
    kind: Kind
    title: str
    url: str
    published_at: datetime  # UTC
    meeting_date: date
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


class DocStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "text"

    def path(self, doc: Document) -> Path:
        return self.root / doc.source / doc.kind / f"{doc.doc_id}.json"

    def exists(self, source: str, kind: str, doc_id: str) -> bool:
        return (self.root / source / kind / f"{doc_id}.json").exists()

    def save(self, doc: Document) -> bool:
        """Write ``doc`` unless already stored (documents are immutable). True if new."""
        p = self.path(doc)
        if p.exists():
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(doc.model_dump_json(indent=2))
        tmp.replace(p)
        return True

    def iter(
        self,
        *,
        source: str | None = None,
        kind: str | None = None,
        as_of: datetime | None = None,
    ) -> Iterator[Document]:
        """Documents published at or before ``as_of``, oldest first."""
        kinds = [kind] if kind else list(KINDS)  # derived data (e.g. tone/) sits alongside
        paths = [p for k in kinds for p in self.root.glob(f"{source or '*'}/{k}/*.json")]
        docs = (Document.model_validate_json(p.read_text()) for p in paths)
        selected = [d for d in docs if as_of is None or d.published_at <= as_of]
        yield from sorted(selected, key=lambda d: d.published_at)


def sentences(paragraphs: list[str]) -> list[str]:
    out: list[str] = []
    for p in paragraphs:
        out += [s.strip() for s in _SENTENCE.split(p) if s.strip()]
    return out


def statement_changes(previous: Document, current: Document) -> dict[str, list[str]]:
    """Sentences added to and removed from a statement since the previous one.

    Sentences that only changed a few words are reported as a removed/added pair; the vote
    paragraph is left out (names change every year and carry no policy signal).
    """

    def body(doc: Document) -> list[str]:
        policy = [p for p in doc.paragraphs if not p.startswith(("Voting for", "Voting against"))]
        return sentences(policy)

    old, new = body(previous), body(current)
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            removed += old[i1:i2]
        if op in ("replace", "insert"):
            added += new[j1:j2]
    return {"added": added, "removed": removed}


_RATE = re.compile(
    r"(federal funds rate|key ECB interest rates|interest rates? on the main refinancing|"
    r"deposit facility rate|key interest rates)",
    re.I,
)
_VERB = re.compile(
    r"\b(decided|decision|maintain|keep|kept|leave|left|remain|unchanged|raise|lower|reduce|"
    r"increase|cut)",
    re.I,
)
_GUIDANCE = re.compile(
    r"\b(expects?|intends?|anticipates?|appropriate|would|could|may|likely|ready to|prepared|"
    r"will remain|extended period|in the future)\b",
    re.I,
)
_HOLD = re.compile(r"\b(unchanged|keep|kept|maintain(ed)?|left|leave|remain(ed)?|reaffirm)", re.I)
# "kept rates unchanged, following the increase of 25 basis points in March": the move named
# after these words is the previous meeting's, not today's decision
_PAST = re.compile(r"\b(following|after|since)\b", re.I)
_UP = re.compile(r"\b(raise[sd]?|increase[sd]?|hike)", re.I)
_DOWN = re.compile(r"\b(lower(ed)?|reduce[sd]?|cut|decrease[sd]?)", re.I)


def decision(doc: Document, previous: Document | None = None) -> tuple[str, str] | None:
    """(direction, evidence) of the policy-rate decision a statement announces.

    Direction is ``raise``, ``lower`` or ``hold``. For FOMC statements the stated federal
    funds target is compared with the previous statement's when both are given (exact);
    otherwise the first sentence naming the policy rate with a decision verb is read, skipping
    forward guidance ("expects", "intends", ...). None if neither works.
    """
    if previous is not None and doc.source == "fomc":
        from atlas.text.tone import text_change_bp, text_target

        bp = text_change_bp(doc, previous)
        if bp is not None:
            direction = "raise" if bp > 0 else "lower" if bp < 0 else "hold"
            return direction, f"federal funds target {text_target(doc)}% ({bp:+d}bp)"
    for sent in sentences(doc.paragraphs):
        if not (_RATE.search(sent) and _VERB.search(sent)) or _GUIDANCE.search(sent):
            continue
        today = _PAST.split(sent, maxsplit=1)[0]
        up, down, hold = _UP.search(today), _DOWN.search(today), _HOLD.search(today)
        if up and not down:
            return "raise", sent
        if down and not up:
            return "lower", sent
        if hold:
            return "hold", sent
        # e.g. "we took a number of decisions on key ECB interest rates": read on
    return None

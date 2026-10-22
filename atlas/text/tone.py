"""Central-bank statement tone extraction (PRD §6.6): Claude Haiku reads each statement once.

FOMC and ECB statements, each with its own prompt (the ECB publishes no votes and decides on
three key rates) and its own deterministic check: FOMC against the federal funds target
stated in consecutive statements, ECB against the size stated in the decision sentence.

Output per statement (stored in ``data/text/fomc/tone/<doc_id>.json``, never recomputed):
the policy action and size, a hawkish/dovish stance score from -2 (very dovish) to +2 (very
hawkish) relative to the previous statement's language, the balance-sheet direction,
dissent count and a one-line summary. The action and size are checked deterministically:
each statement states the federal funds target ("... target range for the federal funds
rate at 1-1/2 to 1-3/4 percent"), and the change from the previous statement's stated level
must match. A mismatch is kept but flagged (``consistent``), so a misread decision is
visible downstream.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from atlas.text.documents import DocStore, Document

EXTRACT_MODEL = "claude-haiku-5-5"
MAX_ATTEMPTS = 3  # a statement whose reply never parses is not re-read every day

PROMPT = """You read Federal Open Market Committee statements and extract facts. You get the \
current statement and, when available, the previous one. Reply with one JSON object only:

{"action": "hike" | "cut" | "hold" | "other",
 "change_bp": integer (federal funds target change in basis points; 0 for hold),
 "stance": integer -2..2 (dovish -2 .. hawkish +2: the policy signal of the language and \
the decision, judged against the previous statement),
 "balance_sheet": "expanding" | "shrinking" | "steady" | "not_mentioned",
 "dissents": integer (members voting against),
 "summary": "one sentence on what changed and why it matters"}

Use only the statement text. "other" is for emergency or liquidity statements without a \
rate decision."""

ECB_PROMPT = """You read European Central Bank monetary policy statements and extract facts. \
You get the current statement and, when available, the previous one. Reply with one JSON \
object only:

{"action": "hike" | "cut" | "hold" | "other",
 "change_bp": integer (change in the key ECB interest rates in basis points; 0 for hold),
 "stance": integer -2..2 (dovish -2 .. hawkish +2: the policy signal of the language and \
the decision, judged against the previous statement),
 "balance_sheet": "expanding" | "shrinking" | "steady" | "not_mentioned" (asset purchases \
such as the APP or PEPP),
 "dissents": integer (0: the ECB does not publish votes),
 "summary": "one sentence on what changed and why it matters"}

Use only the statement text. "other" is for statements without a rate decision."""
PROMPTS = {"fomc": PROMPT, "ecb": ECB_PROMPT}
_ECB_BP = re.compile(r"(\d+)\s*basis points?", re.I)
_ECB_PP = re.compile(r"(0[.,]\d+)\s*percentage points?", re.I)

_NUM = r"\d+(?:-\d/\d)?|\d/\d"
_TARGET = re.compile(
    r"target (?:range )?for the federal funds rate[^.]*?\b(?:at|to|of) "
    rf"({_NUM})(?: to ({_NUM}))? percent",
    re.I,
)


class Tone(BaseModel):
    doc_id: str
    action: Literal["hike", "cut", "hold", "other"]
    change_bp: int
    stance: int = Field(ge=-2, le=2)
    balance_sheet: Literal["expanding", "shrinking", "steady", "not_mentioned"]
    dissents: int = Field(ge=0)
    summary: str
    model: str = ""
    text_change_bp: int | None = None  # size found in the text by regex, if any
    consistent: bool = True


def _number(raw: str) -> float:
    whole, _, frac = (
        raw.partition("-") if "-" in raw else ("0", "", raw) if "/" in raw else (raw, "", "")
    )
    value = float(whole)
    if frac:
        num, den = frac.split("/")
        value += int(num) / int(den)
    return value


def text_target(doc: Document) -> float | None:
    """Federal funds target (upper end of the range), percent, as the statement states it."""
    for para in doc.paragraphs:
        m = _TARGET.search(para)
        if m:
            return _number(m.group(2) or m.group(1))
    return None


def text_change_bp(doc: Document, previous: Document | None) -> int | None:
    """Target change implied by this statement's stated level vs the previous statement's."""
    if previous is None:
        return None
    cur, prev = text_target(doc), text_target(previous)
    if cur is None or prev is None:
        return None
    return round((cur - prev) * 100)


def ecb_change_bp(doc: Document) -> int | None:
    """Rate change the ECB decision sentence states (None when it states no size)."""
    from atlas.text.documents import decision

    d = decision(doc)
    if d is None:
        return None
    direction, sentence = d
    if direction == "hold":
        return 0
    m = _ECB_BP.search(sentence)
    size = int(m.group(1)) if m else None
    if size is None and (pm := _ECB_PP.search(sentence)):
        size = round(float(pm.group(1).replace(",", ".")) * 100)
    if size is None:
        return None
    return size if direction == "raise" else -size


def stated_change_bp(doc: Document, previous: Document | None) -> int | None:
    """The deterministic check value for a statement, per bank."""
    if doc.source == "ecb":
        return ecb_change_bp(doc)
    return text_change_bp(doc, previous)


def _consistent(tone: Tone) -> bool:
    found = tone.text_change_bp
    if found is None:
        return True  # nothing to check against
    if tone.action == "hold":
        return found == 0 and tone.change_bp == 0
    if tone.action in ("hike", "cut"):
        return abs(tone.change_bp) == abs(found) and (found > 0) == (tone.action == "hike")
    return True


def parse_tone(text: str, doc_id: str, model: str) -> Tone | None:
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
            return Tone.model_validate({**obj, "doc_id": doc_id, "model": model})
        except (ValueError, ValidationError):
            continue
    return None


def extract(
    client: Any, doc: Document, previous: Document | None, model: str = EXTRACT_MODEL
) -> Tone | None:
    user = f"Current statement ({doc.meeting_date}):\n{doc.text}"
    if previous is not None:
        user = f"Previous statement ({previous.meeting_date}):\n{previous.text}\n\n" + user
    resp = client.beta.messages.create(
        model=model,
        max_tokens=1000,
        system=PROMPTS.get(doc.source, PROMPT),
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    tone = parse_tone(text, doc.doc_id, getattr(resp, "model", model))
    if tone is None:
        return None
    tone.text_change_bp = stated_change_bp(doc, previous)
    tone.consistent = _consistent(tone)
    return tone


def tone_dir(data_dir: Path, bank: str = "fomc") -> Path:
    return data_dir / "text" / bank / "tone"


def load_tones(data_dir: Path, bank: str = "fomc") -> dict[str, Tone]:
    return {
        p.stem: Tone.model_validate_json(p.read_text())
        for p in tone_dir(data_dir, bank).glob("*.json")
    }


def run_tone(
    data_dir: Path,
    client: Any,
    *,
    limit: int | None = None,
    model: str = EXTRACT_MODEL,
    on_done: Callable[[Tone], None] | None = None,
    bank: str = "fomc",
) -> dict[str, int]:
    """Extract tone for every ``bank`` statement that has none yet (oldest first)."""
    docs = list(DocStore(data_dir).iter(source=bank, kind="statement"))
    have = set(load_tones(data_dir, bank))
    folder = tone_dir(data_dir, bank)
    folder.mkdir(parents=True, exist_ok=True)
    attempts_path = folder.parent / "tone_failures.json"  # not in tone/: not a reading
    attempts: dict[str, int] = (
        json.loads(attempts_path.read_text()) if attempts_path.exists() else {}
    )
    done = failed = 0
    for i, doc in enumerate(docs):
        if doc.doc_id in have or attempts.get(doc.doc_id, 0) >= MAX_ATTEMPTS:
            continue
        if limit is not None and done >= limit:
            break
        tone = extract(client, doc, docs[i - 1] if i else None, model)
        if tone is None:
            failed += 1
            attempts[doc.doc_id] = attempts.get(doc.doc_id, 0) + 1
            attempts_path.write_text(json.dumps(attempts, indent=2))
            continue
        (folder / f"{doc.doc_id}.json").write_text(tone.model_dump_json(indent=2))
        done += 1
        if on_done:
            on_done(tone)
    rechecked = recheck(data_dir, bank)
    return {"extracted": done, "failed": failed, "statements": len(docs), "inconsistent": rechecked}


def recheck(data_dir: Path, bank: str = "fomc") -> int:
    """Re-run the deterministic size check on every stored reading; returns inconsistencies."""
    docs = list(DocStore(data_dir).iter(source=bank, kind="statement"))
    folder = tone_dir(data_dir, bank)
    bad = 0
    for i, doc in enumerate(docs):
        path = folder / f"{doc.doc_id}.json"
        if not path.exists():
            continue
        tone = Tone.model_validate_json(path.read_text())
        tone.text_change_bp = stated_change_bp(doc, docs[i - 1] if i else None)
        tone.consistent = _consistent(tone)
        bad += not tone.consistent
        path.write_text(tone.model_dump_json(indent=2))
    return bad

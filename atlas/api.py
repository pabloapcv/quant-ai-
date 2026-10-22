"""Read-only HTTP API (PRD §11.1), a thin layer over the assistant's ``ToolKit``.

The API and the assistant share the same tools, so a number on the API is the same number the
assistant quotes. Auth: when ``ATLAS_API_TOKEN`` is set every ``/v1`` route requires
``Authorization: Bearer <token>``; when it is unset the server is meant for localhost only
(``atlas api serve`` binds 127.0.0.1 by default and refuses other hosts without a token).

The tool cache is rebuilt after ``cache_ttl_s`` so data from the morning pipeline is picked
up without a restart.

``POST /v1/chat`` streams Server-Sent Events: one ``tool`` event per tool call, then an
``answer`` event (text, grounding, disclaimer) or an ``error`` event. Every Claude call goes
through the metered client, so chat shares the daily token budget (``budget`` error when it
is spent) with the CLI, reports and evals.
"""

from __future__ import annotations

import hmac
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from atlas import jsonsafe
from atlas.agent.tools import ToolKit
from atlas.models.scenario import FACTOR_NAMES
from atlas.settings import Settings
from atlas.signals.universes import UNIVERSES

WEB_DIR = Path(__file__).parent / "web"  # the single-page app served at /app/


class SafeJSON(JSONResponse):
    """JSON with NaN/inf mapped to null, like every other Ramon output."""

    def render(self, content: Any) -> bytes:
        return jsonsafe.dumps(content).encode()


class ScenarioRequest(BaseModel):
    universe: str = "ind12"
    shocks: dict[str, float] = Field(
        description=f"factor -> shock; factors: {', '.join(FACTOR_NAMES)}",
        examples=[{"us10y": 0.5, "vix": 5.0}],
    )
    same_regime: bool = False


class ChatTurn(BaseModel):
    question: str = Field(max_length=4000)
    answer: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    style: Literal["analyst", "executive"] = "analyst"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {jsonsafe.dumps(data)}\n\n"


class _Kit:
    """One ToolKit shared by requests, rebuilt when older than the TTL."""

    def __init__(self, settings: Settings, ttl_s: float) -> None:
        self.settings, self.ttl_s = settings, ttl_s
        self._lock = threading.Lock()
        self._kit: ToolKit | None = None
        self._born = 0.0

    def get(self) -> ToolKit:
        with self._lock:
            if self._kit is None or time.monotonic() - self._born > self.ttl_s:
                self._kit = ToolKit(self.settings.data_dir, self.settings.config_dir)
                self._born = time.monotonic()
            return self._kit


def _unwrap(out: dict[str, Any]) -> dict[str, Any]:
    if "error" in out:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


def _universe(universe: str) -> str:
    if universe not in UNIVERSES:
        raise HTTPException(422, f"unknown universe {universe!r}: {', '.join(sorted(UNIVERSES))}")
    return universe


def create_app(
    settings: Settings, *, cache_ttl_s: float = 3600.0, llm_client: Any = None
) -> FastAPI:
    """``llm_client`` overrides the metered Anthropic client (tests)."""
    app = FastAPI(
        title="Ramon API",
        version="0.1.0",
        description="Point-in-time macro regime, views, scenarios and track record. "
        "Research output, not investment advice.",
        default_response_class=SafeJSON,
    )
    kit = _Kit(settings, cache_ttl_s)
    data_dir: Path = settings.data_dir
    token = settings.api_token.get_secret_value() if settings.api_token else ""

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if not token:
            return
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "missing or invalid bearer token")

    guarded: list[Any] = [Depends(auth)]

    def call(fn: Callable[[ToolKit], dict[str, Any]]) -> dict[str, Any]:
        try:
            return _unwrap(fn(kit.get()))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/metrics", dependencies=guarded, response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        from atlas.metrics import collect, render

        text = render(
            collect(data_dir, settings.config_dir, daily_budget=settings.llm_daily_token_budget)
        )
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/app/")

    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="app")

    @app.get("/v1/briefing", dependencies=guarded)
    def briefing() -> dict[str, Any]:
        """Plain-language executive briefing, built by fixed rules from the engine's data."""
        from atlas.briefing import build_briefing

        cutoff = (datetime.now(UTC) - timedelta(days=14)).date().isoformat()
        recent: list[dict[str, Any]] = []
        for f in sorted((data_dir / "alerts").glob("*.jsonl")):
            if f.stem >= cutoff:
                recent += [json.loads(line) for line in f.read_text().splitlines() if line]
        return build_briefing(kit.get(), recent[::-1])

    @app.get("/v1/central-banks", dependencies=guarded)
    def central_banks() -> dict[str, Any]:
        """The latest FOMC and ECB statements: decision, reading and what changed."""
        kit_ = kit.get()
        return {"fomc": kit_.get_fomc_statement(""), "ecb": kit_.get_ecb_statement("")}

    @app.get("/health")
    def health() -> dict[str, Any]:
        runs = sorted((data_dir / "runs").glob("2*.json"))
        last = json.loads(runs[-1].read_text()) if runs else None
        return {
            "status": "ok",
            "last_pipeline_run": last and last.get("started_at"),
            "last_pipeline_ok": last and last.get("ok"),
        }

    @app.get("/v1/universes", dependencies=guarded)
    def universes() -> dict[str, Any]:
        return call(lambda k: k.list_universes())

    @app.get("/v1/regime", dependencies=guarded)
    def regime(as_of: str = "") -> dict[str, Any]:
        return call(lambda k: k.get_regime(as_of))

    @app.get("/v1/views", dependencies=guarded)
    def views(
        universe: str = "ind12", horizon: Annotated[int, Query(ge=1, le=3)] = 3
    ) -> dict[str, Any]:
        u = _universe(universe)
        return call(lambda k: k.get_views(u, horizon))

    @app.get("/v1/views/frozen", dependencies=guarded)
    def frozen_views(universe: str = "ind12", horizon: int = 3) -> dict[str, Any]:
        """Immutable published views (the live track's inputs), newest month first."""
        folder = data_dir / "gold" / "views" / _universe(universe) / f"{horizon}M"
        files = sorted(folder.glob("*.json"), reverse=True)
        return {
            "universe": universe,
            "horizon": f"{horizon}M",
            "months": [{"as_of": f.stem, "views": json.loads(f.read_text())} for f in files[:12]],
        }

    @app.get("/v1/views/{view_id}", dependencies=guarded)
    def view_by_id(view_id: str) -> dict[str, Any]:
        for f in sorted((data_dir / "gold" / "views").glob("*/*/*.json")):
            for v in json.loads(f.read_text()):
                if v.get("view_id") == view_id:
                    out: dict[str, Any] = v
                    return out
        raise HTTPException(404, f"no frozen view {view_id!r}")

    @app.get("/v1/track-record", dependencies=guarded)
    def track_record(universe: str = "ind12", horizon: int = 3) -> dict[str, Any]:
        """Walk-forward record since 1970, plus the live paper track from the last run."""
        u = _universe(universe)
        out = call(lambda k: k.get_track_record(u, horizon))
        runs = sorted((data_dir / "runs").glob("2*.json"))
        out["live_track"] = json.loads(runs[-1].read_text()).get("track") if runs else None
        return out

    @app.get("/v1/portfolio", dependencies=guarded)
    def portfolio(universe: str = "ind12") -> dict[str, Any]:
        u = _universe(universe)
        return call(lambda k: k.get_portfolio(u))

    @app.get("/v1/gates", dependencies=guarded)
    def gates(universe: str = "ind12") -> dict[str, Any]:
        u = _universe(universe)
        return call(lambda k: k.get_signal_gates(u))

    @app.get("/v1/analogs", dependencies=guarded)
    def analogs(
        universe: str = "ind12", as_of: str = "", k: Annotated[int, Query(ge=3, le=20)] = 10
    ) -> dict[str, Any]:
        u = _universe(universe)
        return call(lambda kit_: kit_.find_analogs(u, as_of, k))

    @app.post("/v1/scenario", dependencies=guarded)
    def scenario(req: ScenarioRequest) -> dict[str, Any]:
        u = _universe(req.universe)
        shocks = [{"factor": f, "value": v} for f, v in req.shocks.items()]
        return call(lambda k: k.run_scenario(u, shocks, req.same_regime))

    @app.get("/v1/series/{series_id}", dependencies=guarded)
    def series(
        series_id: str, last_n: Annotated[int, Query(ge=1, le=120)] = 24, as_of: str = ""
    ) -> dict[str, Any]:
        try:
            return call(lambda k: k.get_series(series_id, last_n, as_of))
        except KeyError:
            raise HTTPException(404, f"unknown series {series_id!r}") from None

    @app.get("/v1/catalog", dependencies=guarded)
    def catalog(q: str) -> dict[str, Any]:
        return call(lambda k: k.search_catalog(q))

    @app.get("/v1/alerts", dependencies=guarded)
    def alerts(days: Annotated[int, Query(ge=1, le=90)] = 7) -> dict[str, Any]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
        rows: list[dict[str, Any]] = []
        for f in sorted((data_dir / "alerts").glob("*.jsonl")):
            if f.stem >= cutoff:
                rows += [json.loads(line) for line in f.read_text().splitlines() if line]
        return {"days": days, "alerts": rows[::-1]}

    @app.get("/v1/snapshots/{snapshot_id}", dependencies=guarded)
    def snapshot(snapshot_id: str) -> dict[str, Any]:
        from atlas.pit.snapshot import load_snapshot, verify_snapshot

        if "/" in snapshot_id or ".." in snapshot_id:
            raise HTTPException(422, "invalid snapshot id")
        try:
            snap = load_snapshot(data_dir, snapshot_id)
        except FileNotFoundError:
            raise HTTPException(404, f"no snapshot {snapshot_id!r}") from None
        out: dict[str, Any] = json.loads(snap.model_dump_json())
        out["problems"] = verify_snapshot(data_dir, snap)
        out["n_files"] = len(out.pop("files", []))
        return out

    chat_lock = threading.Lock()  # one chat at a time: the ToolKit is not built for parallel use

    @app.post("/v1/chat", dependencies=guarded)
    def chat(req: ChatRequest) -> StreamingResponse:
        from atlas.agent.assistant import DISCLAIMER, Assistant
        from atlas.agent.usage import BudgetExceeded, make_client

        events: queue.Queue[str | None] = queue.Queue()

        def work() -> None:
            try:
                with chat_lock:
                    client = llm_client or make_client(settings, "chat")
                    assistant = Assistant(
                        client,
                        kit.get(),
                        model=settings.assistant_model,
                        effort=settings.assistant_effort,
                        style=req.style,
                    )
                    ans = assistant.ask(
                        req.question,
                        history=[(t.question, t.answer) for t in req.history],
                        on_tool=lambda c: events.put(
                            _sse(
                                "tool",
                                {"name": c.name, "arguments": c.arguments, "error": c.is_error},
                            )
                        ),
                    )
                events.put(
                    _sse(
                        "answer",
                        {
                            "text": ans.text,
                            "grounded": ans.grounding.grounded,
                            "unsupported": [c.text for c in ans.grounding.unsupported],
                            "model": ans.model,
                            "disclaimer": DISCLAIMER,
                        },
                    )
                )
            except BudgetExceeded as exc:
                events.put(_sse("error", {"kind": "budget", "message": str(exc)}))
            except Exception as exc:
                events.put(_sse("error", {"kind": type(exc).__name__, "message": str(exc)}))
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream() -> Iterator[str]:
            while (item := events.get()) is not None:
                yield item

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/usage", dependencies=guarded)
    def usage_today() -> dict[str, Any]:
        from atlas.agent.usage import UsageLedger

        t = UsageLedger(data_dir).totals(datetime.now(UTC).date())
        return {"today": t, "daily_budget": settings.llm_daily_token_budget}

    return app

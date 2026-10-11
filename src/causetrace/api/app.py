"""The CauseTrace HTTP API.

- ``POST /v1/search``: hybrid search over history passages (``history_only``, so the cutoff
  is capped at 2024-01-01). A reranker failure falls back to RRF and says so.
- ``POST /v1/analyze``: the full agent (with verify) on a stored event (``ev_id``) or a
  free-text packet. Packets pass the input guardrails (400 otherwise); findings without a
  valid citation are dropped and the grounding score is reported. Unavailable LLM (timeout,
  overload, no credentials) or budget reached: 503. Every run is stored.
- ``GET /v1/runs/{run_id}``: a stored run with its full trace.
- ``GET /health``: 200 when the database answers, 503 otherwise.

Run: ``uvicorn causetrace.api.app:create_app --factory`` (or ``causetrace serve``).
"""

import logging
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry.trace import Status, StatusCode
from psycopg_pool import ConnectionPool, PoolTimeout

from causetrace.agent import Agent
from causetrace.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    Finding,
    HealthResponse,
    PacketInput,
    Passage,
    RunResponse,
    SearchRequest,
    SearchResponse,
)
from causetrace.api.runs import get_run, save_run
from causetrace.config import database_url
from causetrace.db import Connection
from causetrace.embeddings import Embedder
from causetrace.guardrails import GuardrailViolation, check_free_text, ground_findings
from causetrace.llm import (
    BudgetExceededError,
    Provider,
    ProviderUnavailableError,
    ReliableProvider,
)
from causetrace.packets import AircraftFacts, IncidentPacket, Injuries, Location, Weather
from causetrace.rerank import Reranker
from causetrace.retrieval import Retriever, effective_before_date
from causetrace.telemetry import setup_tracing, tracer

log = logging.getLogger(__name__)

ProviderFactory = Callable[[], Provider]
RetrieverFactory = Callable[[Connection], Retriever]
DB_TIMEOUT_S = 3.0


class _Locked:
    """Serializes calls to a shared retrieval model across request threads."""

    def __init__(self, inner: Any, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        with self._lock:
            result: list[list[float]] = self._inner.embed(texts, is_query=is_query)
            return result

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        with self._lock:
            result: list[float] = self._inner.score(query, passages)
            return result


def default_retriever_factory() -> RetrieverFactory:
    """Retrievers sharing one embedder and one reranker, loaded on first use."""
    lock = threading.Lock()
    model_lock = threading.Lock()  # one lock for both: they share the accelerator
    models: dict[str, Any] = {}

    def make(conn: Connection) -> Retriever:
        with lock:
            if not models:
                from causetrace.embeddings import default_embedder
                from causetrace.rerank import default_reranker

                models["embedder"] = _Locked(_Lazy(default_embedder), model_lock)
                models["reranker"] = _Locked(_Lazy(default_reranker), model_lock)
        embedder: Embedder = models["embedder"]
        reranker: Reranker = models["reranker"]
        return Retriever(conn, embedder, reranker, rerank_fallback=True)

    return make


class _Lazy:
    """Loads the model on first attribute access, so a missing reranker only fails reranking."""

    def __init__(self, load: Callable[[], Any]) -> None:
        self._load = load
        self._model: Any = None

    def __getattr__(self, name: str) -> Any:
        if self._model is None:
            self._model = self._load()
        return getattr(self._model, name)


def default_provider_factory() -> Provider:
    from causetrace.llm.anthropic_provider import AnthropicProvider

    return ReliableProvider(AnthropicProvider())


def adhoc_packet(packet: PacketInput, run_id: uuid.UUID) -> IncidentPacket:
    """An ``IncidentPacket`` from a free-text request (no stored event behind it)."""
    return IncidentPacket(
        ev_id=f"adhoc-{run_id.hex}",
        ev_date=packet.ev_date,
        ev_time=None,
        ev_timezone=None,
        location=Location(
            city=packet.city,
            state=packet.state,
            country=packet.country,
            latitude=None,
            longitude=None,
            nearest_airport_id=None,
            airport_name=None,
        ),
        weather=Weather(
            basic_conditions=packet.weather,
            light_condition=None,
            ceiling_condition=None,
            ceiling_height_ft=None,
            visibility_sm=None,
            temperature=None,
            dew_point=None,
            wind_direction_deg=None,
            wind_speed_kts=None,
            gust_kts=None,
            altimeter_inhg=None,
        ),
        injuries=Injuries(
            highest=packet.highest_injury, fatal=None, serious=None, minor=None, none=None
        ),
        aircraft=[
            AircraftFacts(
                aircraft_key=1,
                category=packet.aircraft_category,
                make=packet.make,
                model=packet.model,
                series=None,
                phase_of_flight=None,
                damage=packet.damage,
                narr_accp=packet.narrative,
            )
        ],
    )


def create_app(
    *,
    db_url: str | None = None,
    provider_factory: ProviderFactory | None = None,
    retriever_factory: RetrieverFactory | None = None,
    tracing: bool = True,
) -> FastAPI:
    """The application; factories are injectable so tests use fakes (no network)."""
    if tracing:
        setup_tracing()
    pool = ConnectionPool(
        db_url or database_url(),
        min_size=1,
        max_size=10,
        open=False,
        timeout=DB_TIMEOUT_S,
        kwargs={"autocommit": True, "connect_timeout": int(DB_TIMEOUT_S)},
    )
    make_retriever = retriever_factory or default_retriever_factory()
    make_provider = provider_factory or default_provider_factory
    provider_holder: dict[str, Provider] = {}
    provider_lock = threading.Lock()

    def provider() -> Provider:
        with provider_lock:
            if "p" not in provider_holder:
                provider_holder["p"] = make_provider()
            return provider_holder["p"]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool.open(wait=False)  # a database that is down must not stop the API starting
        yield
        pool.close()

    app = FastAPI(title="CauseTrace", version="0.1.0", lifespan=lifespan)

    @contextmanager
    def connection() -> Iterator[Connection]:
        try:
            with pool.connection(timeout=DB_TIMEOUT_S) as conn:
                yield conn
        except (PoolTimeout, psycopg.OperationalError) as err:
            raise HTTPException(503, f"database unavailable: {type(err).__name__}") from err

    @app.middleware("http")
    async def trace_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
            span.set_attribute("http.request.method", request.method)
            response = await call_next(request)
            route = request.scope.get("route")
            if route is not None and hasattr(route, "path"):
                span.update_name(f"{request.method} {route.path}")
                span.set_attribute("http.route", route.path)
            span.set_attribute("http.response.status_code", response.status_code)
            if response.status_code >= 500:
                span.set_status(Status(StatusCode.ERROR))
            return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> JSONResponse:
        try:
            with pool.connection(timeout=DB_TIMEOUT_S) as conn:
                conn.execute("SELECT 1")
        except Exception as err:  # any failure to reach the database is "unavailable"
            body = HealthResponse(status="unavailable", database=f"error: {type(err).__name__}")
            return JSONResponse(body.model_dump(), status_code=503)
        return JSONResponse(HealthResponse(status="ok", database="ok").model_dump())

    @app.post("/v1/search", response_model=SearchResponse)
    def search(body: SearchRequest) -> SearchResponse:
        with connection() as conn:
            result = make_retriever(conn).search(
                body.query, body.top_k, before_date=body.before_date, mode=body.mode
            )
        applied = effective_before_date(body.before_date, True)
        assert applied is not None
        return SearchResponse(
            query=body.query,
            mode=body.mode,
            before_date=applied,
            hits=[
                Passage(
                    passage_id=h.passage_id,
                    event_id=h.event_id,
                    event_date=h.metadata["event_date"],
                    section_type=h.metadata["section_type"],
                    rank=h.rank,
                    score=h.score,
                    content=h.content,
                )
                for h in result.hits
            ],
            fallback=result.fallback,
            timings_ms=result.timings_ms,
        )

    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
        run_id = uuid.uuid4()
        request_json = body.model_dump(mode="json", exclude_none=True)
        if body.packet is not None:
            try:
                check_free_text(body.packet.free_text())
            except GuardrailViolation as err:
                raise HTTPException(400, str(err)) from err
        with connection() as conn:
            llm = provider()
            try:
                agent = Agent(conn, make_retriever(conn), llm)
                if body.ev_id is not None:
                    result = agent.run(body.ev_id)
                else:
                    assert body.packet is not None
                    result = agent.run_packet(adhoc_packet(body.packet, run_id))
            except LookupError as err:
                raise HTTPException(404, str(err)) from err
            except (ProviderUnavailableError, BudgetExceededError) as err:
                message = f"LLM unavailable: {type(err).__name__}"
                log.warning("run %s failed: %s", run_id, err)
                save_run(
                    conn,
                    run_id,
                    ev_id=body.ev_id,
                    status="failed",
                    request=request_json,
                    model=llm.model,
                    error=f"{type(err).__name__}: {err}",
                )
                raise HTTPException(503, message) from err
            target = body.ev_id or result["ev_id"]
            kept, dropped, score = ground_findings(
                result["findings"],
                result["passages"],
                before_date=result["before_date"],
                ev_id=target,
            )
            result = {
                **result,
                "guardrail": {"kept": len(kept), "dropped": dropped, "grounding_score": score},
            }
            abstained = bool(result["abstained"]) or not kept
            save_run(
                conn,
                run_id,
                ev_id=body.ev_id,
                status="completed",
                request=request_json,
                model=llm.model,
                result=result,
                abstained=abstained,
                grounding_score=score,
            )
        return AnalyzeResponse(
            run_id=run_id,
            ev_id=body.ev_id,
            before_date=date.fromisoformat(result["before_date"]),
            abstained=abstained,
            findings=[
                Finding(
                    label=f["label"],
                    short_name=f.get("short_name"),
                    confidence=f.get("confidence"),
                    citations=f["citations"],
                    rationale=f.get("rationale"),
                )
                for f in kept
            ],
            grounding_score=score,
            findings_dropped=dropped,
            passages=result["passages"],
            usage=result["usage"],
        )

    @app.get("/v1/runs/{run_id}", response_model=RunResponse)
    def run(run_id: uuid.UUID) -> RunResponse:
        with connection() as conn:
            stored = get_run(conn, run_id)
        if stored is None:
            raise HTTPException(404, f"no run {run_id}")
        return RunResponse(**stored)

    return app

"""Request and response models of the HTTP API.

Responses carry model predictions, retrieved history passages and run metadata only. No
field is read from ``causetrace.gold``: an analysis never returns the target event's gold
labels.
"""

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causetrace.guardrails import MAX_FIELD_CHARS, MAX_NARRATIVE_CHARS
from causetrace.retrieval import CANDIDATES, Mode

MAX_QUERY_CHARS = MAX_NARRATIVE_CHARS


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchRequest(_Model):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    before_date: date = Field(description="only passages of events strictly before this date")
    top_k: int = Field(default=10, ge=1, le=CANDIDATES)
    mode: Mode = "rrf_rerank"


class Passage(_Model):
    passage_id: str
    event_id: str
    event_date: date
    section_type: str
    rank: int
    score: float
    content: str


class SearchResponse(_Model):
    query: str
    mode: Mode
    before_date: date = Field(description="the cutoff applied (capped at 2024-01-01)")
    hits: list[Passage]
    fallback: str | None = Field(default=None, description="set when reranking failed")
    timings_ms: dict[str, float]


class PacketInput(_Model):
    """A free-text incident: facts only, checked by the input guardrails."""

    ev_date: date
    narrative: str = Field(min_length=1)
    city: str | None = None
    state: str | None = None
    country: str | None = None
    aircraft_category: str | None = None
    make: str | None = None
    model: str | None = None
    damage: str | None = None
    weather: str | None = Field(default=None, description="VMC or IMC")
    highest_injury: str | None = None

    def free_text(self) -> dict[str, str | None]:
        return self.model_dump(exclude={"ev_date"})


class AnalyzeRequest(_Model):
    ev_id: str | None = Field(default=None, max_length=MAX_FIELD_CHARS)
    packet: PacketInput | None = None

    @model_validator(mode="after")
    def _one_input(self) -> "AnalyzeRequest":
        if (self.ev_id is None) == (self.packet is None):
            raise ValueError("give exactly one of ev_id or packet")
        return self


class Finding(_Model):
    label: str
    short_name: str | None
    confidence: float | None
    citations: list[str]
    rationale: str | None


class AnalyzeResponse(_Model):
    run_id: UUID
    ev_id: str | None
    before_date: date
    abstained: bool
    findings: list[Finding]
    grounding_score: float | None = Field(
        description="share of the agent's findings with at least one valid citation"
    )
    findings_dropped: int
    passages: list[dict[str, Any]]
    usage: dict[str, Any]


class RunResponse(_Model):
    run_id: UUID
    created_at: datetime
    ev_id: str | None
    status: Literal["completed", "failed"]
    model: str | None
    abstained: bool | None
    grounding_score: float | None
    request: dict[str, Any]
    result: dict[str, Any] | None = Field(description="the agent result with its full trace")
    error: str | None


class HealthResponse(_Model):
    status: Literal["ok", "unavailable"]
    database: str

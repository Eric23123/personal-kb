"""Validated contracts for Personal KB query planning and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from ..core.openviking_backend import PERSONAL_NAMESPACE
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from scripts.core.openviking_backend import PERSONAL_NAMESPACE

SUPPORTED_INTENTS = frozenset(
    {
        "document_lookup",
        "personal_memory",
        "temporal_event",
        "entity_relationship",
        "formula_exact_term",
        "multi_source_synthesis",
    }
)
SUPPORTED_BACKENDS = frozenset({"openviking", "hindsight"})
SUPPORTED_MODES = frozenset({"dense", "hybrid", "hybrid_rerank"})
FILTER_KEYS = frozenset({"course", "lecture", "source_type", "source_scope", "semester", "date"})
SOURCE_TYPES = frozenset({"lecture", "textbook", "homework", "exam", "diagram", "transcript", "source"})
SOURCE_SCOPES = frozenset({"course", "assessment", "all"})


@dataclass(frozen=True)
class QueryFilters:
    course: str | None = None
    lecture: int | None = None
    source_type: str | None = None
    source_scope: str | None = None
    semester: str | None = None
    date: str | None = None

    def validate(self) -> None:
        if self.lecture is not None and not 1 <= self.lecture <= 1000:
            raise ValueError("lecture must be between 1 and 1000")
        if self.source_type is not None and self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unsupported source_type: {self.source_type}")
        if self.source_scope is not None and self.source_scope not in SOURCE_SCOPES:
            raise ValueError(f"unsupported source_scope: {self.source_scope}")
        for name in ("course", "semester", "date"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} filter must be non-empty text")

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class PlannedQuery:
    text: str
    original_query: str
    trace_id: str
    role: str = "retrieval"
    backend: str | None = None

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("planned query text cannot be empty")
        if not self.original_query.strip():
            raise ValueError("planned query original_query cannot be empty")
        if self.backend is not None and self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported planned-query backend: {self.backend}")


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    intent: str
    backends: tuple[str, ...]
    queries: tuple[PlannedQuery, ...]
    filters: QueryFilters
    retrieval_mode: str
    limit: int
    trace_id: str
    planner_model: str | None = None
    planner_provider: str | None = None
    status: str = "planned"

    def validate(self) -> None:
        if not self.original_query.strip():
            raise ValueError("original query cannot be empty")
        if self.intent not in SUPPORTED_INTENTS:
            raise ValueError(f"unsupported intent: {self.intent}")
        if not self.backends or not set(self.backends) <= SUPPORTED_BACKENDS:
            raise ValueError("backends must contain only supported backend names")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("duplicate backends are not allowed")
        if not self.queries or len(self.queries) > 4:
            raise ValueError("plan must contain between 1 and 4 queries")
        if self.retrieval_mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported retrieval mode: {self.retrieval_mode}")
        if not 1 <= self.limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        if not self.trace_id.strip():
            raise ValueError("trace_id cannot be empty")
        self.filters.validate()
        for query in self.queries:
            query.validate()
            if query.original_query != self.original_query or query.trace_id != self.trace_id:
                raise ValueError("planned query provenance does not match plan")
        if self.intent in {"personal_memory", "temporal_event", "entity_relationship"} and self.backends != ("hindsight",):
            raise ValueError("memory intents must route only to Hindsight")
        if self.intent in {"document_lookup", "formula_exact_term"} and "openviking" not in self.backends:
            raise ValueError("document/formula intents require OpenViking")

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "backends": list(self.backends),
            "queries": [asdict(query) for query in self.queries],
            "filters": self.filters.to_dict(),
            "retrieval_mode": self.retrieval_mode,
            "limit": self.limit,
            "trace_id": self.trace_id,
            "planner_model": self.planner_model,
            "planner_provider": self.planner_provider,
            "status": self.status,
            "namespace": PERSONAL_NAMESPACE,
        }


@dataclass
class QueryTrace:
    original_query: str
    trace_id: str
    status: str = "planned"
    planner_attempts: int = 0
    fallback_used: bool = False
    warning: str | None = None
    errors: list[str] = field(default_factory=list)
    execution: list[dict[str, Any]] = field(default_factory=list)
    source_uris: list[str] = field(default_factory=list)

    def add_error(self, error: Exception | str) -> None:
        message = str(error).replace("\n", " ")[:500]
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningResult:
    plan: QueryPlan
    trace: QueryTrace


__all__ = [
    "PERSONAL_NAMESPACE",
    "FILTER_KEYS",
    "PlannedQuery",
    "PlanningResult",
    "QueryFilters",
    "QueryPlan",
    "QueryTrace",
    "SOURCE_TYPES",
    "SOURCE_SCOPES",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_INTENTS",
    "SUPPORTED_MODES",
]

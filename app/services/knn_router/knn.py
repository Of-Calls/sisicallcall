"""Tenant-scoped KNN router backed by persisted knn_intents rows.

This module is the seventh bootstrap/runtime stage for intent routing. It
loads tenant-specific KNN examples from the database, builds an in-memory
index, runs cosine-similarity search, and returns both leaf/coarse intent
signals with a deterministic confidence heuristic.

It does not mutate graph state, call the fallback LLM, or persist bootstrap
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.services.knn_router.base import BaseKNNRouterService
from app.utils.logger import get_logger

logger = get_logger(__name__)

LOAD_SQL = """
SELECT
    intent_label,
    example_text,
    embedding,
    updated_at
FROM knn_intents
WHERE tenant_id = $1
ORDER BY updated_at ASC, intent_label ASC, example_text ASC
""".strip()

KNOWN_BRANCH_INTENTS = frozenset(
    {
        "intent_faq",
        "intent_task",
        "intent_auth",
        "intent_escalation",
    }
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.85


class AsyncDBFetchProtocol(Protocol):
    async def fetch(self, query: str, *args: Any) -> list[Any]:
        ...


@dataclass(frozen=True)
class KNNIntentRecord:
    intent_label: str
    example_text: str
    embedding: list[float]
    updated_at: str = ""


@dataclass(frozen=True)
class KNNNeighbor:
    intent_label: str
    example_text: str
    score: float
    branch_intent: str


@dataclass(frozen=True)
class KNNPredictionResult:
    tenant_id: str
    top_neighbors: list[KNNNeighbor] = field(default_factory=list)
    top_intent_label: str = ""
    top_branch_intent: str = ""
    confidence: float = 0.0
    is_direct: bool = False
    needs_fallback: bool = True
    record_count: int = 0
    top_score: float = 0.0
    margin_score: float = 0.0
    branch_consistency: float = 0.0


@dataclass(frozen=True)
class TenantKNNIndex:
    tenant_id: str
    embeddings: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    example_texts: list[str] = field(default_factory=list)
    branch_intents: list[str] = field(default_factory=list)
    updated_at_max: str = ""
    record_count: int = 0
    embedding_dimension: int = 0


class KNNRouterService(BaseKNNRouterService):
    def __init__(
        self,
        db: AsyncDBFetchProtocol,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._db = db
        self._confidence_threshold = confidence_threshold
        self._tenant_indexes: dict[str, TenantKNNIndex] = {}

    async def classify(
        self,
        embedding: list[float],
        tenant_id: str,
    ) -> tuple[str | None, float]:
        prediction = await self.predict(
            tenant_id=tenant_id,
            query_embedding=embedding,
            top_k=5,
            confidence_threshold=self._confidence_threshold,
        )
        if prediction.top_intent_label:
            return prediction.top_intent_label, prediction.confidence
        return None, prediction.confidence

    async def load_tenant_intents(
        self,
        tenant_id: str,
    ) -> list[KNNIntentRecord]:
        self._validate_tenant_id(tenant_id)
        rows = await self._db.fetch(LOAD_SQL, tenant_id)
        records: list[KNNIntentRecord] = []

        for row in rows:
            intent_label = self._row_value(row, "intent_label", "").strip()
            example_text = self._row_value(row, "example_text", "").strip()
            raw_embedding = self._row_value(row, "embedding", [])
            embedding = self._coerce_embedding(raw_embedding)
            updated_at = self._coerce_updated_at(self._row_value(row, "updated_at", ""))

            records.append(
                KNNIntentRecord(
                    intent_label=intent_label,
                    example_text=example_text,
                    embedding=embedding,
                    updated_at=updated_at,
                )
            )

        logger.info(
            "loaded knn tenant intents tenant_id=%s record_count=%d",
            tenant_id,
            len(records),
        )
        return records

    def build_tenant_index(
        self,
        tenant_id: str,
        records: list[KNNIntentRecord],
    ) -> TenantKNNIndex:
        self._validate_tenant_id(tenant_id)

        embeddings: list[list[float]] = []
        labels: list[str] = []
        example_texts: list[str] = []
        branch_intents: list[str] = []
        updated_at_values: list[str] = []

        expected_dimension = 0
        for record in records:
            if not record.intent_label.strip():
                logger.warning(
                    "skipping knn record with blank intent_label tenant_id=%s example_text=%s",
                    tenant_id,
                    record.example_text,
                )
                continue
            if not record.example_text.strip():
                logger.warning(
                    "skipping knn record with blank example_text tenant_id=%s intent_label=%s",
                    tenant_id,
                    record.intent_label,
                )
                continue
            if not record.embedding:
                logger.warning(
                    "skipping knn record with empty embedding tenant_id=%s intent_label=%s example_text=%s",
                    tenant_id,
                    record.intent_label,
                    record.example_text,
                )
                continue

            embedding_dimension = len(record.embedding)
            if expected_dimension == 0:
                expected_dimension = embedding_dimension
            elif embedding_dimension != expected_dimension:
                logger.warning(
                    "skipping knn record with mismatched embedding dimension tenant_id=%s intent_label=%s expected=%d actual=%d",
                    tenant_id,
                    record.intent_label,
                    expected_dimension,
                    embedding_dimension,
                )
                continue

            embeddings.append(list(record.embedding))
            labels.append(record.intent_label)
            example_texts.append(record.example_text)
            branch_intents.append(self.get_branch_intent_from_leaf(record.intent_label))
            if record.updated_at:
                updated_at_values.append(record.updated_at)

        return TenantKNNIndex(
            tenant_id=tenant_id,
            embeddings=embeddings,
            labels=labels,
            example_texts=example_texts,
            branch_intents=branch_intents,
            updated_at_max=max(updated_at_values, default=""),
            record_count=len(labels),
            embedding_dimension=expected_dimension,
        )

    async def ensure_tenant_index(
        self,
        tenant_id: str,
    ) -> TenantKNNIndex:
        self._validate_tenant_id(tenant_id)
        cached = self._tenant_indexes.get(tenant_id)
        if cached is not None:
            return cached

        records = await self.load_tenant_intents(tenant_id)
        index = self.build_tenant_index(tenant_id, records)
        self._tenant_indexes[tenant_id] = index
        return index

    async def predict(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        confidence_threshold: float | None = None,
    ) -> KNNPredictionResult:
        self._validate_tenant_id(tenant_id)
        self._validate_query_embedding(query_embedding)

        index = await self.ensure_tenant_index(tenant_id)
        if index.record_count == 0:
            return self._empty_prediction_result(tenant_id, record_count=0)

        if len(query_embedding) != index.embedding_dimension:
            raise ValueError(
                "query embedding dimension must match tenant index dimension"
            )

        scored_neighbors = [
            KNNNeighbor(
                intent_label=index.labels[position],
                example_text=index.example_texts[position],
                score=self._cosine_similarity(query_embedding, embedding),
                branch_intent=index.branch_intents[position],
            )
            for position, embedding in enumerate(index.embeddings)
        ]
        scored_neighbors.sort(
            key=lambda neighbor: (
                -neighbor.score,
                neighbor.intent_label,
                neighbor.example_text,
            )
        )

        limited_neighbors = scored_neighbors[: max(top_k, 0)]
        if not limited_neighbors:
            return self._empty_prediction_result(
                tenant_id,
                record_count=index.record_count,
            )

        top_neighbor = limited_neighbors[0]
        top_score = self._normalize_similarity_to_unit_interval(top_neighbor.score)
        competing_branch_score = self._best_competing_branch_score(
            top_neighbor.branch_intent,
            limited_neighbors,
        )
        margin_score = max(top_score - competing_branch_score, 0.0)
        branch_consistency = self._branch_consistency(
            top_neighbor.branch_intent,
            limited_neighbors,
        )
        confidence = self._clamp_confidence(
            (0.6 * top_score)
            + (0.3 * margin_score)
            + (0.1 * branch_consistency)
        )

        effective_threshold = (
            self._confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        is_direct = confidence >= effective_threshold and bool(top_neighbor.intent_label)

        return KNNPredictionResult(
            tenant_id=tenant_id,
            top_neighbors=limited_neighbors,
            top_intent_label=top_neighbor.intent_label,
            top_branch_intent=top_neighbor.branch_intent,
            confidence=confidence,
            is_direct=is_direct,
            needs_fallback=not is_direct,
            record_count=index.record_count,
            top_score=top_score,
            margin_score=margin_score,
            branch_consistency=branch_consistency,
        )

    def invalidate_tenant_index(self, tenant_id: str) -> None:
        self._validate_tenant_id(tenant_id)
        self._tenant_indexes.pop(tenant_id, None)

    def invalidate_all(self) -> None:
        self._tenant_indexes.clear()

    def get_branch_intent_from_leaf(self, intent_label: str) -> str:
        normalized = intent_label.strip()
        for known_branch in KNOWN_BRANCH_INTENTS:
            if normalized == known_branch or normalized.startswith(f"{known_branch}_"):
                return known_branch

        logger.debug(
            "unknown knn leaf intent branch fallback intent_label=%s fallback=intent_faq",
            intent_label,
        )
        return "intent_faq"

    def _branch_consistency(
        self,
        top_branch_intent: str,
        neighbors: list[KNNNeighbor],
    ) -> float:
        if not neighbors:
            return 0.0
        same_branch_count = sum(
            1 for neighbor in neighbors if neighbor.branch_intent == top_branch_intent
        )
        return same_branch_count / len(neighbors)

    def _best_competing_branch_score(
        self,
        top_branch_intent: str,
        neighbors: list[KNNNeighbor],
    ) -> float:
        competing_scores = [
            self._normalize_similarity_to_unit_interval(neighbor.score)
            for neighbor in neighbors
            if neighbor.branch_intent != top_branch_intent
        ]
        return max(competing_scores, default=0.0)

    def _normalize_similarity_to_unit_interval(self, score: float) -> float:
        return self._clamp_confidence((score + 1.0) / 2.0)

    def _clamp_confidence(self, score: float) -> float:
        return max(0.0, min(score, 1.0))

    def _cosine_similarity(
        self,
        left: list[float],
        right: list[float],
    ) -> float:
        dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
        norm_left = math.sqrt(sum(value * value for value in left))
        norm_right = math.sqrt(sum(value * value for value in right))
        if norm_left == 0.0 or norm_right == 0.0:
            return 0.0
        return dot / (norm_left * norm_right)

    def _validate_tenant_id(self, tenant_id: str) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must not be blank")

    def _validate_query_embedding(self, query_embedding: list[float]) -> None:
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")

    def _empty_prediction_result(
        self,
        tenant_id: str,
        record_count: int,
    ) -> KNNPredictionResult:
        return KNNPredictionResult(
            tenant_id=tenant_id,
            top_neighbors=[],
            top_intent_label="",
            top_branch_intent="",
            confidence=0.0,
            is_direct=False,
            needs_fallback=True,
            record_count=record_count,
            top_score=0.0,
            margin_score=0.0,
            branch_consistency=0.0,
        )

    def _row_value(
        self,
        row: Any,
        key: str,
        default: Any,
    ) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        if hasattr(row, key):
            return getattr(row, key)
        try:
            return row[key]
        except Exception:
            return default

    def _coerce_embedding(self, raw_embedding: Any) -> list[float]:
        if not isinstance(raw_embedding, (list, tuple)):
            return []
        try:
            return [float(value) for value in raw_embedding]
        except (TypeError, ValueError):
            return []

    def _coerce_updated_at(self, raw_updated_at: Any) -> str:
        if raw_updated_at is None:
            return ""
        if hasattr(raw_updated_at, "isoformat"):
            return str(raw_updated_at.isoformat())
        return str(raw_updated_at)

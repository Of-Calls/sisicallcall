from __future__ import annotations

from typing import Any

from app.agents.conversational.state import CallState
from app.services.knn_router.knn import KNNPredictionResult, KNNRouterService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class _AsyncPGKNNDBAdapter:
    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        import asyncpg

        conn = await asyncpg.connect(settings.database_url)
        try:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
        finally:
            await conn.close()


_knn_router_service = KNNRouterService(
    _AsyncPGKNNDBAdapter(),
    confidence_threshold=settings.knn_confidence_threshold,
)


async def knn_router_node(state: CallState) -> dict:
    tenant_id = state.get("tenant_id") or ""
    query_embedding = state.get("query_embedding") or []

    if not query_embedding:
        logger.info(
            "knn routing skipped tenant_id=%s knn_intent=%s knn_confidence=%.4f primary_intent=%s routing_reason=%s",
            tenant_id,
            "",
            0.0,
            None,
            "missing_query_embedding",
        )
        return _fallback_result("missing_query_embedding")

    try:
        prediction = await _knn_router_service.predict(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=5,
        )
    except Exception as exc:
        logger.exception(
            "knn routing failed tenant_id=%s routing_reason=%s error=%s",
            tenant_id,
            "knn_error",
            exc,
        )
        return _fallback_result("knn_error")

    primary_intent = prediction.top_branch_intent if prediction.is_direct else None
    routing_reason = "knn_direct" if prediction.is_direct else "knn_fallback"
    knn_top_k = _serialize_top_neighbors(prediction)

    logger.info(
        "knn routing completed tenant_id=%s knn_intent=%s knn_confidence=%.4f primary_intent=%s routing_reason=%s",
        tenant_id,
        prediction.top_intent_label,
        prediction.confidence,
        primary_intent,
        routing_reason,
    )
    return {
        "knn_intent": prediction.top_intent_label,
        "knn_confidence": prediction.confidence,
        "primary_intent": primary_intent,
        "routing_reason": routing_reason,
        "knn_top_k": knn_top_k,
    }


def _serialize_top_neighbors(prediction: KNNPredictionResult) -> list[dict[str, Any]]:
    return [
        {
            "intent_label": neighbor.intent_label,
            "branch_intent": neighbor.branch_intent,
            "score": float(neighbor.score),
            "example_text": neighbor.example_text,
        }
        for neighbor in prediction.top_neighbors
    ]


def _fallback_result(routing_reason: str) -> dict:
    return {
        "knn_intent": "",
        "knn_confidence": 0.0,
        "primary_intent": None,
        "routing_reason": routing_reason,
        "knn_top_k": [],
    }

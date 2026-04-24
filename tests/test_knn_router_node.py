import json

import pytest

from app.agents.conversational.nodes.knn_router_node import knn_router_node as knn_mod
from app.services.knn_router.knn import KNNNeighbor, KNNPredictionResult


class FakeKNNRouterService:
    def __init__(self, prediction: KNNPredictionResult | None = None, error: Exception | None = None) -> None:
        self.prediction = prediction
        self.error = error
        self.calls: list[tuple[str, list[float], int]] = []

    async def predict(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> KNNPredictionResult:
        self.calls.append((tenant_id, list(query_embedding), top_k))
        if self.error is not None:
            raise self.error
        assert self.prediction is not None
        return self.prediction


def _state(**overrides) -> dict:
    base = {
        "call_id": "call-knn-1",
        "tenant_id": "tenant-a",
        "normalized_text": "book my appointment",
        "query_embedding": [0.8, 0.2],
        "session_view": {"turn_count": 1},
    }
    base.update(overrides)
    return base


def _prediction(
    *,
    top_intent_label: str,
    top_branch_intent: str,
    confidence: float,
    is_direct: bool,
) -> KNNPredictionResult:
    return KNNPredictionResult(
        tenant_id="tenant-a",
        top_neighbors=[
            KNNNeighbor(
                intent_label=top_intent_label,
                example_text="book appointment",
                score=0.97,
                branch_intent=top_branch_intent,
            ),
            KNNNeighbor(
                intent_label="intent_faq_business_hours_outpatient",
                example_text="clinic hours",
                score=0.61,
                branch_intent="intent_faq",
            ),
        ],
        top_intent_label=top_intent_label,
        top_branch_intent=top_branch_intent,
        confidence=confidence,
        is_direct=is_direct,
        needs_fallback=not is_direct,
        record_count=2,
        top_score=confidence,
        margin_score=0.1,
        branch_consistency=0.5,
    )


@pytest.mark.asyncio
async def test_knn_router_node_sets_leaf_confidence_and_primary_for_direct(monkeypatch):
    fake_service = FakeKNNRouterService(
        prediction=_prediction(
            top_intent_label="intent_task_booking_outpatient",
            top_branch_intent="intent_task",
            confidence=0.93,
            is_direct=True,
        )
    )
    monkeypatch.setattr(knn_mod, "_knn_router_service", fake_service)

    result = await knn_mod.knn_router_node(_state())

    assert result["knn_intent"] == "intent_task_booking_outpatient"
    assert result["knn_confidence"] == pytest.approx(0.93)
    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_direct"
    assert fake_service.calls == [("tenant-a", [0.8, 0.2], 5)]


@pytest.mark.asyncio
async def test_knn_router_node_sets_fallback_state_when_prediction_is_low_confidence(monkeypatch):
    fake_service = FakeKNNRouterService(
        prediction=_prediction(
            top_intent_label="intent_faq_business_hours_outpatient",
            top_branch_intent="intent_faq",
            confidence=0.41,
            is_direct=False,
        )
    )
    monkeypatch.setattr(knn_mod, "_knn_router_service", fake_service)

    result = await knn_mod.knn_router_node(_state())

    assert result["knn_intent"] == "intent_faq_business_hours_outpatient"
    assert result["knn_confidence"] == pytest.approx(0.41)
    assert result["primary_intent"] is None
    assert result["routing_reason"] == "knn_fallback"


@pytest.mark.asyncio
async def test_knn_router_node_safely_falls_back_when_query_embedding_is_missing(monkeypatch):
    fake_service = FakeKNNRouterService(
        prediction=_prediction(
            top_intent_label="intent_task_booking_outpatient",
            top_branch_intent="intent_task",
            confidence=0.93,
            is_direct=True,
        )
    )
    monkeypatch.setattr(knn_mod, "_knn_router_service", fake_service)

    result = await knn_mod.knn_router_node(_state(query_embedding=[]))

    assert result == {
        "knn_intent": "",
        "knn_confidence": 0.0,
        "primary_intent": None,
        "routing_reason": "missing_query_embedding",
        "knn_top_k": [],
    }
    assert fake_service.calls == []


@pytest.mark.asyncio
async def test_knn_router_node_safely_falls_back_when_predict_raises(monkeypatch):
    fake_service = FakeKNNRouterService(error=RuntimeError("db down"))
    monkeypatch.setattr(knn_mod, "_knn_router_service", fake_service)

    result = await knn_mod.knn_router_node(_state())

    assert result == {
        "knn_intent": "",
        "knn_confidence": 0.0,
        "primary_intent": None,
        "routing_reason": "knn_error",
        "knn_top_k": [],
    }


@pytest.mark.asyncio
async def test_knn_router_node_serializes_top_k_for_llm_fallback(monkeypatch):
    fake_service = FakeKNNRouterService(
        prediction=_prediction(
            top_intent_label="intent_task_booking_outpatient",
            top_branch_intent="intent_task",
            confidence=0.93,
            is_direct=True,
        )
    )
    monkeypatch.setattr(knn_mod, "_knn_router_service", fake_service)

    result = await knn_mod.knn_router_node(_state())

    assert result["knn_top_k"] == [
        {
            "intent_label": "intent_task_booking_outpatient",
            "branch_intent": "intent_task",
            "score": pytest.approx(0.97),
            "example_text": "book appointment",
        },
        {
            "intent_label": "intent_faq_business_hours_outpatient",
            "branch_intent": "intent_faq",
            "score": pytest.approx(0.61),
            "example_text": "clinic hours",
        },
    ]
    assert json.loads(json.dumps(result["knn_top_k"]))[0]["intent_label"] == "intent_task_booking_outpatient"


def test_route_after_knn_goes_directly_to_branch_when_primary_intent_exists():
    from app.agents.conversational import graph as graph_mod

    assert graph_mod.route_after_knn({"primary_intent": "intent_faq"}) == "faq"
    assert graph_mod.route_after_knn({"primary_intent": "intent_task"}) == "task"


def test_route_after_knn_falls_back_to_intent_router_llm_when_primary_intent_missing():
    from app.agents.conversational import graph as graph_mod

    assert graph_mod.route_after_knn({"primary_intent": None}) == "intent_router_llm"
    assert graph_mod.route_after_knn({"primary_intent": ""}) == "intent_router_llm"
    assert graph_mod.route_after_knn({"primary_intent": "intent_unknown"}) == "intent_router_llm"

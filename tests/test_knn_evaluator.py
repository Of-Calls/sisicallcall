import pytest

from app.services.knn_router.knn import KNNPredictionResult
from app.services.knn_router.knn_evaluator import (
    KNNEvaluationCase,
    evaluate_cases,
    sweep_thresholds,
)


class FakeRouter:
    def __init__(
        self,
        responses: dict[tuple[str, tuple[float, ...], float], KNNPredictionResult],
    ) -> None:
        self._responses = responses
        self.calls: list[tuple[str, tuple[float, ...], int, float | None]] = []

    async def predict(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        confidence_threshold: float | None = None,
    ) -> KNNPredictionResult:
        key = (tenant_id, tuple(query_embedding), float(confidence_threshold or 0.0))
        self.calls.append((tenant_id, tuple(query_embedding), top_k, confidence_threshold))
        if key not in self._responses:
            raise AssertionError(f"missing fake response for key={key}")
        return self._responses[key]


def _case(
    case_id: str,
    tenant_id: str,
    query_embedding: list[float],
    expected_leaf_intent: str,
    expected_branch_intent: str,
) -> KNNEvaluationCase:
    return KNNEvaluationCase(
        case_id=case_id,
        tenant_id=tenant_id,
        query_embedding=query_embedding,
        expected_leaf_intent=expected_leaf_intent,
        expected_branch_intent=expected_branch_intent,
    )


def _prediction(
    tenant_id: str,
    top_intent_label: str,
    top_branch_intent: str,
    confidence: float,
    is_direct: bool,
    needs_fallback: bool,
    record_count: int = 3,
) -> KNNPredictionResult:
    return KNNPredictionResult(
        tenant_id=tenant_id,
        top_neighbors=[],
        top_intent_label=top_intent_label,
        top_branch_intent=top_branch_intent,
        confidence=confidence,
        is_direct=is_direct,
        needs_fallback=needs_fallback,
        record_count=record_count,
        top_score=confidence,
        margin_score=0.0,
        branch_consistency=0.0,
    )


@pytest.mark.asyncio
async def test_evaluate_cases_calculates_accuracy_and_direct_metrics():
    threshold = 0.8
    cases = [
        _case(
            "case-1",
            "tenant-a",
            [1.0, 0.0],
            "intent_task_booking_outpatient",
            "intent_task",
        ),
        _case(
            "case-2",
            "tenant-a",
            [0.0, 1.0],
            "intent_task_cancel_outpatient",
            "intent_task",
        ),
        _case(
            "case-3",
            "tenant-b",
            [1.0, 1.0],
            "intent_auth_identity_outpatient",
            "intent_auth",
        ),
    ]
    router = FakeRouter(
        {
            ("tenant-a", (1.0, 0.0), threshold): _prediction(
                tenant_id="tenant-a",
                top_intent_label="intent_task_booking_outpatient",
                top_branch_intent="intent_task",
                confidence=0.95,
                is_direct=True,
                needs_fallback=False,
            ),
            ("tenant-a", (0.0, 1.0), threshold): _prediction(
                tenant_id="tenant-a",
                top_intent_label="intent_task_booking_outpatient",
                top_branch_intent="intent_task",
                confidence=0.40,
                is_direct=False,
                needs_fallback=True,
            ),
            ("tenant-b", (1.0, 1.0), threshold): _prediction(
                tenant_id="tenant-b",
                top_intent_label="intent_faq_business_hours_outpatient",
                top_branch_intent="intent_faq",
                confidence=0.90,
                is_direct=True,
                needs_fallback=False,
            ),
        }
    )

    summary = await evaluate_cases(router, cases, threshold=threshold, top_k=3)

    assert summary.total_cases == 3
    assert summary.leaf_accuracy == pytest.approx(1 / 3)
    assert summary.branch_accuracy == pytest.approx(2 / 3)
    assert summary.direct_rate == pytest.approx(2 / 3)
    assert summary.fallback_rate == pytest.approx(1 / 3)
    assert summary.direct_leaf_accuracy == pytest.approx(1 / 2)
    assert summary.direct_branch_accuracy == pytest.approx(1 / 2)
    assert summary.fallback_count == 1
    assert len(summary.rows) == 3
    assert summary.rows[1].needs_fallback is True
    assert summary.rows[1].branch_correct is True


@pytest.mark.asyncio
async def test_evaluate_cases_returns_safe_summary_for_empty_case_list():
    router = FakeRouter({})

    summary = await evaluate_cases(router, [], threshold=0.8, top_k=3)

    assert summary.total_cases == 0
    assert summary.leaf_accuracy == 0.0
    assert summary.branch_accuracy == 0.0
    assert summary.direct_rate == 0.0
    assert summary.fallback_rate == 0.0
    assert summary.direct_leaf_accuracy == 0.0
    assert summary.direct_branch_accuracy == 0.0
    assert summary.fallback_count == 0
    assert summary.rows == []
    assert router.calls == []


@pytest.mark.asyncio
async def test_sweep_thresholds_evaluates_all_candidates_and_selects_best_threshold():
    cases = [
        _case(
            "case-1",
            "tenant-a",
            [1.0, 0.0],
            "intent_task_booking_outpatient",
            "intent_task",
        ),
        _case(
            "case-2",
            "tenant-a",
            [0.0, 1.0],
            "intent_faq_business_hours_outpatient",
            "intent_faq",
        ),
    ]
    router = FakeRouter(
        {
            ("tenant-a", (1.0, 0.0), 0.5): _prediction(
                tenant_id="tenant-a",
                top_intent_label="intent_task_booking_outpatient",
                top_branch_intent="intent_task",
                confidence=0.80,
                is_direct=True,
                needs_fallback=False,
            ),
            ("tenant-a", (0.0, 1.0), 0.5): _prediction(
                tenant_id="tenant-a",
                top_intent_label="intent_faq_business_hours_outpatient",
                top_branch_intent="intent_faq",
                confidence=0.82,
                is_direct=True,
                needs_fallback=False,
            ),
            ("tenant-a", (1.0, 0.0), 0.9): _prediction(
                tenant_id="tenant-a",
                top_intent_label="intent_task_booking_outpatient",
                top_branch_intent="intent_task",
                confidence=0.80,
                is_direct=False,
                needs_fallback=True,
            ),
            ("tenant-a", (0.0, 1.0), 0.9): _prediction(
                tenant_id="tenant-a",
                top_intent_label="intent_faq_business_hours_outpatient",
                top_branch_intent="intent_faq",
                confidence=0.82,
                is_direct=False,
                needs_fallback=True,
            ),
        }
    )

    result = await sweep_thresholds(router, cases, thresholds=[0.5, 0.9], top_k=2)

    assert len(result.all_summaries) == 2
    assert result.best_threshold == 0.5
    assert result.best_summary.threshold == 0.5
    assert result.best_summary.branch_accuracy == 1.0
    assert result.best_summary.direct_branch_accuracy == 1.0
    assert result.best_summary.fallback_rate == 0.0
    assert len(router.calls) == 4


@pytest.mark.asyncio
async def test_sweep_thresholds_raises_for_empty_threshold_list():
    router = FakeRouter({})

    with pytest.raises(ValueError):
        await sweep_thresholds(router, [], thresholds=[], top_k=2)


@pytest.mark.asyncio
async def test_evaluate_cases_keeps_tenant_cases_separated():
    threshold = 0.7
    cases = [
        _case(
            "case-a",
            "tenant-a",
            [1.0, 0.0],
            "intent_task_booking_outpatient",
            "intent_task",
        ),
        _case(
            "case-b",
            "tenant-b",
            [1.0, 0.0],
            "intent_auth_identity_outpatient",
            "intent_auth",
        ),
    ]
    router = FakeRouter(
        {
            ("tenant-a", (1.0, 0.0), threshold): _prediction(
                tenant_id="tenant-a",
                top_intent_label="intent_task_booking_outpatient",
                top_branch_intent="intent_task",
                confidence=0.93,
                is_direct=True,
                needs_fallback=False,
            ),
            ("tenant-b", (1.0, 0.0), threshold): _prediction(
                tenant_id="tenant-b",
                top_intent_label="intent_auth_identity_outpatient",
                top_branch_intent="intent_auth",
                confidence=0.91,
                is_direct=True,
                needs_fallback=False,
            ),
        }
    )

    summary = await evaluate_cases(router, cases, threshold=threshold, top_k=1)

    assert summary.total_cases == 2
    assert summary.leaf_accuracy == 1.0
    assert summary.branch_accuracy == 1.0
    assert [
        (row.tenant_id, row.predicted_leaf_intent)
        for row in summary.rows
    ] == [
        ("tenant-a", "intent_task_booking_outpatient"),
        ("tenant-b", "intent_auth_identity_outpatient"),
    ]
    assert [
        (tenant_id, query_embedding)
        for tenant_id, query_embedding, _top_k, _threshold in router.calls
    ] == [
        ("tenant-a", (1.0, 0.0)),
        ("tenant-b", (1.0, 0.0)),
    ]

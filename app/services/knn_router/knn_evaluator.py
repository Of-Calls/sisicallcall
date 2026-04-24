"""Evaluation utilities for tenant-scoped KNN router predictions.

This module evaluates KNN router outputs against a labeled evaluation set and
supports deterministic confidence-threshold sweeps. It stays at the service
layer and does not depend on graph/node orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.services.knn_router.knn import KNNPredictionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class KNNPredictorProtocol(Protocol):
    async def predict(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        confidence_threshold: float | None = None,
    ) -> KNNPredictionResult:
        ...


@dataclass(frozen=True)
class KNNEvaluationCase:
    tenant_id: str
    query_embedding: list[float]
    expected_leaf_intent: str
    expected_branch_intent: str
    case_id: str = ""


@dataclass(frozen=True)
class KNNEvaluationRow:
    case_id: str
    tenant_id: str
    predicted_leaf_intent: str
    predicted_branch_intent: str
    expected_leaf_intent: str
    expected_branch_intent: str
    confidence: float
    is_direct: bool
    needs_fallback: bool
    leaf_correct: bool
    branch_correct: bool


@dataclass(frozen=True)
class KNNEvaluationSummary:
    total_cases: int
    leaf_accuracy: float
    branch_accuracy: float
    direct_rate: float
    fallback_rate: float
    direct_leaf_accuracy: float
    direct_branch_accuracy: float
    fallback_count: int
    threshold: float
    rows: list[KNNEvaluationRow] = field(default_factory=list)


@dataclass(frozen=True)
class ThresholdSweepResult:
    best_threshold: float
    best_summary: KNNEvaluationSummary
    all_summaries: list[KNNEvaluationSummary] = field(default_factory=list)


async def evaluate_cases(
    router: KNNPredictorProtocol,
    cases: list[KNNEvaluationCase],
    threshold: float,
    top_k: int = 5,
) -> KNNEvaluationSummary:
    rows: list[KNNEvaluationRow] = []

    for case in cases:
        prediction = await router.predict(
            tenant_id=case.tenant_id,
            query_embedding=case.query_embedding,
            top_k=top_k,
            confidence_threshold=threshold,
        )

        predicted_leaf_intent = prediction.top_intent_label
        predicted_branch_intent = prediction.top_branch_intent
        row = KNNEvaluationRow(
            case_id=case.case_id,
            tenant_id=case.tenant_id,
            predicted_leaf_intent=predicted_leaf_intent,
            predicted_branch_intent=predicted_branch_intent,
            expected_leaf_intent=case.expected_leaf_intent,
            expected_branch_intent=case.expected_branch_intent,
            confidence=prediction.confidence,
            is_direct=prediction.is_direct,
            needs_fallback=prediction.needs_fallback,
            leaf_correct=predicted_leaf_intent == case.expected_leaf_intent,
            branch_correct=predicted_branch_intent == case.expected_branch_intent,
        )
        rows.append(row)

    summary = _build_summary(rows, threshold)
    logger.info(
        "evaluated knn cases total_cases=%d threshold=%.4f leaf_accuracy=%.4f branch_accuracy=%.4f direct_rate=%.4f",
        summary.total_cases,
        threshold,
        summary.leaf_accuracy,
        summary.branch_accuracy,
        summary.direct_rate,
    )
    return summary


async def sweep_thresholds(
    router: KNNPredictorProtocol,
    cases: list[KNNEvaluationCase],
    thresholds: list[float],
    top_k: int = 5,
) -> ThresholdSweepResult:
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    all_summaries: list[KNNEvaluationSummary] = []
    for threshold in thresholds:
        all_summaries.append(
            await evaluate_cases(
                router=router,
                cases=cases,
                threshold=threshold,
                top_k=top_k,
            )
        )

    best_summary = max(
        all_summaries,
        key=lambda summary: (
            summary.branch_accuracy,
            summary.direct_branch_accuracy,
            -summary.fallback_rate,
            summary.threshold,
        ),
    )

    logger.info(
        "completed knn threshold sweep threshold_count=%d best_threshold=%.4f best_branch_accuracy=%.4f",
        len(all_summaries),
        best_summary.threshold,
        best_summary.branch_accuracy,
    )
    return ThresholdSweepResult(
        best_threshold=best_summary.threshold,
        best_summary=best_summary,
        all_summaries=all_summaries,
    )


def _build_summary(
    rows: list[KNNEvaluationRow],
    threshold: float,
) -> KNNEvaluationSummary:
    total_cases = len(rows)
    if total_cases == 0:
        return KNNEvaluationSummary(
            total_cases=0,
            leaf_accuracy=0.0,
            branch_accuracy=0.0,
            direct_rate=0.0,
            fallback_rate=0.0,
            direct_leaf_accuracy=0.0,
            direct_branch_accuracy=0.0,
            fallback_count=0,
            threshold=threshold,
            rows=[],
        )

    leaf_correct_count = sum(1 for row in rows if row.leaf_correct)
    branch_correct_count = sum(1 for row in rows if row.branch_correct)
    direct_rows = [row for row in rows if row.is_direct]
    direct_count = len(direct_rows)
    fallback_count = sum(1 for row in rows if row.needs_fallback)
    direct_leaf_correct_count = sum(1 for row in direct_rows if row.leaf_correct)
    direct_branch_correct_count = sum(1 for row in direct_rows if row.branch_correct)

    return KNNEvaluationSummary(
        total_cases=total_cases,
        leaf_accuracy=leaf_correct_count / total_cases,
        branch_accuracy=branch_correct_count / total_cases,
        direct_rate=direct_count / total_cases,
        fallback_rate=fallback_count / total_cases,
        direct_leaf_accuracy=(
            direct_leaf_correct_count / direct_count if direct_count else 0.0
        ),
        direct_branch_accuracy=(
            direct_branch_correct_count / direct_count if direct_count else 0.0
        ),
        fallback_count=fallback_count,
        threshold=threshold,
        rows=rows,
    )

from dataclasses import dataclass
from typing import Any

import pytest

from app.services.knn_router.knn import (
    LOAD_SQL,
    KNNIntentRecord,
    KNNRouterService,
)


@dataclass
class FakeKNNIntentRow:
    tenant_id: str
    intent_label: str
    example_text: str
    embedding: list[float]
    updated_at: str = ""


class FakeDB:
    def __init__(self, rows: list[FakeKNNIntentRow] | None = None) -> None:
        self.rows: list[FakeKNNIntentRow] = list(rows or [])
        self.fetch_calls = 0
        self.fetch_history: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[FakeKNNIntentRow]:
        normalized_query = self._normalize_sql(query)
        if normalized_query != self._normalize_sql(LOAD_SQL):
            raise AssertionError(f"unexpected fetch query: {query}")

        self.fetch_calls += 1
        self.fetch_history.append((query, args))

        tenant_id = str(args[0])
        filtered_rows = [row for row in self.rows if row.tenant_id == tenant_id]
        filtered_rows.sort(
            key=lambda row: (
                row.updated_at,
                row.intent_label,
                row.example_text,
            )
        )
        return list(filtered_rows)

    def _normalize_sql(self, query: str) -> str:
        return " ".join(query.split())


def _row(
    tenant_id: str,
    intent_label: str,
    example_text: str,
    embedding: list[float],
    updated_at: str,
) -> FakeKNNIntentRow:
    return FakeKNNIntentRow(
        tenant_id=tenant_id,
        intent_label=intent_label,
        example_text=example_text,
        embedding=embedding,
        updated_at=updated_at,
    )


@pytest.mark.asyncio
async def test_load_tenant_intents_filters_by_tenant_id():
    db = FakeDB(
        [
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [1.0, 0.0], "2026-04-24T10:00:00"),
            _row("tenant-b", "intent_auth_identity_outpatient", "verify me", [0.0, 1.0], "2026-04-24T10:00:00"),
        ]
    )
    service = KNNRouterService(db)

    records = await service.load_tenant_intents("tenant-a")

    assert [record.intent_label for record in records] == [
        "intent_task_booking_outpatient"
    ]
    assert db.fetch_calls == 1
    assert db.fetch_history[0][1] == ("tenant-a",)


def test_build_tenant_index_builds_labels_embeddings_and_record_count():
    service = KNNRouterService(FakeDB())
    records = [
        KNNIntentRecord(
            intent_label="intent_task_booking_outpatient",
            example_text="book appointment",
            embedding=[1.0, 0.0],
            updated_at="2026-04-24T10:00:00",
        ),
        KNNIntentRecord(
            intent_label="intent_faq_business_hours_outpatient",
            example_text="clinic hours",
            embedding=[0.0, 1.0],
            updated_at="2026-04-24T11:00:00",
        ),
    ]

    index = service.build_tenant_index("tenant-a", records)

    assert index.tenant_id == "tenant-a"
    assert index.labels == [
        "intent_task_booking_outpatient",
        "intent_faq_business_hours_outpatient",
    ]
    assert index.embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert index.record_count == 2
    assert index.embedding_dimension == 2
    assert index.updated_at_max == "2026-04-24T11:00:00"


@pytest.mark.asyncio
async def test_predict_returns_top_k_sorted_by_cosine_similarity_desc():
    db = FakeDB(
        [
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [1.0, 0.0], "2026-04-24T10:00:00"),
            _row("tenant-a", "intent_faq_business_hours_outpatient", "clinic hours", [0.6, 0.8], "2026-04-24T10:01:00"),
            _row("tenant-a", "intent_auth_identity_outpatient", "verify identity", [0.0, 1.0], "2026-04-24T10:02:00"),
        ]
    )
    service = KNNRouterService(db)

    result = await service.predict("tenant-a", [1.0, 0.0], top_k=3, confidence_threshold=0.95)

    assert [neighbor.intent_label for neighbor in result.top_neighbors] == [
        "intent_task_booking_outpatient",
        "intent_faq_business_hours_outpatient",
        "intent_auth_identity_outpatient",
    ]
    assert result.top_neighbors[0].score > result.top_neighbors[1].score > result.top_neighbors[2].score


@pytest.mark.asyncio
async def test_predict_returns_top_leaf_and_coarse_branch():
    db = FakeDB(
        [
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [1.0, 0.0], "2026-04-24T10:00:00"),
            _row("tenant-a", "intent_faq_business_hours_outpatient", "clinic hours", [0.0, 1.0], "2026-04-24T10:01:00"),
        ]
    )
    service = KNNRouterService(db)

    result = await service.predict("tenant-a", [1.0, 0.0], top_k=2, confidence_threshold=0.95)

    assert result.top_intent_label == "intent_task_booking_outpatient"
    assert result.top_branch_intent == "intent_task"


@pytest.mark.asyncio
async def test_classify_returns_leaf_intent_not_coarse_branch():
    db = FakeDB(
        [
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [1.0, 0.0], "2026-04-24T10:00:00"),
            _row("tenant-a", "intent_faq_business_hours_outpatient", "clinic hours", [0.97, 0.24], "2026-04-24T10:01:00"),
        ]
    )
    service = KNNRouterService(db)

    intent_label, confidence = await service.classify([1.0, 0.0], "tenant-a")

    assert intent_label == "intent_task_booking_outpatient"
    assert intent_label != "intent_task"
    assert confidence < 0.85


@pytest.mark.asyncio
async def test_classify_returns_none_for_empty_tenant_index():
    service = KNNRouterService(FakeDB())

    intent_label, confidence = await service.classify([1.0, 0.0], "tenant-a")

    assert intent_label is None
    assert confidence == 0.0


@pytest.mark.asyncio
async def test_predict_applies_confidence_threshold_for_direct_vs_fallback():
    db = FakeDB(
        [
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [1.0, 0.0], "2026-04-24T10:00:00"),
            _row("tenant-a", "intent_task_cancel_outpatient", "cancel appointment", [0.98, 0.02], "2026-04-24T10:01:00"),
            _row("tenant-a", "intent_faq_business_hours_outpatient", "clinic hours", [0.0, 1.0], "2026-04-24T10:02:00"),
        ]
    )
    service = KNNRouterService(db)

    direct_result = await service.predict("tenant-a", [1.0, 0.0], top_k=2, confidence_threshold=0.85)
    fallback_result = await service.predict("tenant-a", [0.7, 0.7], top_k=3, confidence_threshold=0.85)

    assert direct_result.is_direct is True
    assert direct_result.needs_fallback is False
    assert fallback_result.is_direct is False
    assert fallback_result.needs_fallback is True


@pytest.mark.asyncio
async def test_predict_keeps_tenant_data_isolated():
    db = FakeDB(
        [
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [1.0, 0.0], "2026-04-24T10:00:00"),
            _row("tenant-b", "intent_auth_identity_outpatient", "verify identity", [0.0, 1.0], "2026-04-24T10:00:00"),
        ]
    )
    service = KNNRouterService(db)

    result_a = await service.predict("tenant-a", [1.0, 0.0], top_k=1, confidence_threshold=0.5)
    result_b = await service.predict("tenant-b", [0.0, 1.0], top_k=1, confidence_threshold=0.5)

    assert result_a.top_intent_label == "intent_task_booking_outpatient"
    assert result_b.top_intent_label == "intent_auth_identity_outpatient"
    assert result_a.top_intent_label != result_b.top_intent_label


@pytest.mark.asyncio
async def test_invalidate_tenant_index_forces_reload():
    db = FakeDB(
        [
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [1.0, 0.0], "2026-04-24T10:00:00"),
        ]
    )
    service = KNNRouterService(db)

    first = await service.predict("tenant-a", [1.0, 0.0], top_k=1, confidence_threshold=0.5)
    db.rows = [
        _row("tenant-a", "intent_auth_identity_outpatient", "verify identity", [1.0, 0.0], "2026-04-24T11:00:00"),
    ]
    second_without_invalidate = await service.predict("tenant-a", [1.0, 0.0], top_k=1, confidence_threshold=0.5)
    service.invalidate_tenant_index("tenant-a")
    third_after_invalidate = await service.predict("tenant-a", [1.0, 0.0], top_k=1, confidence_threshold=0.5)

    assert first.top_intent_label == "intent_task_booking_outpatient"
    assert second_without_invalidate.top_intent_label == "intent_task_booking_outpatient"
    assert third_after_invalidate.top_intent_label == "intent_auth_identity_outpatient"
    assert db.fetch_calls == 2


@pytest.mark.asyncio
async def test_predict_returns_safe_empty_result_for_empty_tenant():
    service = KNNRouterService(FakeDB())

    result = await service.predict("tenant-a", [1.0, 0.0], top_k=3, confidence_threshold=0.5)

    assert result.tenant_id == "tenant-a"
    assert result.top_neighbors == []
    assert result.top_intent_label == ""
    assert result.top_branch_intent == ""
    assert result.confidence == 0.0
    assert result.is_direct is False
    assert result.needs_fallback is True
    assert result.record_count == 0


@pytest.mark.asyncio
async def test_predict_handles_malformed_leaf_intent_with_safe_branch_fallback():
    db = FakeDB(
        [
            _row("tenant-a", "invalid_leaf", "strange intent", [1.0, 0.0], "2026-04-24T10:00:00"),
            _row("tenant-a", "intent_task_booking_outpatient", "book appointment", [0.0, 1.0], "2026-04-24T10:01:00"),
        ]
    )
    service = KNNRouterService(db)

    result = await service.predict("tenant-a", [1.0, 0.0], top_k=2, confidence_threshold=0.95)

    assert result.top_intent_label == "invalid_leaf"
    assert result.top_branch_intent == "intent_faq"
    assert result.top_neighbors[0].branch_intent == "intent_faq"


@pytest.mark.asyncio
async def test_build_tenant_index_skips_dimension_mismatch_rows():
    service = KNNRouterService(FakeDB())
    records = [
        KNNIntentRecord(
            intent_label="intent_task_booking_outpatient",
            example_text="book appointment",
            embedding=[1.0, 0.0],
            updated_at="2026-04-24T10:00:00",
        ),
        KNNIntentRecord(
            intent_label="intent_faq_business_hours_outpatient",
            example_text="clinic hours",
            embedding=[1.0, 0.0, 0.0],
            updated_at="2026-04-24T10:01:00",
        ),
    ]

    index = service.build_tenant_index("tenant-a", records)

    assert index.record_count == 1
    assert index.labels == ["intent_task_booking_outpatient"]
    assert index.embedding_dimension == 2


@pytest.mark.asyncio
async def test_predict_raises_for_empty_query_embedding():
    service = KNNRouterService(FakeDB())

    with pytest.raises(ValueError):
        await service.predict("tenant-a", [], top_k=3, confidence_threshold=0.5)

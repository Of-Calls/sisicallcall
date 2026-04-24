import copy
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import pytest

from app.services.embedding.base import BaseEmbeddingService
from app.services.knn_router.intent_bootstrap.dedup_diversity_selector import (
    FinalExampleSentence,
    FinalExampleSet,
)
from app.services.knn_router.intent_bootstrap.knn_intent_persister import (
    COUNT_SQL,
    DELETE_SQL,
    INSERT_SQL,
    KNNIntentPersister,
)


class FakeEmbeddingService(BaseEmbeddingService):
    def __init__(self) -> None:
        self.embed_calls = 0
        self.embed_batch_calls = 0
        self.embed_batch_inputs: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [float(len(text))]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls += 1
        self.embed_batch_inputs.append(list(texts))
        return [
            [float(index), float(len(text))]
            for index, text in enumerate(texts, start=1)
        ]


@dataclass
class FakeKNNIntentRow:
    tenant_id: str
    intent_label: str
    example_text: str
    embedding: list[float]


class FakeTransaction:
    def __init__(self, db: "FakeDB") -> None:
        self._db = db
        self._snapshot: list[FakeKNNIntentRow] | None = None

    async def __aenter__(self) -> "FakeTransaction":
        self._snapshot = copy.deepcopy(self._db.rows)
        self._db.transaction_entries += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None and self._snapshot is not None:
            self._db.rows = self._snapshot
        return False


class FakeDB:
    def __init__(self, rows: list[FakeKNNIntentRow] | None = None) -> None:
        self.rows: list[FakeKNNIntentRow] = list(rows or [])
        self.transaction_entries = 0

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def fetchval(self, query: str, *args: Any) -> int:
        if self._normalize_sql(query) != self._normalize_sql(COUNT_SQL):
            raise AssertionError(f"unexpected fetchval query: {query}")
        tenant_id = str(args[0])
        return sum(1 for row in self.rows if row.tenant_id == tenant_id)

    async def execute(self, query: str, *args: Any) -> str:
        normalized_query = self._normalize_sql(query)
        if normalized_query == self._normalize_sql(DELETE_SQL):
            tenant_id = str(args[0])
            before_count = len(self.rows)
            self.rows = [row for row in self.rows if row.tenant_id != tenant_id]
            deleted_count = before_count - len(self.rows)
            return f"DELETE {deleted_count}"

        if normalized_query == self._normalize_sql(INSERT_SQL):
            tenant_id, intent_label, example_text, embedding = args
            self.rows.append(
                FakeKNNIntentRow(
                    tenant_id=str(tenant_id),
                    intent_label=str(intent_label),
                    example_text=str(example_text),
                    embedding=list(embedding),
                )
            )
            return "INSERT 0 1"

        raise AssertionError(f"unexpected execute query: {query}")

    def _normalize_sql(self, query: str) -> str:
        return " ".join(query.split())


def _sentence(
    text: str,
    *,
    normalized_text: str = "",
    leaf_intent: str = "intent_faq_business_hours_outpatient",
    source: str = "seed",
    score_hint: float = 0.5,
) -> FinalExampleSentence:
    return FinalExampleSentence(
        text=text,
        normalized_text=normalized_text,
        source=source,
        leaf_intent=leaf_intent,
        topic_slug="business_hours",
        detail_slug="outpatient",
        chunk_id="chunk-1",
        score_hint=score_hint,
        selection_score=score_hint,
    )


def _final_set(
    leaf_intent: str,
    examples: list[FinalExampleSentence],
) -> FinalExampleSet:
    return FinalExampleSet(
        leaf_intent=leaf_intent,
        topic_slug="business_hours",
        detail_slug="outpatient",
        selected_examples=examples,
    )


@pytest.mark.asyncio
async def test_prepare_records_calls_embed_batch_once():
    embedder = FakeEmbeddingService()
    persister = KNNIntentPersister(embedder)

    records = await persister.prepare_records(
        "tenant-a",
        [
            _final_set(
                "intent_faq_business_hours_outpatient",
                [
                    _sentence("What are the clinic hours?", normalized_text="whataretheclinichours"),
                    _sentence("What time do you close?", normalized_text="whattimedoyouclose"),
                ],
            )
        ],
    )

    assert len(records) == 2
    assert embedder.embed_batch_calls == 1
    assert embedder.embed_calls == 0


@pytest.mark.asyncio
async def test_prepare_records_preserves_flatten_and_embedding_order():
    embedder = FakeEmbeddingService()
    persister = KNNIntentPersister(embedder)
    final_sets = [
        _final_set(
            "intent_faq_business_hours_outpatient",
            [
                _sentence("first question", normalized_text="firstquestion"),
                _sentence("second question", normalized_text="secondquestion"),
            ],
        ),
        _final_set(
            "intent_faq_reservation_outpatient",
            [
                _sentence(
                    "third question",
                    normalized_text="thirdquestion",
                    leaf_intent="intent_faq_reservation_outpatient",
                )
            ],
        ),
    ]

    records = await persister.prepare_records("tenant-a", final_sets)

    assert embedder.embed_batch_inputs == [[
        "first question",
        "second question",
        "third question",
    ]]
    assert [record.example_text for record in records] == [
        "first question",
        "second question",
        "third question",
    ]
    assert [record.embedding for record in records] == [
        [1.0, float(len("first question"))],
        [2.0, float(len("second question"))],
        [3.0, float(len("third question"))],
    ]


@pytest.mark.asyncio
async def test_prepare_records_dedupes_by_intent_label_and_normalized_text():
    embedder = FakeEmbeddingService()
    persister = KNNIntentPersister(embedder)
    final_sets = [
        _final_set(
            "intent_faq_business_hours_outpatient",
            [
                _sentence("Clinic hours please?", normalized_text=""),
                _sentence("Clinic   hours please?!", normalized_text=""),
            ],
        ),
        _final_set(
            "intent_faq_location_outpatient",
            [
                _sentence(
                    "Clinic hours please?",
                    normalized_text="clinichoursplease",
                    leaf_intent="intent_faq_location_outpatient",
                )
            ],
        ),
    ]

    records = await persister.prepare_records("tenant-a", final_sets)

    assert len(records) == 2
    assert [
        (record.intent_label, record.normalized_text)
        for record in records
    ] == [
        ("intent_faq_business_hours_outpatient", "clinichoursplease"),
        ("intent_faq_location_outpatient", "clinichoursplease"),
    ]


@pytest.mark.asyncio
async def test_persist_final_sets_replaces_only_target_tenant_rows():
    embedder = FakeEmbeddingService()
    persister = KNNIntentPersister(embedder)
    db = FakeDB(
        rows=[
            FakeKNNIntentRow("tenant-a", "old.intent", "old question 1", [0.1]),
            FakeKNNIntentRow("tenant-a", "old.intent", "old question 2", [0.2]),
            FakeKNNIntentRow("tenant-b", "other.intent", "other tenant question", [9.9]),
        ]
    )

    result = await persister.persist_final_sets(
        db,
        "tenant-a",
        [
            _final_set(
                "intent_faq_business_hours_outpatient",
                [
                    _sentence("What are the clinic hours?", normalized_text="whataretheclinichours"),
                    _sentence("What time do you close?", normalized_text="whattimedoyouclose"),
                ],
            )
        ],
    )

    tenant_a_rows = [row for row in db.rows if row.tenant_id == "tenant-a"]
    tenant_b_rows = [row for row in db.rows if row.tenant_id == "tenant-b"]

    assert result.tenant_id == "tenant-a"
    assert result.prepared_count == 2
    assert result.deleted_count == 2
    assert result.inserted_count == 2
    assert [row.example_text for row in tenant_a_rows] == [
        "What are the clinic hours?",
        "What time do you close?",
    ]
    assert [(row.tenant_id, row.example_text) for row in tenant_b_rows] == [
        ("tenant-b", "other tenant question")
    ]
    assert db.transaction_entries == 1


@pytest.mark.asyncio
async def test_persist_final_sets_with_empty_input_clears_target_tenant_only():
    embedder = FakeEmbeddingService()
    persister = KNNIntentPersister(embedder)
    db = FakeDB(
        rows=[
            FakeKNNIntentRow("tenant-a", "old.intent", "old question 1", [0.1]),
            FakeKNNIntentRow("tenant-a", "old.intent", "old question 2", [0.2]),
            FakeKNNIntentRow("tenant-b", "other.intent", "other tenant question", [9.9]),
        ]
    )

    result = await persister.persist_final_sets(db, "tenant-a", [])

    assert result.tenant_id == "tenant-a"
    assert result.prepared_count == 0
    assert result.deleted_count == 2
    assert result.inserted_count == 0
    assert [row.tenant_id for row in db.rows] == ["tenant-b"]
    assert db.rows[0].example_text == "other tenant question"
    assert embedder.embed_batch_calls == 0


@pytest.mark.asyncio
async def test_prepare_records_skips_blank_example_text():
    embedder = FakeEmbeddingService()
    persister = KNNIntentPersister(embedder)

    records = await persister.prepare_records(
        "tenant-a",
        [
            _final_set(
                "intent_faq_business_hours_outpatient",
                [
                    _sentence(""),
                    _sentence("   "),
                    _sentence("Valid example", normalized_text="validexample"),
                ],
            )
        ],
    )

    assert len(records) == 1
    assert records[0].example_text == "Valid example"
    assert embedder.embed_batch_inputs == [["Valid example"]]


@pytest.mark.asyncio
async def test_persist_final_sets_returns_expected_counts_after_defensive_dedupe():
    embedder = FakeEmbeddingService()
    persister = KNNIntentPersister(embedder)
    db = FakeDB(
        rows=[
            FakeKNNIntentRow("tenant-a", "old.intent", "old question 1", [0.1]),
            FakeKNNIntentRow("tenant-a", "old.intent", "old question 2", [0.2]),
            FakeKNNIntentRow("tenant-a", "old.intent", "old question 3", [0.3]),
        ]
    )

    result = await persister.persist_final_sets(
        db,
        "tenant-a",
        [
            _final_set(
                "intent_faq_business_hours_outpatient",
                [
                    _sentence("Clinic hours please?", normalized_text=""),
                    _sentence("Clinic   hours please?!", normalized_text=""),
                    _sentence("What time do you close?", normalized_text="whattimedoyouclose"),
                ],
            )
        ],
    )

    assert result.prepared_count == 2
    assert result.deleted_count == 3
    assert result.inserted_count == 2
    assert [row.example_text for row in db.rows] == [
        "Clinic hours please?",
        "What time do you close?",
    ]

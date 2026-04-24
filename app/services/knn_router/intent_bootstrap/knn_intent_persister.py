"""Persist final KNN intent examples into the knn_intents table.

This module is the sixth bootstrap stage for intent routing. It takes the
final deduplicated example sets, generates embeddings, and replaces one
tenant's KNN input rows in a single transaction.

It does not extract chunks, normalize intents, generate examples, or build
runtime KNN indexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol

from app.services.embedding.base import BaseEmbeddingService
from app.services.knn_router.intent_bootstrap.dedup_diversity_selector import (
    FinalExampleSet,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

COUNT_SQL = """
SELECT COUNT(*)
FROM knn_intents
WHERE tenant_id = $1
""".strip()

DELETE_SQL = """
DELETE FROM knn_intents
WHERE tenant_id = $1
""".strip()

INSERT_SQL = """
INSERT INTO knn_intents (
    tenant_id,
    intent_label,
    example_text,
    embedding
)
VALUES ($1, $2, $3, $4)
""".strip()


class AsyncTransactionProtocol(Protocol):
    async def __aenter__(self) -> Any:
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        ...


class AsyncDBConnectionProtocol(Protocol):
    async def fetchval(self, query: str, *args: Any) -> Any:
        ...

    async def execute(self, query: str, *args: Any) -> Any:
        ...

    def transaction(self) -> AsyncTransactionProtocol:
        ...


@dataclass(frozen=True)
class PreparedKNNIntentRecord:
    tenant_id: str
    intent_label: str
    example_text: str
    normalized_text: str
    embedding: list[float]


@dataclass(frozen=True)
class KNNIntentPersistResult:
    tenant_id: str
    prepared_count: int
    deleted_count: int
    inserted_count: int


@dataclass(frozen=True)
class _FlattenedExample:
    intent_label: str
    example_text: str
    normalized_text: str


class KNNIntentPersister:
    def __init__(self, embedder: BaseEmbeddingService):
        self._embedder = embedder

    async def persist_final_sets(
        self,
        db: AsyncDBConnectionProtocol,
        tenant_id: str,
        final_sets: list[FinalExampleSet],
    ) -> KNNIntentPersistResult:
        self._validate_tenant_id(tenant_id)
        records = await self.prepare_records(tenant_id, final_sets)
        return await self.replace_tenant_records(db, tenant_id, records)

    async def prepare_records(
        self,
        tenant_id: str,
        final_sets: list[FinalExampleSet],
    ) -> list[PreparedKNNIntentRecord]:
        self._validate_tenant_id(tenant_id)
        flattened_examples = self._flatten_examples(final_sets)
        if not flattened_examples:
            logger.info(
                "prepared no knn intent records tenant_id=%s",
                tenant_id,
            )
            return []

        example_texts = [example.example_text for example in flattened_examples]
        embeddings = await self._embedder.embed_batch(example_texts)
        if len(embeddings) != len(flattened_examples):
            raise ValueError(
                "embed_batch result size must match prepared example count"
            )

        records = [
            PreparedKNNIntentRecord(
                tenant_id=tenant_id,
                intent_label=example.intent_label,
                example_text=example.example_text,
                normalized_text=example.normalized_text,
                embedding=embedding,
            )
            for example, embedding in zip(flattened_examples, embeddings)
        ]

        logger.info(
            "prepared knn intent records tenant_id=%s prepared_count=%d",
            tenant_id,
            len(records),
        )
        return records

    async def replace_tenant_records(
        self,
        db: AsyncDBConnectionProtocol,
        tenant_id: str,
        records: list[PreparedKNNIntentRecord],
    ) -> KNNIntentPersistResult:
        self._validate_tenant_id(tenant_id)
        for record in records:
            if record.tenant_id != tenant_id:
                raise ValueError("record tenant_id must match replace tenant_id")

        async with db.transaction():
            deleted_count = int(await db.fetchval(COUNT_SQL, tenant_id) or 0)
            await db.execute(DELETE_SQL, tenant_id)

            inserted_count = 0
            for record in records:
                await db.execute(
                    INSERT_SQL,
                    record.tenant_id,
                    record.intent_label,
                    record.example_text,
                    record.embedding,
                )
                inserted_count += 1

        logger.info(
            "persisted knn intent records tenant_id=%s prepared_count=%d deleted_count=%d inserted_count=%d",
            tenant_id,
            len(records),
            deleted_count,
            inserted_count,
        )
        return KNNIntentPersistResult(
            tenant_id=tenant_id,
            prepared_count=len(records),
            deleted_count=deleted_count,
            inserted_count=inserted_count,
        )

    def _flatten_examples(
        self,
        final_sets: list[FinalExampleSet],
    ) -> list[_FlattenedExample]:
        deduped: list[_FlattenedExample] = []
        seen_keys: set[tuple[str, str]] = set()

        for final_set in final_sets:
            intent_label = final_set.leaf_intent.strip()
            if not intent_label:
                logger.debug("skipping final set with blank leaf_intent")
                continue

            for example in final_set.selected_examples:
                example_text = example.text.strip()
                if not example_text:
                    logger.debug(
                        "skipping blank example text intent_label=%s",
                        intent_label,
                    )
                    continue

                normalized_text = example.normalized_text.strip() or self._normalize_text(
                    example_text
                )
                if not normalized_text:
                    logger.debug(
                        "skipping example with blank normalized text intent_label=%s example_text=%s",
                        intent_label,
                        example_text,
                    )
                    continue

                dedupe_key = (intent_label, normalized_text)
                if dedupe_key in seen_keys:
                    continue

                seen_keys.add(dedupe_key)
                deduped.append(
                    _FlattenedExample(
                        intent_label=intent_label,
                        example_text=example_text,
                        normalized_text=normalized_text,
                    )
                )

        return deduped

    def _normalize_text(self, text: str) -> str:
        lowered = text.strip().casefold()
        lowered = re.sub(r"\s+", " ", lowered)
        lowered = re.sub(r"[^\w\s\uac00-\ud7a3]", "", lowered)
        lowered = lowered.replace(" ", "")
        return lowered

    def _validate_tenant_id(self, tenant_id: str) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must not be blank")

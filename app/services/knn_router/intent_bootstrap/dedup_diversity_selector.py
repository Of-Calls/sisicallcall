"""Select final KNN training examples with deduplication and diversity control.

This module consumes leaf-intent example candidates, removes redundant examples,
and chooses a balanced final subset for each intent. It preserves metadata for
seed/paraphrase provenance and keeps selection deterministic.

It does not persist data or build downstream KNN indexes.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.embedding.base import BaseEmbeddingService
from app.services.knn_router.intent_bootstrap.example_generator import (
    ExampleSentence,
    LeafIntentExampleSet,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

MIN_QUESTION_LENGTH = 5


@dataclass
class FinalExampleSentence:
    text: str
    normalized_text: str
    source: str
    leaf_intent: str
    topic_slug: str
    detail_slug: str
    chunk_id: str = ""
    score_hint: float = 0.0
    selection_score: float = 0.0


@dataclass
class FinalExampleSet:
    leaf_intent: str
    topic_slug: str
    detail_slug: str
    selected_examples: list[FinalExampleSentence] = field(default_factory=list)


@dataclass
class DedupSelectionResult:
    final_sets: list[FinalExampleSet] = field(default_factory=list)


def dedup_selection_result_to_dict(result: DedupSelectionResult) -> dict[str, Any]:
    return asdict(result)


class DedupDiversitySelector:
    def __init__(self, embedder: BaseEmbeddingService):
        self._embedder = embedder

    def normalize_question_text(self, text: str) -> str:
        lowered = text.strip().casefold()
        lowered = re.sub(r"\s+", " ", lowered)
        lowered = re.sub(r"[^\w\s가-힣]", "", lowered)
        lowered = lowered.replace(" ", "")
        return lowered

    def dedup_exact(self, examples: list[ExampleSentence]) -> list[ExampleSentence]:
        best_by_text: dict[str, ExampleSentence] = {}
        order: list[str] = []

        for example in examples:
            raw_text = example.text.strip()
            if len(raw_text) < MIN_QUESTION_LENGTH:
                continue
            existing = best_by_text.get(raw_text)
            if existing is None:
                best_by_text[raw_text] = example
                order.append(raw_text)
                continue
            if self._example_priority(example) > self._example_priority(existing):
                best_by_text[raw_text] = example

        return [best_by_text[text] for text in order]

    def dedup_normalized(self, examples: list[ExampleSentence]) -> list[ExampleSentence]:
        best_by_normalized: dict[str, ExampleSentence] = {}
        order: list[str] = []

        for example in examples:
            normalized = self.normalize_question_text(example.text)
            if len(normalized) < MIN_QUESTION_LENGTH:
                continue
            existing = best_by_normalized.get(normalized)
            if existing is None:
                best_by_normalized[normalized] = example
                order.append(normalized)
                continue
            if self._example_priority(example) > self._example_priority(existing):
                best_by_normalized[normalized] = example

        return [best_by_normalized[key] for key in order]

    async def dedup_by_similarity(
        self,
        examples: list[ExampleSentence],
        threshold: float = 0.94,
    ) -> list[ExampleSentence]:
        if not examples:
            return []

        embeddings = await self._embedder.embed_batch([example.text for example in examples])
        ranked_indices = sorted(
            range(len(examples)),
            key=lambda index: (
                self._example_priority(examples[index]),
                examples[index].text,
            ),
            reverse=True,
        )

        kept_indices: list[int] = []
        kept_embeddings: list[list[float]] = []

        for index in ranked_indices:
            embedding = self._normalize_embedding(embeddings[index])
            if any(
                self._cosine_similarity(embedding, kept_embedding) >= threshold
                for kept_embedding in kept_embeddings
            ):
                continue
            kept_indices.append(index)
            kept_embeddings.append(embedding)

        kept_indices.sort()
        return [examples[index] for index in kept_indices]

    async def select_diverse_examples(
        self,
        examples: list[ExampleSentence],
        target_k: int = 12,
    ) -> list[FinalExampleSentence]:
        if not examples or target_k <= 0:
            return []

        embeddings = await self._embedder.embed_batch([example.text for example in examples])
        normalized_embeddings = [self._normalize_embedding(embedding) for embedding in embeddings]

        seeds = [index for index, example in enumerate(examples) if example.source == "seed"]
        paraphrases = [index for index, example in enumerate(examples) if example.source != "seed"]

        desired_seed_count = 0
        if seeds:
            desired_seed_count = min(len(seeds), target_k, max(2, min(4, target_k // 3 or 1)))

        selected_indices: list[int] = []
        selected_indices.extend(
            self._mmr_select_indices(
                candidate_indices=seeds,
                examples=examples,
                embeddings=normalized_embeddings,
                quota=desired_seed_count,
                already_selected=[],
            )
        )

        remaining_quota = max(target_k - len(selected_indices), 0)
        if remaining_quota > 0:
            remaining_indices = [index for index in range(len(examples)) if index not in selected_indices]
            selected_indices.extend(
                self._mmr_select_indices(
                    candidate_indices=remaining_indices,
                    examples=examples,
                    embeddings=normalized_embeddings,
                    quota=remaining_quota,
                    already_selected=selected_indices,
                )
            )

        final_examples: list[FinalExampleSentence] = []
        for index in selected_indices[:target_k]:
            example = examples[index]
            final_examples.append(
                FinalExampleSentence(
                    text=example.text,
                    normalized_text=self.normalize_question_text(example.text),
                    source=example.source,
                    leaf_intent=example.leaf_intent,
                    topic_slug=example.topic_slug,
                    detail_slug=example.detail_slug,
                    chunk_id=example.chunk_id,
                    score_hint=example.score_hint,
                    selection_score=self._selection_base_score(example),
                )
            )

        return final_examples

    # dedup_diversity_selector.py
    async def build_final_example_sets(
        self,
        leaf_intent_sets: list[LeafIntentExampleSet],
        target_k: int = 12,
        similarity_threshold: float = 0.94,
    ) -> DedupSelectionResult:
        final_sets: list[FinalExampleSet] = []

        for leaf_intent_set in leaf_intent_sets:
            pool = [
                *leaf_intent_set.seed_examples,
                *leaf_intent_set.paraphrase_examples,
            ]
            exact_deduped = self.dedup_exact(pool)
            normalized_deduped = self.dedup_normalized(exact_deduped)

            if not normalized_deduped:
                final_sets.append(
                    FinalExampleSet(
                        leaf_intent=leaf_intent_set.leaf_intent,
                        topic_slug=leaf_intent_set.topic_slug,
                        detail_slug=leaf_intent_set.detail_slug,
                        selected_examples=[],
                    )
                )
                continue

            embeddings = await self._embedder.embed_batch(
                [example.text for example in normalized_deduped]
            )

            semantic_deduped, semantic_embeddings = self._dedup_by_similarity_with_embeddings(
                normalized_deduped,
                embeddings,
                threshold=similarity_threshold,
            )

            selected_examples = self._select_diverse_examples_with_embeddings(
                semantic_deduped,
                semantic_embeddings,
                target_k=target_k,
            )

            final_sets.append(
                FinalExampleSet(
                    leaf_intent=leaf_intent_set.leaf_intent,
                    topic_slug=leaf_intent_set.topic_slug,
                    detail_slug=leaf_intent_set.detail_slug,
                    selected_examples=selected_examples,
                )
            )

        return DedupSelectionResult(final_sets=final_sets)

    def _mmr_select_indices(
        self,
        candidate_indices: list[int],
        examples: list[ExampleSentence],
        embeddings: list[list[float]],
        quota: int,
        already_selected: list[int],
        lambda_weight: float = 0.7,
    ) -> list[int]:
        if quota <= 0 or not candidate_indices:
            return []

        selected: list[int] = []
        remaining = list(candidate_indices)

        while remaining and len(selected) < quota:
            best_index = max(
                remaining,
                key=lambda index: self._mmr_score(
                    index=index,
                    examples=examples,
                    embeddings=embeddings,
                    selected=already_selected + selected,
                    lambda_weight=lambda_weight,
                ),
            )
            selected.append(best_index)
            remaining.remove(best_index)

        return selected

    def _mmr_score(
        self,
        index: int,
        examples: list[ExampleSentence],
        embeddings: list[list[float]],
        selected: list[int],
        lambda_weight: float,
    ) -> tuple[float, float, str]:
        base_score = self._selection_base_score(examples[index])
        if not selected:
            return (base_score, self._source_bonus(examples[index]), examples[index].text)

        max_similarity = max(
            self._cosine_similarity(embeddings[index], embeddings[selected_index])
            for selected_index in selected
        )
        mmr = (lambda_weight * base_score) - ((1.0 - lambda_weight) * max_similarity)
        return (mmr, self._source_bonus(examples[index]), examples[index].text)

    def _selection_base_score(self, example: ExampleSentence) -> float:
        return self._example_priority(example)

    def _example_priority(self, example: ExampleSentence) -> float:
        return example.score_hint + self._source_bonus(example)

    def _source_bonus(self, example: ExampleSentence) -> float:
        return 0.2 if example.source == "seed" else 0.0

    def _normalize_embedding(self, vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            return list(vec)
        return [value / norm for value in vec]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        dot = sum(x * y for x, y in zip(left, right))
        norm_left = math.sqrt(sum(value * value for value in left))
        norm_right = math.sqrt(sum(value * value for value in right))
        if norm_left == 0.0 or norm_right == 0.0:
            return 0.0
        return dot / (norm_left * norm_right)

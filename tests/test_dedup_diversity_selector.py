import pytest

from app.services.embedding.base import BaseEmbeddingService
from app.services.knn_router.intent_bootstrap.dedup_diversity_selector import (
    DedupDiversitySelector,
)
from app.services.knn_router.intent_bootstrap.example_generator import (
    ExampleSentence,
    LeafIntentExampleSet,
)


class FakeEmbeddingService(BaseEmbeddingService):
    def __init__(self) -> None:
        self.embed_calls = 0
        self.embed_batch_calls = 0

    def _vector_for(self, text: str) -> list[float]:
        lowered = text.casefold()
        if "weekend" in lowered or "saturday" in lowered:
            return [0.7, 0.7, 0.0]
        if "hours" in lowered or "open" in lowered or "close" in lowered:
            return [1.0, 0.0, 0.0]
        if "reservation" in lowered or "appointment" in lowered:
            return [0.0, 1.0, 0.0]
        if "parking" in lowered or "location" in lowered:
            return [0.0, 0.0, 1.0]
        return [0.3, 0.3, 0.4]

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._vector_for(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls += 1
        return [self._vector_for(text) for text in texts]


def _example(
    text: str,
    source: str = "paraphrase",
    score_hint: float = 0.5,
) -> ExampleSentence:
    return ExampleSentence(
        text=text,
        source=source,
        leaf_intent="intent_faq_business_hours_outpatient",
        topic_slug="business_hours",
        detail_slug="outpatient",
        chunk_id="chunk-1",
        score_hint=score_hint,
    )


def _leaf_set(examples: list[ExampleSentence]) -> LeafIntentExampleSet:
    return LeafIntentExampleSet(
        leaf_intent="intent_faq_business_hours_outpatient",
        topic_slug="business_hours",
        detail_slug="outpatient",
        seed_examples=[example for example in examples if example.source == "seed"],
        paraphrase_examples=[example for example in examples if example.source != "seed"],
    )


def test_dedup_exact_prefers_seed_for_same_text():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("Clinic hours please?", source="paraphrase", score_hint=0.1),
        _example("Clinic hours please?", source="seed", score_hint=0.2),
    ]

    deduped = selector.dedup_exact(examples)

    assert len(deduped) == 1
    assert deduped[0].source == "seed"


def test_normalization_duplicate_removes_spacing_and_punctuation_variants():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("Clinic hours please?", source="seed"),
        _example("Clinic   hours please?!", source="paraphrase"),
    ]

    deduped = selector.dedup_normalized(examples)

    assert len(deduped) == 1


@pytest.mark.asyncio
async def test_similarity_threshold_removes_semantic_duplicates():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("What are the clinic hours?", source="seed", score_hint=0.9),
        _example("What time do you close?", source="paraphrase", score_hint=0.8),
        _example("Do I need a reservation?", source="paraphrase", score_hint=0.7),
    ]

    deduped = await selector.dedup_by_similarity(examples, threshold=0.9)

    assert len(deduped) == 2
    texts = {example.text for example in deduped}
    assert "Do I need a reservation?" in texts


@pytest.mark.asyncio
async def test_build_final_example_sets_keeps_seed_balance_and_target_limit():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("What are the clinic hours?", source="seed", score_hint=0.9),
        _example("Are you open on weekends?", source="seed", score_hint=0.85),
        _example("What time do you open on Saturday?", source="seed", score_hint=0.8),
        _example("What time do you close?", source="paraphrase", score_hint=0.7),
        _example("Please share the clinic hours.", source="paraphrase", score_hint=0.68),
        _example("Do I need a reservation?", source="paraphrase", score_hint=0.66),
        _example("Is an appointment required?", source="paraphrase", score_hint=0.65),
        _example("Where is the parking lot?", source="paraphrase", score_hint=0.64),
        _example("What is the parking location?", source="paraphrase", score_hint=0.63),
    ]

    result = await selector.build_final_example_sets(
        [_leaf_set(examples)],
        target_k=5,
        similarity_threshold=0.92,
    )

    assert len(result.final_sets) == 1
    selected = result.final_sets[0].selected_examples
    assert len(selected) <= 5
    assert sum(1 for example in selected if example.source == "seed") >= 2


@pytest.mark.asyncio
async def test_final_selection_is_deterministic():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("What are the clinic hours?", source="seed", score_hint=0.9),
        _example("Are you open on weekends?", source="seed", score_hint=0.85),
        _example("What time do you open on Saturday?", source="seed", score_hint=0.8),
        _example("What time do you close?", source="paraphrase", score_hint=0.7),
        _example("Do I need a reservation?", source="paraphrase", score_hint=0.66),
        _example("Where is the parking lot?", source="paraphrase", score_hint=0.64),
    ]
    leaf_set = _leaf_set(examples)

    first = await selector.build_final_example_sets(
        [leaf_set],
        target_k=4,
        similarity_threshold=0.92,
    )
    second = await selector.build_final_example_sets(
        [leaf_set],
        target_k=4,
        similarity_threshold=0.92,
    )

    first_texts = [example.text for example in first.final_sets[0].selected_examples]
    second_texts = [example.text for example in second.final_sets[0].selected_examples]

    assert first_texts == second_texts


@pytest.mark.asyncio
async def test_build_final_example_sets_reuses_embeddings_for_dedup_and_diversity():
    embedder = FakeEmbeddingService()
    selector = DedupDiversitySelector(embedder)
    examples = [
        _example("What are the clinic hours?", source="seed", score_hint=0.9),
        _example("What time do you close?", source="paraphrase", score_hint=0.7),
        _example("Do I need a reservation?", source="paraphrase", score_hint=0.6),
    ]

    result = await selector.build_final_example_sets(
        [_leaf_set(examples)],
        target_k=3,
        similarity_threshold=0.92,
    )

    assert len(result.final_sets[0].selected_examples) >= 1
    assert embedder.embed_batch_calls == 1
    assert embedder.embed_calls == 0


@pytest.mark.asyncio
async def test_build_final_example_sets_returns_empty_selection_for_empty_pool():
    embedder = FakeEmbeddingService()
    selector = DedupDiversitySelector(embedder)

    result = await selector.build_final_example_sets(
        [_leaf_set([])],
        target_k=4,
        similarity_threshold=0.92,
    )

    assert len(result.final_sets) == 1
    assert result.final_sets[0].selected_examples == []
    assert embedder.embed_batch_calls == 0


@pytest.mark.asyncio
async def test_build_final_example_sets_applies_normalized_and_semantic_dedup_before_diversity():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("Clinic hours please?", source="seed", score_hint=0.9),
        _example("Clinic   hours please?!", source="paraphrase", score_hint=0.8),
        _example("What time do you close?", source="paraphrase", score_hint=0.7),
        _example("Do I need a reservation?", source="paraphrase", score_hint=0.6),
    ]

    result = await selector.build_final_example_sets(
        [_leaf_set(examples)],
        target_k=4,
        similarity_threshold=0.9,
    )

    selected_texts = {example.text for example in result.final_sets[0].selected_examples}
    assert "Clinic hours please?" in selected_texts
    assert "Do I need a reservation?" in selected_texts
    assert "Clinic   hours please?!" not in selected_texts
    assert "What time do you close?" not in selected_texts

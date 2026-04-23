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
    async def embed(self, text: str) -> list[float]:
        lowered = text.casefold()
        if "토요일" in lowered or "주말" in lowered:
            return [0.7, 0.7, 0.0]
        if "운영 시간" in lowered or "몇 시" in lowered or "몇시" in lowered:
            return [1.0, 0.0, 0.0]
        if "예약" in lowered:
            return [0.0, 1.0, 0.0]
        if "주차" in lowered:
            return [0.0, 0.0, 1.0]
        return [0.3, 0.3, 0.4]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


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
        _example("외래 진료 시간 알려주세요", source="paraphrase", score_hint=0.1),
        _example("외래 진료 시간 알려주세요", source="seed", score_hint=0.2),
    ]

    deduped = selector.dedup_exact(examples)

    assert len(deduped) == 1
    assert deduped[0].source == "seed"


def test_normalization_duplicate_removes_spacing_and_punctuation_variants():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("외래 진료 시간 알려주세요", source="seed"),
        _example("외래   진료 시간 알려주세요!!", source="paraphrase"),
    ]

    deduped = selector.dedup_normalized(examples)

    assert len(deduped) == 1


@pytest.mark.asyncio
async def test_similarity_threshold_removes_semantic_duplicates():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("외래 진료 운영 시간이 어떻게 되나요?", source="seed", score_hint=0.9),
        _example("외래 진료 몇 시까지 하나요?", source="paraphrase", score_hint=0.8),
        _example("외래 진료 예약해야 하나요?", source="paraphrase", score_hint=0.7),
    ]

    deduped = await selector.dedup_by_similarity(examples, threshold=0.9)

    assert len(deduped) == 2
    texts = {example.text for example in deduped}
    assert "외래 진료 예약해야 하나요?" in texts


@pytest.mark.asyncio
async def test_build_final_example_sets_keeps_seed_balance_and_target_limit():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("외래 진료 운영 시간이 어떻게 되나요?", source="seed", score_hint=0.9),
        _example("토요일에도 외래 진료하나요?", source="seed", score_hint=0.85),
        _example("평일 외래 진료는 몇 시까지 하나요?", source="seed", score_hint=0.8),
        _example("외래 진료 몇 시까지 해요?", source="paraphrase", score_hint=0.7),
        _example("외래 진료 운영 시간 문의드려요", source="paraphrase", score_hint=0.68),
        _example("외래 진료 예약해야 하나요?", source="paraphrase", score_hint=0.66),
        _example("외래 진료 예약 필요한가요?", source="paraphrase", score_hint=0.65),
        _example("주차는 어디에 하면 되나요?", source="paraphrase", score_hint=0.64),
        _example("주차장 위치가 어디인가요?", source="paraphrase", score_hint=0.63),
    ]

    result = await selector.build_final_example_sets([_leaf_set(examples)], target_k=5, similarity_threshold=0.92)

    assert len(result.final_sets) == 1
    selected = result.final_sets[0].selected_examples
    assert len(selected) <= 5
    assert sum(1 for example in selected if example.source == "seed") >= 2


@pytest.mark.asyncio
async def test_final_selection_is_deterministic():
    selector = DedupDiversitySelector(FakeEmbeddingService())
    examples = [
        _example("외래 진료 운영 시간이 어떻게 되나요?", source="seed", score_hint=0.9),
        _example("토요일에도 외래 진료하나요?", source="seed", score_hint=0.85),
        _example("평일 외래 진료는 몇 시까지 하나요?", source="seed", score_hint=0.8),
        _example("외래 진료 몇 시까지 해요?", source="paraphrase", score_hint=0.7),
        _example("외래 진료 예약해야 하나요?", source="paraphrase", score_hint=0.66),
        _example("주차는 어디에 하면 되나요?", source="paraphrase", score_hint=0.64),
    ]
    leaf_set = _leaf_set(examples)

    first = await selector.build_final_example_sets([leaf_set], target_k=4, similarity_threshold=0.92)
    second = await selector.build_final_example_sets([leaf_set], target_k=4, similarity_threshold=0.92)

    first_texts = [example.text for example in first.final_sets[0].selected_examples]
    second_texts = [example.text for example in second.final_sets[0].selected_examples]

    assert first_texts == second_texts

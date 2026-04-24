import pytest

from app.services.knn_router.intent_bootstrap.intent_registry import IntentRegistryBuilder
from app.services.knn_router.intent_bootstrap.normalizer import (
    ChunkIntentCandidates,
    NormalizedAskableUnit,
)


def _unit(
    leaf_intent: str,
    topic_slug: str,
    detail_slug: str,
    salience: float = 0.9,
    answerability: float = 0.9,
) -> NormalizedAskableUnit:
    return NormalizedAskableUnit(
        attribute="business hours",
        entity="outpatient clinic",
        value_hint="09:00-18:00",
        evidence="Mon-Fri 09:00-18:00",
        salience=salience,
        answerability=answerability,
        topic_slug=topic_slug,
        detail_slug=detail_slug,
        leaf_intent=leaf_intent,
    )


def _candidate(
    chunk_id: str,
    primary_leaf_intent: str,
    candidate_leaf_intents: list[str],
    normalized_units: list[NormalizedAskableUnit],
) -> ChunkIntentCandidates:
    return ChunkIntentCandidates(
        chunk_id=chunk_id,
        chunk_text="sample chunk text",
        category="faq",
        product_name="outpatient clinic",
        raw_keywords=["business hours"],
        primary_leaf_intent=primary_leaf_intent,
        candidate_leaf_intents=candidate_leaf_intents,
        normalized_units=normalized_units,
    )


async def _build_result(candidates: list[ChunkIntentCandidates]):
    builder = IntentRegistryBuilder()
    return await builder.build_registry(candidates)


def _canonical_inventory_item(result, canonical_leaf_intent: str):
    for item in result.registry.inventory:
        if item.canonical_leaf_intent == canonical_leaf_intent:
            return item
    raise AssertionError(f"missing inventory item for {canonical_leaf_intent}")


@pytest.mark.asyncio
async def test_alias_leaf_intents_merge_into_one_canonical():
    candidates = [
        _candidate(
            "chunk-1",
            "intent_faq_hours_outpatient",
            ["intent_faq_hours_outpatient"],
            [_unit("intent_faq_hours_outpatient", "hours", "outpatient")],
        ),
        _candidate(
            "chunk-2",
            "intent_faq_business_hours_outpatient",
            ["intent_faq_business_hours_outpatient"],
            [_unit("intent_faq_business_hours_outpatient", "business_hours", "outpatient")],
        ),
        _candidate(
            "chunk-3",
            "intent_faq_opening_time_outpatient",
            ["intent_faq_opening_time_outpatient"],
            [_unit("intent_faq_opening_time_outpatient", "opening_time", "outpatient")],
        ),
    ]

    result = await _build_result(candidates)

    assert result.registry.alias_to_canonical["intent_faq_hours_outpatient"] == (
        "intent_faq_business_hours_outpatient"
    )
    assert result.registry.alias_to_canonical["intent_faq_opening_time_outpatient"] == (
        "intent_faq_business_hours_outpatient"
    )
    assert len(result.registry.inventory) == 1

    item = _canonical_inventory_item(result, "intent_faq_business_hours_outpatient")
    assert item.support_count == 3
    assert set(item.aliases) == {
        "intent_faq_hours_outpatient",
        "intent_faq_business_hours_outpatient",
        "intent_faq_opening_time_outpatient",
    }


@pytest.mark.asyncio
async def test_different_branches_never_merge():
    candidates = [
        _candidate(
            "chunk-1",
            "intent_faq_business_hours_outpatient",
            ["intent_faq_business_hours_outpatient"],
            [_unit("intent_faq_business_hours_outpatient", "business_hours", "outpatient")],
        ),
        _candidate(
            "chunk-2",
            "intent_task_business_hours_outpatient",
            ["intent_task_business_hours_outpatient"],
            [
                _unit(
                    "intent_task_business_hours_outpatient",
                    "business_hours",
                    "outpatient",
                )
            ],
        ),
    ]

    result = await _build_result(candidates)

    assert len(result.registry.inventory) == 2
    assert result.registry.alias_to_canonical["intent_faq_business_hours_outpatient"] == (
        "intent_faq_business_hours_outpatient"
    )
    assert result.registry.alias_to_canonical["intent_task_business_hours_outpatient"] == (
        "intent_task_business_hours_outpatient"
    )


@pytest.mark.asyncio
async def test_specific_leaf_intent_beats_general_fallback_alias():
    candidates = [
        _candidate(
            "chunk-1",
            "intent_faq_general_ab12cd34_outpatient",
            ["intent_faq_general_ab12cd34_outpatient"],
            [
                _unit(
                    "intent_faq_general_ab12cd34_outpatient",
                    "general_ab12cd34",
                    "outpatient",
                )
            ],
        ),
        _candidate(
            "chunk-2",
            "intent_faq_business_hours_outpatient",
            ["intent_faq_business_hours_outpatient"],
            [_unit("intent_faq_business_hours_outpatient", "business_hours", "outpatient")],
        ),
    ]

    result = await _build_result(candidates)

    assert result.registry.alias_to_canonical["intent_faq_general_ab12cd34_outpatient"] == (
        "intent_faq_business_hours_outpatient"
    )


@pytest.mark.asyncio
async def test_apply_registry_rewrites_primary_candidates_and_units():
    builder = IntentRegistryBuilder()
    source_candidates = [
        _candidate(
            "chunk-1",
            "intent_faq_opening_time_outpatient",
            [
                "intent_faq_hours_outpatient",
                "intent_faq_business_hours_outpatient",
                "intent_faq_opening_time_outpatient",
            ],
            [
                _unit("intent_faq_opening_time_outpatient", "opening_time", "outpatient"),
                _unit("intent_faq_hours_outpatient", "hours", "outpatient"),
            ],
        ),
        _candidate(
            "chunk-2",
            "intent_faq_business_hours_outpatient",
            ["intent_faq_business_hours_outpatient"],
            [_unit("intent_faq_business_hours_outpatient", "business_hours", "outpatient")],
        ),
    ]

    result = await builder.build_registry(source_candidates)
    applied = builder.apply_registry_to_candidates(source_candidates, result.registry)

    candidate = applied[0]
    assert candidate.primary_leaf_intent == "intent_faq_business_hours_outpatient"
    assert candidate.candidate_leaf_intents == ["intent_faq_business_hours_outpatient"]
    assert {unit.leaf_intent for unit in candidate.normalized_units} == {
        "intent_faq_business_hours_outpatient"
    }
    assert {unit.topic_slug for unit in candidate.normalized_units} == {"business_hours"}
    assert {unit.detail_slug for unit in candidate.normalized_units} == {"outpatient"}


@pytest.mark.asyncio
async def test_apply_registry_recovers_candidate_list_from_primary_when_candidates_empty():
    builder = IntentRegistryBuilder()
    source_candidates = [
        _candidate(
            "chunk-1",
            "intent_faq_opening_time_outpatient",
            [],
            [_unit("intent_faq_opening_time_outpatient", "opening_time", "outpatient")],
        ),
        _candidate(
            "chunk-2",
            "intent_faq_business_hours_outpatient",
            ["intent_faq_business_hours_outpatient"],
            [_unit("intent_faq_business_hours_outpatient", "business_hours", "outpatient")],
        ),
    ]

    result = await builder.build_registry(source_candidates)
    applied = builder.apply_registry_to_candidates(source_candidates, result.registry)

    candidate = applied[0]
    assert candidate.primary_leaf_intent == "intent_faq_business_hours_outpatient"
    assert candidate.candidate_leaf_intents == ["intent_faq_business_hours_outpatient"]


@pytest.mark.asyncio
async def test_build_registry_uses_fallback_when_normalized_units_are_empty():
    result = await _build_result(
        [
            _candidate(
                "chunk-1",
                "intent_faq_business_hours_outpatient",
                [],
                [],
            )
        ]
    )

    item = _canonical_inventory_item(result, "intent_faq_business_hours_outpatient")
    assert item.branch_intent == "intent_faq"
    assert item.topic_slug == "business_hours"
    assert item.detail_slug == "outpatient"

    candidate = result.candidates[0]
    assert candidate.primary_leaf_intent == "intent_faq_business_hours_outpatient"
    assert candidate.candidate_leaf_intents == ["intent_faq_business_hours_outpatient"]
    assert candidate.normalized_units == []


@pytest.mark.asyncio
async def test_build_registry_preserves_safe_fallback_for_malformed_leaf_intent():
    result = await _build_result(
        [
            _candidate(
                "chunk-1",
                "invalid_leaf",
                [],
                [],
            )
        ]
    )

    item = _canonical_inventory_item(result, "invalid_leaf")
    assert item.branch_intent == "intent_faq"
    assert item.topic_slug == "general"
    assert item.detail_slug == "general"

    candidate = result.candidates[0]
    assert candidate.primary_leaf_intent == "invalid_leaf"
    assert candidate.candidate_leaf_intents == ["invalid_leaf"]
    assert candidate.normalized_units == []

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.agents.conversational.nodes.intent_router_llm_node import (
    intent_router_llm_node as intent_mod,
)


def _state(**overrides) -> dict:
    base = {
        "call_id": "call-int-1",
        "tenant_id": "tenant-xyz",
        "normalized_text": "please change my reservation",
        "knn_intent": "intent_task",
        "session_view": {"turn_count": 2},
    }
    base.update(overrides)
    return base


@pytest.fixture
def fake_llm(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(intent_mod, "_llm", fake)
    return fake


@pytest.mark.asyncio
async def test_valid_json_returns_parsed(fake_llm):
    fake_llm.generate = AsyncMock(
        return_value='{"primary_intent": "intent_faq", "secondary_intents": ["info"], "routing_reason": "faq question"}'
    )

    result = await intent_mod.intent_router_llm_node(_state())

    assert result == {
        "primary_intent": "intent_faq",
        "secondary_intents": ["info"],
        "routing_reason": "faq question",
    }


@pytest.mark.asyncio
async def test_json_with_surrounding_text(fake_llm):
    fake_llm.generate = AsyncMock(
        return_value='classification result: {"primary_intent": "intent_task", "secondary_intents": [], "routing_reason": "task"} thanks'
    )

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "task"


@pytest.mark.asyncio
async def test_missing_optional_fields_defaults(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_auth"}')

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_auth"
    assert result["secondary_intents"] == []
    assert result["routing_reason"] == "intent_router_llm"


@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_knn(fake_llm):
    fake_llm.generate = AsyncMock(return_value="{primary_intent: broken")

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_parse_error"


@pytest.mark.asyncio
async def test_unknown_intent_label_falls_back(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_weather"}')

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_parse_error"


@pytest.mark.asyncio
async def test_no_json_block_falls_back(fake_llm):
    fake_llm.generate = AsyncMock(return_value="no structured output")

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_parse_error"


@pytest.mark.asyncio
async def test_llm_timeout_falls_back_with_timeout_reason(monkeypatch, fake_llm):
    async def slow_generate(**_):
        await asyncio.sleep(1.0)
        return ""

    fake_llm.generate = slow_generate
    monkeypatch.setattr(intent_mod, "INTENT_ROUTER_TIMEOUT_SEC", 0.01)

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_timeout"


@pytest.mark.asyncio
async def test_llm_exception_falls_back_with_error_reason(fake_llm):
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("openai 500"))

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_error"


@pytest.mark.asyncio
async def test_no_knn_intent_escalates(fake_llm):
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("boom"))

    result = await intent_mod.intent_router_llm_node(_state(knn_intent=None))

    assert result["primary_intent"] == "intent_escalation"
    assert result["routing_reason"] == "knn_fallback_error_no_knn"


@pytest.mark.asyncio
async def test_leaf_knn_intent_can_fallback_to_branch(fake_llm):
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("boom"))

    result = await intent_mod.intent_router_llm_node(
        _state(knn_intent="intent_task_booking_outpatient")
    )

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_error"


@pytest.mark.asyncio
async def test_knn_top_k_branch_hint_is_included_in_prompt(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_faq"}')

    await intent_mod.intent_router_llm_node(
        _state(
            knn_top_k=[
                {
                    "intent_label": "intent_faq_business_hours_outpatient",
                    "branch_intent": "intent_faq",
                    "score": 0.82,
                    "example_text": "clinic hours",
                }
            ]
        )
    )

    user_message = fake_llm.generate.await_args.kwargs["user_message"]
    assert "intent_faq_business_hours_outpatient" in user_message
    assert "intent_faq" in user_message


@pytest.mark.asyncio
async def test_missing_knn_intent_key(fake_llm):
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("boom"))

    state = {
        "call_id": "c1",
        "tenant_id": "t1",
        "normalized_text": "hello",
        "session_view": {},
    }
    result = await intent_mod.intent_router_llm_node(state)

    assert result["primary_intent"] == "intent_escalation"


@pytest.mark.asyncio
async def test_session_view_none_safe(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_faq"}')

    state = {
        "call_id": "c1",
        "tenant_id": "t1",
        "normalized_text": "hello",
        "knn_intent": None,
        "session_view": None,
    }
    result = await intent_mod.intent_router_llm_node(state)

    assert result["primary_intent"] == "intent_faq"


@pytest.mark.asyncio
async def test_llm_call_uses_low_temperature_and_system_prompt(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_faq"}')

    await intent_mod.intent_router_llm_node(_state())

    call = fake_llm.generate.await_args
    assert call.kwargs["system_prompt"] == intent_mod.INTENT_ROUTER_SYSTEM_PROMPT
    assert call.kwargs["temperature"] <= 0.2
    assert "please change my reservation" in call.kwargs["user_message"]
    assert "intent_task" in call.kwargs["user_message"]

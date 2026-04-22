"""intent_router_llm_node 유닛 테스트.

모듈 레벨 _llm 을 AsyncMock 으로 교체.
검증 계층:
 - JSON 파싱: 정상 / 텍스트 혼재 / invalid / intent 라벨 무효 / 빈 응답
 - 타임아웃: wait_for 실발동
 - 예외: 일반 Exception
 - 폴백 체인: knn_intent 유효 → 사용 / 없음 → intent_escalation / invalid → intent_escalation
 - State 안정성: knn_intent 누락, session_view=None
"""
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
        "normalized_text": "예약 변경하고 싶어요",
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


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_json_returns_parsed(fake_llm):
    fake_llm.generate = AsyncMock(return_value=(
        '{"primary_intent": "intent_faq", "secondary_intents": ["info"], '
        '"routing_reason": "영업시간 질문"}'
    ))

    result = await intent_mod.intent_router_llm_node(_state())

    assert result == {
        "primary_intent": "intent_faq",
        "secondary_intents": ["info"],
        "routing_reason": "영업시간 질문",
    }


@pytest.mark.asyncio
async def test_json_with_surrounding_text(fake_llm):
    """LLM이 prose 포함해도 첫 JSON 블록을 추출."""
    fake_llm.generate = AsyncMock(return_value=(
        '분류 결과:\n{"primary_intent": "intent_task", "secondary_intents": [], '
        '"routing_reason": "예약 변경"}\n감사합니다.'
    ))

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "예약 변경"


@pytest.mark.asyncio
async def test_missing_optional_fields_defaults(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_auth"}')

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_auth"
    assert result["secondary_intents"] == []
    assert result["routing_reason"] == "llm_routed"


# ---------------------------------------------------------------------------
# 파싱 실패 → 폴백
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_knn(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{primary_intent: intent_faq, broken')

    result = await intent_mod.intent_router_llm_node(_state())

    # knn_intent="intent_task" 사용
    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_parse_error"


@pytest.mark.asyncio
async def test_unknown_intent_label_falls_back(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_weather"}')

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"  # knn_intent
    assert result["routing_reason"] == "knn_fallback_parse_error"


@pytest.mark.asyncio
async def test_no_json_block_falls_back(fake_llm):
    fake_llm.generate = AsyncMock(return_value='분류할 수 없는 응답입니다')

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_parse_error"


# ---------------------------------------------------------------------------
# 타임아웃
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_timeout_falls_back_with_timeout_reason(monkeypatch, fake_llm):
    async def slow_generate(**_):
        await asyncio.sleep(1.0)
        return ""

    fake_llm.generate = slow_generate
    monkeypatch.setattr(intent_mod, "INTENT_ROUTER_TIMEOUT_SEC", 0.01)

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"  # knn_intent
    assert result["routing_reason"] == "knn_fallback_timeout"


# ---------------------------------------------------------------------------
# 예외
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_exception_falls_back_with_error_reason(fake_llm):
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("openai 500"))

    result = await intent_mod.intent_router_llm_node(_state())

    assert result["primary_intent"] == "intent_task"
    assert result["routing_reason"] == "knn_fallback_error"


# ---------------------------------------------------------------------------
# 폴백 체인 — knn 없음 / 무효
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_knn_intent_escalates(fake_llm):
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("boom"))

    result = await intent_mod.intent_router_llm_node(_state(knn_intent=None))

    assert result["primary_intent"] == "intent_escalation"
    assert result["routing_reason"] == "knn_fallback_error_no_knn"


@pytest.mark.asyncio
async def test_invalid_knn_intent_escalates(fake_llm):
    """knn_intent가 VALID_INTENTS 밖이면 escalation으로 안전 라우팅."""
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("boom"))

    result = await intent_mod.intent_router_llm_node(_state(knn_intent="intent_unknown"))

    assert result["primary_intent"] == "intent_escalation"
    assert result["routing_reason"] == "knn_fallback_error_no_knn"


# ---------------------------------------------------------------------------
# State 안정성
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_knn_intent_key(fake_llm):
    """state에 knn_intent 키 자체가 없어도 crash 안 남."""
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("boom"))

    state = {
        "call_id": "c1",
        "tenant_id": "t1",
        "normalized_text": "안녕하세요",
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
        "normalized_text": "안녕하세요",
        "knn_intent": None,
        "session_view": None,
    }
    result = await intent_mod.intent_router_llm_node(state)

    assert result["primary_intent"] == "intent_faq"


# ---------------------------------------------------------------------------
# Interaction: 프롬프트 / temperature 가 약속대로 전달되는가
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_call_uses_low_temperature_and_system_prompt(fake_llm):
    fake_llm.generate = AsyncMock(return_value='{"primary_intent": "intent_faq"}')

    await intent_mod.intent_router_llm_node(_state())

    call = fake_llm.generate.await_args
    assert call.kwargs["system_prompt"] == intent_mod.INTENT_ROUTER_SYSTEM_PROMPT
    assert call.kwargs["temperature"] <= 0.2  # CLAUDE.md 규약
    assert "예약 변경하고 싶어요" in call.kwargs["user_message"]
    assert "intent_task" in call.kwargs["user_message"]  # KNN 후보가 user_message에 실려 감

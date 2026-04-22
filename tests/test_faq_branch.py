"""faq_branch_node 유닛 테스트.

모듈 레벨 _rag, _llm 을 AsyncMock 으로 교체.
검증:
 - RAG 성공 / 실패 / 임베딩 부재 시 스킵
 - LLM 성공 (strip / 공백-only 폴백)
 - LLM 타임아웃 폴백: rag top-1 / FALLBACK_MESSAGE
 - LLM 예외 → FALLBACK_MESSAGE (is_timeout=False)
 - user_message 에 RAG 청크가 번호 매겨 들어가는지
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.agents.conversational.nodes.faq_branch_node import (
    faq_branch_node as faq_mod,
)


def _state(**overrides) -> dict:
    base = {
        "call_id": "call-faq-1",
        "tenant_id": "tenant-xyz",
        "normalized_text": "영업시간이 어떻게 되나요",
        "query_embedding": [0.1, 0.2, 0.3],
    }
    base.update(overrides)
    return base


@pytest.fixture
def patched(monkeypatch):
    fake_rag = AsyncMock()
    fake_llm = AsyncMock()
    monkeypatch.setattr(faq_mod, "_rag", fake_rag)
    monkeypatch.setattr(faq_mod, "_llm", fake_llm)
    return fake_rag, fake_llm


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_hit_llm_success(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["오전 9시부터 오후 6시 영업", "공휴일 휴무"])
    fake_llm.generate = AsyncMock(return_value="오전 9시부터 오후 6시까지 영업합니다.")

    result = await faq_mod.faq_branch_node(_state())

    assert result == {
        "rag_results": ["오전 9시부터 오후 6시 영업", "공휴일 휴무"],
        "response_text": "오전 9시부터 오후 6시까지 영업합니다.",
        "response_path": "faq",
        "is_timeout": False,
    }


@pytest.mark.asyncio
async def test_llm_response_is_stripped(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk"])
    fake_llm.generate = AsyncMock(return_value="   안녕하세요.  \n")

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == "안녕하세요."


@pytest.mark.asyncio
async def test_llm_blank_response_falls_back(patched):
    """LLM이 공백만 반환해도 FALLBACK_MESSAGE 로 안전 치환."""
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk"])
    fake_llm.generate = AsyncMock(return_value="   \n   ")

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == faq_mod.FALLBACK_MESSAGE
    assert result["is_timeout"] is False


# ---------------------------------------------------------------------------
# RAG 스킵 / 실패
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_embedding_skips_rag_search(patched):
    """cache_node 에서 embed 실패 → query_embedding=[] → RAG 호출 금지."""
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock()  # 호출되면 안 됨
    fake_llm.generate = AsyncMock(return_value="답변")

    result = await faq_mod.faq_branch_node(_state(query_embedding=[]))

    assert result["rag_results"] == []
    assert result["response_text"] == "답변"
    fake_rag.search.assert_not_called()


@pytest.mark.asyncio
async def test_missing_embedding_key_skips_rag(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock()
    fake_llm.generate = AsyncMock(return_value="답변")

    state = {
        "call_id": "c1",
        "tenant_id": "t1",
        "normalized_text": "질문",
    }
    result = await faq_mod.faq_branch_node(state)

    assert result["rag_results"] == []
    fake_rag.search.assert_not_called()


@pytest.mark.asyncio
async def test_rag_exception_still_calls_llm(patched):
    """RAG 가 터져도 LLM 은 호출 (참고자료 없음 상태로)."""
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(side_effect=ConnectionError("chroma down"))
    fake_llm.generate = AsyncMock(return_value="일반 응답")

    result = await faq_mod.faq_branch_node(_state())

    assert result["rag_results"] == []
    assert result["response_text"] == "일반 응답"
    assert result["is_timeout"] is False
    fake_llm.generate.assert_awaited_once()


# ---------------------------------------------------------------------------
# LLM 타임아웃 폴백
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_timeout_with_rag_uses_top1_chunk(monkeypatch, patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["첫번째 청크", "두번째 청크"])

    async def slow_generate(**_):
        await asyncio.sleep(1.0)
        return ""

    fake_llm.generate = slow_generate
    monkeypatch.setattr(faq_mod, "FAQ_LLM_TIMEOUT_SEC", 0.01)

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == "첫번째 청크"
    assert result["response_path"] == "faq"
    assert result["is_timeout"] is True
    assert result["rag_results"] == ["첫번째 청크", "두번째 청크"]


@pytest.mark.asyncio
async def test_llm_timeout_without_rag_uses_fallback_message(monkeypatch, patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=[])

    async def slow_generate(**_):
        await asyncio.sleep(1.0)

    fake_llm.generate = slow_generate
    monkeypatch.setattr(faq_mod, "FAQ_LLM_TIMEOUT_SEC", 0.01)

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == faq_mod.FALLBACK_MESSAGE
    assert result["is_timeout"] is True
    assert result["rag_results"] == []


# ---------------------------------------------------------------------------
# LLM 일반 예외
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_exception_returns_fallback_not_timeout(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk A"])
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("openai 500"))

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == faq_mod.FALLBACK_MESSAGE
    assert result["is_timeout"] is False  # 타임아웃이 아니라 에러
    assert result["rag_results"] == ["chunk A"]


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_search_called_with_correct_args(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=[])
    fake_llm.generate = AsyncMock(return_value="응답")

    await faq_mod.faq_branch_node(_state())

    fake_rag.search.assert_awaited_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        tenant_id="tenant-xyz",
        top_k=faq_mod.RAG_TOP_K,
    )


@pytest.mark.asyncio
async def test_user_message_numbers_rag_chunks(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["청크A", "청크B"])
    fake_llm.generate = AsyncMock(return_value="응답")

    await faq_mod.faq_branch_node(_state())

    user_msg = fake_llm.generate.await_args.kwargs["user_message"]
    assert "[1] 청크A" in user_msg
    assert "[2] 청크B" in user_msg
    assert "영업시간이 어떻게 되나요" in user_msg


@pytest.mark.asyncio
async def test_user_message_handles_empty_rag(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=[])
    fake_llm.generate = AsyncMock(return_value="응답")

    await faq_mod.faq_branch_node(_state())

    user_msg = fake_llm.generate.await_args.kwargs["user_message"]
    assert "(참고 자료 없음)" in user_msg


@pytest.mark.asyncio
async def test_llm_call_uses_low_temperature_and_system_prompt(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk"])
    fake_llm.generate = AsyncMock(return_value="응답")

    await faq_mod.faq_branch_node(_state())

    kwargs = fake_llm.generate.await_args.kwargs
    assert kwargs["system_prompt"] == faq_mod.FAQ_SYSTEM_PROMPT
    assert kwargs["temperature"] <= 0.2

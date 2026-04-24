import asyncio
from unittest.mock import AsyncMock

import pytest

from app.agents.conversational.nodes.faq_branch_node import faq_branch_node as faq_mod


def _state(**overrides) -> dict:
    base = {
        "call_id": "call-faq-1",
        "tenant_id": "tenant-xyz",
        "normalized_text": "what are your business hours",
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


@pytest.mark.asyncio
async def test_rag_hit_llm_success(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["open from 9 to 6", "closed on holidays"])
    fake_llm.generate = AsyncMock(return_value="We are open from 9 to 6.")

    result = await faq_mod.faq_branch_node(_state())

    assert result == {
        "rag_results": ["open from 9 to 6", "closed on holidays"],
        "response_text": "We are open from 9 to 6.",
        "response_path": "faq",
        "is_timeout": False,
        "is_fallback": False,
    }


@pytest.mark.asyncio
async def test_llm_response_is_stripped(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk"])
    fake_llm.generate = AsyncMock(return_value="   hello  \n")

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == "hello"


@pytest.mark.asyncio
async def test_llm_blank_response_falls_back(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk"])
    fake_llm.generate = AsyncMock(return_value="   \n   ")

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == faq_mod.FALLBACK_MESSAGE
    assert result["is_timeout"] is False
    assert result["is_fallback"] is True


@pytest.mark.asyncio
async def test_empty_embedding_skips_rag_search(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock()
    fake_llm.generate = AsyncMock(return_value="answer")

    result = await faq_mod.faq_branch_node(_state(query_embedding=[]))

    assert result["rag_results"] == []
    assert result["response_text"] == "answer"
    assert result["is_fallback"] is True
    fake_rag.search.assert_not_called()


@pytest.mark.asyncio
async def test_missing_embedding_key_skips_rag(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock()
    fake_llm.generate = AsyncMock(return_value="answer")

    state = {
        "call_id": "c1",
        "tenant_id": "t1",
        "normalized_text": "question",
    }
    result = await faq_mod.faq_branch_node(state)

    assert result["rag_results"] == []
    fake_rag.search.assert_not_called()


@pytest.mark.asyncio
async def test_rag_exception_still_calls_llm(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(side_effect=ConnectionError("chroma down"))
    fake_llm.generate = AsyncMock(return_value="generic answer")

    result = await faq_mod.faq_branch_node(_state())

    assert result["rag_results"] == []
    assert result["response_text"] == "generic answer"
    assert result["is_timeout"] is False
    assert result["is_fallback"] is True
    fake_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_timeout_with_rag_uses_top1_chunk(monkeypatch, patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["first chunk", "second chunk"])

    async def slow_generate(**_):
        await asyncio.sleep(1.0)
        return ""

    fake_llm.generate = slow_generate
    monkeypatch.setattr(faq_mod, "FAQ_LLM_TIMEOUT_SEC", 0.01)

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == "first chunk"
    assert result["response_path"] == "faq"
    assert result["is_timeout"] is True
    assert result["rag_results"] == ["first chunk", "second chunk"]
    assert result["is_fallback"] is False


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
    assert result["is_fallback"] is True


@pytest.mark.asyncio
async def test_llm_exception_returns_fallback_not_timeout(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk A"])
    fake_llm.generate = AsyncMock(side_effect=RuntimeError("openai 500"))

    result = await faq_mod.faq_branch_node(_state())

    assert result["response_text"] == faq_mod.FALLBACK_MESSAGE
    assert result["is_timeout"] is False
    assert result["rag_results"] == ["chunk A"]
    assert result["is_fallback"] is True


@pytest.mark.asyncio
async def test_rag_search_called_with_correct_args(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=[])
    fake_llm.generate = AsyncMock(return_value="answer")

    await faq_mod.faq_branch_node(_state())

    fake_rag.search.assert_awaited_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        tenant_id="tenant-xyz",
        top_k=faq_mod.RAG_TOP_K,
    )


@pytest.mark.asyncio
async def test_user_message_numbers_rag_chunks(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunkA", "chunkB"])
    fake_llm.generate = AsyncMock(return_value="answer")

    await faq_mod.faq_branch_node(_state())

    user_msg = fake_llm.generate.await_args.kwargs["user_message"]
    assert "[1] chunkA" in user_msg
    assert "[2] chunkB" in user_msg
    assert "what are your business hours" in user_msg


@pytest.mark.asyncio
async def test_user_message_handles_empty_rag(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=[])
    fake_llm.generate = AsyncMock(return_value="answer")

    await faq_mod.faq_branch_node(_state())

    user_msg = fake_llm.generate.await_args.kwargs["user_message"]
    assert "(no reference material)" in user_msg


@pytest.mark.asyncio
async def test_llm_call_uses_low_temperature_and_system_prompt(patched):
    fake_rag, fake_llm = patched
    fake_rag.search = AsyncMock(return_value=["chunk"])
    fake_llm.generate = AsyncMock(return_value="answer")

    await faq_mod.faq_branch_node(_state())

    kwargs = fake_llm.generate.await_args.kwargs
    assert kwargs["system_prompt"] == faq_mod.FAQ_SYSTEM_PROMPT
    assert kwargs["temperature"] <= 0.2

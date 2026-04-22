"""Scenario B — faq_branch_node end-to-end (Chroma + OpenAI).

Chroma tenant 컬렉션에 문서 1개 시드 → faq_branch_node 실행 → 실제 RAG + GPT-4o-mini
응답이 계약(response_path="faq", rag_results 전달, is_timeout 상태)대로 오는지 검증.

실행:
  venv/Scripts/python -m pytest tests/integration/test_faq_e2e.py -v -s

사전 조건:
  make up 으로 chromadb 기동 + .env OPENAI_API_KEY 설정.
비용: gpt-4o-mini 1회 호출 (~1~2센트 미만).
"""
import uuid

import pytest
import pytest_asyncio

from app.agents.conversational.nodes.faq_branch_node import (
    faq_branch_node as faq_mod,
)
from app.services.embedding.mock import MockEmbeddingService
from app.services.rag.chroma import ChromaRAGService

pytestmark = pytest.mark.integration

_FAQ_DOC = (
    "서울중앙병원 영업시간은 평일 오전 9시부터 오후 6시, 토요일은 오전 9시부터 오후 1시, "
    "일요일과 공휴일은 휴무입니다."
)
_FAQ_QUERY = "영업시간이 어떻게 되나요"


@pytest_asyncio.fixture(autouse=True)
async def _reset_module_singletons(monkeypatch):
    """AsyncOpenAI client 는 import 시점 event loop 에 바인딩된다.
    pytest-asyncio strict 모드의 per-test loop 와 충돌하므로 매 테스트마다 리셋."""
    from app.services.llm.gpt4o_mini import GPT4OMiniService
    from app.services.rag.chroma import ChromaRAGService
    monkeypatch.setattr(faq_mod, "_llm", GPT4OMiniService())
    monkeypatch.setattr(faq_mod, "_rag", ChromaRAGService())
    yield


@pytest_asyncio.fixture
async def seeded_tenant():
    """고유 tenant 컬렉션에 FAQ 문서 1개 시드 + 종료 시 컬렉션 삭제."""
    tenant_id = f"integtest-{uuid.uuid4().hex[:12]}"
    rag = ChromaRAGService()
    embedder = MockEmbeddingService()

    doc_embedding = await embedder.embed(_FAQ_DOC)
    await rag.upsert(
        doc_id="faq-1",
        content=_FAQ_DOC,
        embedding=doc_embedding,
        tenant_id=tenant_id,
        metadata={"document_id": "seed-doc"},
    )

    yield tenant_id, embedder

    # cleanup: 컬렉션 자체를 삭제 (테스트 간 isolate)
    collection_name = rag._collection_name(tenant_id)
    try:
        rag._client.delete_collection(collection_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Happy path — RAG 매칭 문서 + 실제 GPT-4o-mini 응답
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_faq_branch_end_to_end(seeded_tenant):
    tenant_id, embedder = seeded_tenant
    query_embedding = await embedder.embed(_FAQ_QUERY)

    result = await faq_mod.faq_branch_node({
        "call_id": f"integ-{uuid.uuid4().hex[:6]}",
        "tenant_id": tenant_id,
        "normalized_text": _FAQ_QUERY,
        "query_embedding": query_embedding,
    })

    # --- 공통 계약 ---
    assert result["response_path"] == "faq"
    assert isinstance(result["response_text"], str) and result["response_text"].strip()
    assert isinstance(result["rag_results"], list)

    # --- RAG 검증: 시드한 문서가 top-K 안에 실제로 들어와야 함 ---
    # MockEmbedding은 hash(text) 기반이라 query_embedding != doc_embedding.
    # Chroma의 cosine similarity가 가까운 항목을 올리긴 하지만, 우리가 seed한 유일한
    # 문서이므로 top-1 이 무조건 _FAQ_DOC 이어야 한다.
    assert _FAQ_DOC in result["rag_results"]

    # --- 타임아웃 분기 기록 ---
    # is_timeout=True 면 rag_results 의 첫 청크가 response_text 로 폴백됐다는 뜻.
    # is_timeout=False 면 GPT-4o-mini 가 2초 내 응답했다는 뜻.
    # 어느 쪽이든 계약상 유효하므로 assert 하지 않고 출력만 한다.
    print(f"\n[FAQ E2E] is_timeout={result['is_timeout']}")
    print(f"[FAQ E2E] response_text={result['response_text'][:200]}")

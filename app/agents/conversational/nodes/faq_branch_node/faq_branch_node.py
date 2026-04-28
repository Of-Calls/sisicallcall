from app.agents.conversational.state import CallState
from app.agents.conversational.utils.stall import FALLBACK_MESSAGE, _run_with_stall
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.base import BaseRAGService
from app.services.rag.chroma import ChromaRAGService
from app.utils.logger import get_logger

# FAQ 브랜치 — RAG + GPT-4o-mini 3초 하드컷 (RFC 001 v0.2 §6.8)
# stall trigger 1초, TTSOutputChannel 경유

logger = get_logger(__name__)

_rag: BaseRAGService = ChromaRAGService()
_llm: BaseLLMService = GPT4OMiniService()

# 2026-04-28: 3.0 → 5.0 상향. 첫 turn LLM 응답이 3초 초과해 hardcut → raw chunk
# fallback 으로 빠지는 사례 빈발 (165225.log:69, 90). stall scheduler 가 1.5초에
# "잠시만요" 이미 발화하므로 hardcut 까지 시간을 늘려도 사용자 체감 동일.
FAQ_LLM_TIMEOUT_SEC = 5.0
RAG_TOP_K = 8           # ChromaDB 후보 풀 (re-rank 전)
RAG_RERANK_TOP_N = 3    # hybrid score 정렬 후 LLM 에 전달할 상위 N
# 2026-04-28: ChromaDB cosine distance 가 이 값 초과인 chunk 는 LLM 입력에서 제외.
# 변경 이력:
#   0.85 → opendataloader chunks (평균 898자) 가 너무 길어 BGE-M3 임베딩 dilute,
#         "메뉴가 뭐가 있어요" 같은 짧은 일반 질문이 chunk_4 (메뉴) distance 0.91~1.04
#         로 임계값 못 넘는 사례 발생 (195111.log:117, 199, 221).
#   0.85 → 0.95 — 임계값 완화 + chunk 700자 분할로 distance 자체도 줄어들 것.
RAG_DISTANCE_THRESHOLD = 0.95
# Hybrid retrieval: query 와 chunk 의 llm_keywords 가 substring 매칭되면 거리 차감.
# 매칭 keyword 1개당 -0.05 — 너무 크면 distance 무시, 너무 작으면 효과 미미. 측정 후 조정.
RAG_KEYWORD_BONUS = 0.05


def _hybrid_score(distance: float, query: str, llm_keywords: str) -> tuple[float, list[str]]:
    """(hybrid_score, matched_keywords) — distance 에서 keyword overlap 만큼 차감."""
    keywords = [k.strip() for k in (llm_keywords or "").split(",") if k.strip()]
    if not keywords or not query:
        return distance, []
    matched = [kw for kw in keywords if kw in query]
    return max(distance - RAG_KEYWORD_BONUS * len(matched), 0.0), matched

# TODO(agents.md 이관): 담당자 배정 후 프롬프트를 agents.md 로 이관
FAQ_SYSTEM_PROMPT = """당신은 전화 고객센터의 FAQ 응답 AI입니다. 사용자는 음성으로 답변을 듣습니다.
반드시 아래 규칙을 지키세요.
1) 제공된 '참고 자료' 범위 안에서만 답변합니다.
2) **참고 자료에 답이 없을 때 (RAG miss) 절대 첫 시도부터 "담당자에게 연결" 이라고 답하지 마세요.**
   입력에 'rag_miss_count' 가 포함되어 있고 참고 자료가 부족할 때:
     - rag_miss_count == 1 (첫 시도): "제가 그 부분은 잘 모르겠습니다. 어떤 부분이 궁금하신가요?" 같이
       모른다는 사실 + 짧은 재질문 한 문장으로만 응답하세요. 정보를 만들지 마세요.
       (★음성 통화 응답이므로 사무적·기계적 표현 절대 금지. "제공되지 않았습니다",
       "확인이 어렵습니다", "정보가 없습니다" 같은 시스템 응답 어투 사용 금지.
       자연스러운 한국어 대화체로 답하세요: "잘 모르겠어요", "그 부분은 안내가 어려워서요" 같이.
       나쁜 예: "교통 안내에 대한 정보는 제공되지 않았습니다. 어떤 부분이 궁금하신가요?"
       좋은 예: "교통편은 잘 모르겠어요. 어떤 부분이 궁금하신가요?")
     - rag_miss_count >= 2 (반복): "제가 안내드릴 수 있는 분야는 {available_categories} 입니다. 어떤 정보가 필요하신가요?"
       형태로 입력의 'available_categories' 를 그대로 자연스럽게 안내하세요.
       만약 available_categories 가 비어 있으면 일반적인 옵션 ("위치, 진료시간, 예약 등") 으로 안내하세요.
3) **★최우선: 응답은 반드시 100자 이내, 1~2문장으로 끝낸다. 100자가 넘으면 안 된다.**
4) 표/목록을 풀어서 길게 나열하지 말고, 핵심만 한 문장으로 요약한다.
   (예: "평일 09:00~17:30, 토요일 09:00~12:00 운영됩니다." — 점심시간/예외사항은 묻지 않으면 생략)
5) 한국어 존댓말로 자연스럽게 답한다.
6) 참고 자료에 없는 정보를 추측하거나 생성하지 않는다.
7) RAG miss 응답은 1~2문장의 후속 질문으로 끝낸다. 정보를 만들지 않는다."""


def _compose_user_message(
    normalized_text: str,
    rag_results: list[str],
    rag_miss_count: int = 0,
    available_categories: list[str] | None = None,
) -> str:
    if rag_results:
        joined = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(rag_results))
    else:
        joined = "(참고 자료 없음)"
    cats = available_categories or []
    cats_line = ", ".join(cats) if cats else "(없음)"
    return (
        f"참고 자료:\n{joined}\n\n"
        f"rag_miss_count: {rag_miss_count}\n"
        f"available_categories: {cats_line}\n\n"
        f"고객 질문: {normalized_text}"
    )


def _short_chunk_id(full_id: str) -> str:
    """`{document_uuid}_chunk_{n}` → `chunk_{n}` (UUID 생략, 로그용)."""
    if "_chunk_" in full_id:
        return "chunk_" + full_id.rsplit("_chunk_", 1)[-1]
    return full_id[:12]


def _pick_stall_msg(state: CallState) -> str:
    """CallState 에서 FAQ 대기 멘트 선택. 없으면 general → 하드코딩 순으로 fallback."""
    msgs = state.get("stall_messages") or {}
    return msgs.get("faq") or msgs.get("general") or "잠시만요, 확인해 드리겠습니다."


async def faq_branch_node(state: CallState) -> dict:
    call_id = state["call_id"]
    query_embedding = state.get("query_embedding") or []
    prev_miss_count = state.get("rag_miss_count", 0)

    # RAG 검색 — 임베딩 없으면 스킵 (cache_node 에서 실패한 경우)
    rag_results: list[str] = []
    if query_embedding:
        try:
            rag_meta = await _rag.search_with_meta(
                query_embedding=query_embedding,
                tenant_id=state["tenant_id"],
                top_k=RAG_TOP_K,
            )
            # Hybrid score: distance 에서 query 와 chunk metadata 의 llm_keywords 매칭만큼 차감.
            # ChromaDB metadata.llm_keywords 는 chunking 시 LLM 추출 ("메뉴, 한정식, 코스" 같은 string).
            query_text = state.get("normalized_text", "") or state.get("raw_transcript", "") or ""
            for r in rag_meta:
                kws = (r.get("metadata") or {}).get("llm_keywords", "")
                dist = r.get("distance") if r.get("distance") is not None else 1.0
                hybrid, matched = _hybrid_score(dist, query_text, kws)
                r["_hybrid"] = hybrid
                r["_matched_kw"] = matched

            # hybrid score 오름차순 정렬 → 상위 RAG_RERANK_TOP_N 추출
            rag_meta.sort(key=lambda r: r.get("_hybrid", 1.0))
            top_n = rag_meta[:RAG_RERANK_TOP_N]

            # distance threshold 는 원본 distance 기준 filter (hybrid 가 아닌)
            rag_results = [
                r["document"] for r in top_n
                if r.get("document")
                and (r["distance"] if r.get("distance") is not None else 1.0)
                <= RAG_DISTANCE_THRESHOLD
            ]

            if top_n:
                summary = " | ".join(
                    "%s@d=%.3f/h=%.3f%s%s:%r" % (
                        _short_chunk_id(r.get("id") or ""),
                        r.get("distance") if r.get("distance") is not None else float("nan"),
                        r.get("_hybrid", float("nan")),
                        ("(kw=" + "+".join(r["_matched_kw"]) + ")") if r.get("_matched_kw") else "",
                        ""
                        if (r.get("distance") if r.get("distance") is not None else 1.0)
                        <= RAG_DISTANCE_THRESHOLD
                        else "(filtered)",
                        (r.get("document") or "")[:60],
                    )
                    for r in top_n
                )
                logger.info(
                    "rag top_k=%d rerank=%d kept=%d threshold=%.2f bonus=%.2f call_id=%s tenant_id=%s top_n=[%s]",
                    RAG_TOP_K, RAG_RERANK_TOP_N, len(rag_results),
                    RAG_DISTANCE_THRESHOLD, RAG_KEYWORD_BONUS,
                    call_id, state["tenant_id"], summary,
                )
        except Exception as e:
            logger.error("rag search failed call_id=%s: %s", call_id, e)

    # LLM 분기 입력용 — RAG hit 이면 0, miss 면 prev + 1.
    # 첫 RAG miss 면 1 (모른다 + 재질문), 2+ 이면 카테고리 안내.
    incoming_miss_count = prev_miss_count + 1 if not rag_results else 0
    # available_categories: Commit 2 에서 call.py 가 Redis 조회 후 주입. 없으면 빈 list.
    available_categories = state.get("available_categories") or []  # type: ignore[typeddict-item]
    user_message = _compose_user_message(
        state["normalized_text"],
        rag_results,
        rag_miss_count=incoming_miss_count,
        available_categories=available_categories,
    )

    # LLM 호출 + stall trigger race — RFC 001 v0.2 §6.5
    # max_tokens=150 — 100자 응답 + 안전 마진 (한국어 1자 ≈ 1~2 토큰)
    # 실측: 300 → bytes 137~191KB (17~24초 음성). 150 으로 축소해 7~10초 응답.
    response_text, is_timeout = await _run_with_stall(
        coro=_llm.generate(
            system_prompt=FAQ_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1,
            max_tokens=150,
        ),
        call_id=call_id,
        stall_msg=_pick_stall_msg(state),
        stall_audio_field="faq",
        delay=state.get("stall_delay_sec", 1.0),
        hardcut_sec=FAQ_LLM_TIMEOUT_SEC,
        rag_results=rag_results,
        fallback_text=FALLBACK_MESSAGE,
    )

    # LLM 이 공백만 반환한 엣지 케이스도 FALLBACK 으로 보정
    if not response_text:
        response_text = FALLBACK_MESSAGE

    # fallback 판정 — Semantic Cache 저장 차단 신호 (cache_store_node 가 검사)
    is_fallback = (not rag_results) or (response_text == FALLBACK_MESSAGE)
    # 다음 turn 으로 넘길 누적 카운터 — RAG miss 이거나 fallback 응답이면 +1, 정상이면 reset
    new_miss_count = prev_miss_count + 1 if is_fallback else 0

    return {
        "rag_results": rag_results,
        "response_text": response_text,
        "response_path": "faq",
        "is_timeout": is_timeout,
        "is_fallback": is_fallback,
        "rag_miss_count": new_miss_count,
    }

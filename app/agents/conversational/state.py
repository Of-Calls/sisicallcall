from typing import NotRequired, Optional, TypedDict


class CallState(TypedDict):
    # 식별자
    call_id: str
    tenant_id: str
    turn_index: int

    # 오디오 입력
    audio_chunk: bytes

    # VAD / 화자 검증
    is_speech: bool
    is_speaker_verified: bool

    # STT 결과
    raw_transcript: str
    normalized_text: str

    # 임베딩 및 Cache 분기
    query_embedding: list[float]
    cache_hit: bool

    # 라우팅
    knn_intent: Optional[str]
    knn_confidence: float
    primary_intent: Optional[str]       # "intent_faq" | "intent_task" | "intent_auth" | "intent_escalation"
    secondary_intents: list[str]
    routing_reason: Optional[str]

    # 세션 view (Redis 에서 로드한 당 턴 관점 정보)
    session_view: dict

    # FAQ 브랜치 내부용
    rag_results: list[str]

    # 최종 응답
    response_text: str
    response_path: str                  # "cache" | "faq" | "task" | "auth" | "escalation"

    # Reviewer
    reviewer_applied: bool
    reviewer_verdict: Optional[str]     # "pass" | "revise"

    # 에러 / 타임아웃
    is_timeout: bool
    error: Optional[str]

    # 대기 멘트 (RFC 001 v0.2) — run_turn 진입 시 tenant settings 에서 pre-load
    # 노드는 .get() 로 안전 접근. 키 부재 시 _run_with_stall 이 하드코딩 기본값 사용.
    stall_messages: NotRequired[dict]   # {"general": "잠시만요...", "faq": "...", ...}
    stall_delay_sec: NotRequired[float] # 기본 1.0

    # cache_store_node 동작 결과 (관측·디버깅용)
    cache_stored: NotRequired[bool]

    # 응답이 fallback 메시지인지 (Semantic Cache 저장 차단용).
    # 브랜치 노드가 명시적으로 설정. RAG miss / LLM 고정 fallback 텍스트 등.
    is_fallback: NotRequired[bool]

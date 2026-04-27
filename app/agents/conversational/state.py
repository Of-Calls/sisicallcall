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
    primary_intent: Optional[str]       # "intent_faq" | "intent_task" | "intent_auth" | "intent_clarify" | "intent_escalation"
    secondary_intents: list[str]
    routing_reason: Optional[str]

    # 세션 view (Redis 에서 로드한 당 턴 관점 정보)
    session_view: dict

    # FAQ 브랜치 내부용
    rag_results: list[str]

    # 최종 응답
    response_text: str
    response_path: str                  # "cache" | "faq" | "task" | "auth" | "clarify" | "escalation" | "repeat"

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

    # 빈 STT 연속 횟수 — call.py 에서 주입, stt_node 에서 증가.
    # N회 초과 시 escalation 처리 (안내 멘트 → 상담원 연결).
    empty_stt_count: NotRequired[int]

    # voiceprint 등록 완료 여부 — enrollment_node 에서 관리 (관측용)
    enrollment_done: NotRequired[bool]

    # Intent Router LLM 이 모호한 발화에 대해 생성한 역질문 (intent_clarify 시만 채워짐).
    # clarify_branch_node 가 그대로 response_text 로 사용. 다른 intent 일 때는 키 자체가 없음.
    clarify_question: NotRequired[str]

    # Barge-in (TTS 재생 중 사용자 발화로 직전 응답이 끊긴 경우).
    # call.py 가 turn 시작 시 채워 넣고, intent_router_llm 이 컨텍스트로 활용한다.
    # is_bargein=True 면 interrupted_response_text 에 끊긴 응답 원문이 보존됨.
    is_bargein: NotRequired[bool]
    interrupted_response_text: NotRequired[str]

    # FAQ RAG miss 누적 횟수 — call.py 가 turn 간 누적 후 state 주입.
    # faq_branch_node 가 LLM 분기 (1회: 모른다+재질문, 2+회: 카테고리 안내) 에 사용.
    # response_path != "faq" 인 turn 은 0 으로 reset.
    rag_miss_count: NotRequired[int]

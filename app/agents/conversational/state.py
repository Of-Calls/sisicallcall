from typing import NotRequired, Optional, TypedDict


class CallState(TypedDict):
    """통화 응대 graph 상태 — 텍스트 워크플로우 전용 (2026-04-30 구조 개편).

    audio 도메인 (VAD / 화자검증 / STT / enrollment) 은 call.py 가 graph 진입 전에
    처리. graph 는 텍스트 + 컨텍스트만 받아 cache → rag_probe → intent → branch →
    cache_store → tts 흐름 담당. graph 가 audio_chunk: bytes 를 들고 다니던 이중
    책임 구조 제거 + 단위 테스트 시 mock audio 불필요.
    """
    # 식별자
    call_id: str
    tenant_id: str
    turn_index: int

    # STT 결과 — call.py 가 graph 진입 전 streaming flush + (fallback) prerecorded
    # 까지 마친 후 채워서 넘김. 빈 문자열이면 graph 진입 자체가 발생하지 않음.
    raw_transcript: str
    normalized_text: str

    # 임베딩 및 Cache 분기
    query_embedding: list[float]
    cache_hit: bool

    # 라우팅
    primary_intent: Optional[str]       # "intent_faq" | "intent_task" | "intent_auth" | "intent_clarify" | "intent_escalation"

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

    # 타임아웃
    is_timeout: bool

    # 대기 멘트 (RFC 001 v0.2) — run_turn 진입 시 tenant settings 에서 pre-load
    # 노드는 .get() 로 안전 접근. 키 부재 시 _run_with_stall 이 하드코딩 기본값 사용.
    stall_messages: NotRequired[dict]   # {"general": "잠시만요...", "faq": "...", ...}
    stall_delay_sec: NotRequired[float] # 기본 1.0

    # cache_store_node 동작 결과 (관측·디버깅용)
    cache_stored: NotRequired[bool]

    # 응답이 fallback 메시지인지 (Semantic Cache 저장 차단용).
    # 브랜치 노드가 명시적으로 설정. RAG miss / LLM 고정 fallback 텍스트 등.
    is_fallback: NotRequired[bool]

    # 빈 STT 연속 횟수 — call.py 가 직접 관리 + 누적. graph 는 사용 안 하지만
    # 후속 escalation 분기 (N회 초과 → 상담원 연결) 위해 통과 시 보존.
    empty_stt_count: NotRequired[int]

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

    # tenant 가용 RAG 카테고리 (자연어, LLM 정제) — call.py 가 start 이벤트에서 Redis 조회 후
    # 매 turn state 에 주입. faq_branch_node 의 LLM 이 rag_miss_count >= 2 일 때 안내 멘트 생성.
    # 빈 list 면 LLM 이 일반 옵션 ("위치, 진료시간, 예약 등") 으로 fallback.
    available_categories: NotRequired[list[str]]

    # RAG probe 신호 (cache miss → rag_probe_node 가 채움 → intent_router_llm 이 활용).
    # dict 또는 None. dict 일 때 키:
    #   top_distance(float), matched_keywords(list[str]), top_topic(str),
    #   top_title(str), top_chunk_id(str), is_auth(bool)
    # 임베딩 미보유·검색 오류·결과 0 시 None.
    rag_probe: NotRequired[Optional[dict]]

    # 인증 브랜치 상태 — call.py 가 session_view 경유로 턴 간 유지.
    # Turn 1: auth_branch_node 가 확인 질문 발화 후 True 설정.
    # Turn 2: 사용자 응답 처리 후 False 로 리셋.
    auth_pending: NotRequired[bool]

    # SMS 발송 완료 플래그 — 관측·로깅용. auth_branch_node 가 설정.
    auth_sms_sent: NotRequired[bool]

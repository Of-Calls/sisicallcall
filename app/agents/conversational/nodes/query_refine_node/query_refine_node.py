"""query_refine_node — Option γ 설계: 결정론적 STT 정규화 + ambiguity gate.

역할 (graph 상 위치: cache miss → rag_probe 이후, intent_router_llm 진입 전):
  1. STT raw_transcript 의 선두 필러(음/어/그/저) 제거 + tenant 키텀 띄어쓰기 정정.
  2. 정제 결과(refined_text)를 normalized_text 에도 덮어써 하위 노드(intent_router,
     faq_branch 등)가 자동으로 개선된 텍스트를 사용하도록 한다.
  3. Ambiguity gate:
     - refined_text 가 너무 짧거나 필러 단독이면서 rag_probe 신호도 약한 경우
       is_ambiguous=True 반환 → graph 가 clarify_author_node 로 라우팅
     - 신호가 명확하면 is_ambiguous=False + fire-and-forget ack stall 발화
       ("사용자님 질문을 이해했어요. 잠시만요.") → intent_router_llm 진입
  4. MAX_CLARIFY_TURNS 소진 → primary_intent="intent_escalation" 강제 반환.

LLM 없음. 목표 레이턴시 ~50ms.
"""
import asyncio
import re
import time
from typing import Optional

from app.agents.conversational.state import CallState
from app.services.tts.channel import tts_channel
from app.utils.logger import get_logger

logger = get_logger(__name__)

# MAX_CLARIFY_TURNS — intent_router_llm_node 에서 이관 (그쪽 슬림화 작업 스트림과 협의).
# clarify 역질문 턴 상한. 초과 시 바로 escalation 분기.
MAX_CLARIFY_TURNS = 6

# ambiguity gate 파라미터
_AMBIGUITY_DISTANCE_THRESHOLD = 0.95   # rag_probe.top_distance > this → 잠재적 모호
_AMBIGUITY_MAX_LEN = 4                 # refined_text 길이 ≤ this → 잠재적 모호

# 단독으로 왔을 때 모호 신호인 필러 집합 (정제 후 텍스트 전체가 이 중 하나이면 모호)
_FILLER_ONLY_PATTERNS: set[str] = {
    "네", "응", "어", "음", "글쎄", "그게", "아", "예", "맞아", "아니", "아니요",
}

# 선두 필러 제거 정규식 — 음/어/그/저. 다음 문자가 반드시 공백/문자열 끝이어야 함
# (lookahead) — 그렇지 않으면 "그게", "어떻게" 같은 정상 단어의 첫 음절을 잘못 제거함
# (예: 옛 패턴 `^(음|어|그|저)\s*` 는 "그게 뭐냐면" 의 "그" 만 잘라 "게 뭐냐면" 생성).
_FILLER_PREFIX_RE = re.compile(r"^(음|어|그|저)(?:\s+|$)")

# ack stall 하드코딩 폴백 텍스트.
# 사용자 피드백 (server_123844.log): "이해했어요" 부분이 중복·과하다는 인상.
# rag_probe 의 stall ("잠시만요, 확인해 드리겠습니다") 직후에 별도 ack 가 한 번 더 발사
# 되므로 짧고 단순한 "잠시만요." 로 충분 — 시스템이 처리 중임을 자연스럽게 전달.
_ACK_FALLBACK_TEXT = "잠시만요."


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────


def _normalize_text(raw: str, tenant_keyterms: list[str]) -> str:
    """STT raw 텍스트 정규화.

    단계:
      1. 선두 필러(음/어/그/저) 제거 (반복 제거).
      2. tenant 키텀 띄어쓰기 정정: 키텀 앞뒤에 공백이 없으면 삽입.
         긴 키텀을 먼저 처리해 부분 키텀이 더 긴 키텀을 깨뜨리지 않도록 정렬.
      3. 다중 공백 → 단일 공백, 양단 strip.
      4. 결과가 비거나 공백뿐이면 raw 를 strip 해서 반환.
    """
    text = raw.strip()

    # 선두 필러 반복 제거 (예: "음 어 그게 뭐냐면" → "그게 뭐냐면" → "뭐냐면"은 아님,
    # 패턴이 "음|어|그|저" 단독이므로 "그게" 는 걸리지 않음)
    prev = None
    while prev != text:
        prev = text
        text = _FILLER_PREFIX_RE.sub("", text)
    text = text.strip()

    # tenant 키텀 띄어쓰기 정정 — 길이 내림차순으로 처리.
    # 한국어는 조사 ("이/가/을/를/은/는" 등) 가 명사 뒤에 붙는 교착어 — 키워드 *뒤*에
    # 공백을 강제 삽입하면 "영업시간이" → "영업시간 이" 처럼 부자연스러워짐. 따라서
    # 키워드 *앞* 에만 공백 삽입. "구청을찾아가려고요" → "구청을 찾아가려고요" 형태.
    # (substring 매칭은 기본 동작이므로 공백이 없어도 router prompt 의 부분 문자열
    # 매칭 규칙으로 detect 됨 — 본 정규화는 가독성·LLM 토크나이저 안정성 보조용.)
    for kw in sorted(tenant_keyterms, key=len, reverse=True):
        if not kw:
            continue
        idx = 0
        result_parts: list[str] = []
        while idx < len(text):
            pos = text.find(kw, idx)
            if pos == -1:
                result_parts.append(text[idx:])
                break
            end = pos + len(kw)
            # kw 앞 문자가 공백이 아니면 → 앞에만 공백 삽입 (조사 분리 방지)
            need_pre = pos > 0 and text[pos - 1] != " "
            if need_pre:
                result_parts.append(text[idx:pos])
                result_parts.append(" ")
                result_parts.append(kw)
            else:
                result_parts.append(text[idx:end])
            idx = end

        text = "".join(result_parts)

    # 다중 공백 정리
    text = re.sub(r" {2,}", " ", text).strip()

    return text if text else raw.strip()


def _is_ambiguous(
    refined: str,
    rag_probe: Optional[dict],
    auth_pending: bool,
) -> tuple[bool, str]:
    """ambiguity gate.

    Returns:
        (is_ambiguous, reason)

    모호 판정 조건 (모두 충족 시 True):
      A. len(refined) ≤ _AMBIGUITY_MAX_LEN  OR  refined ∈ _FILLER_ONLY_PATTERNS
      B. rag_probe 없거나  top_distance > _AMBIGUITY_DISTANCE_THRESHOLD
      C. matched_keywords 없음
      D. is_auth 플래그 없음
      E. auth_pending=False
    """
    # 조건 E — auth_pending 이면 명확 (auth 분기 대기 중)
    if auth_pending:
        return False, "auth_pending"

    # 조건 A — 너무 짧거나 필러 단독
    short_or_filler = len(refined) <= _AMBIGUITY_MAX_LEN or refined in _FILLER_ONLY_PATTERNS

    if not short_or_filler:
        # 길이/필러 조건 자체를 통과 못 하면 명확
        return False, "sufficient_length"

    # 조건 B, C, D — rag_probe 신호 확인
    if rag_probe is not None:
        top_distance = rag_probe.get("top_distance", 2.0)
        matched_keywords = rag_probe.get("matched_keywords") or []
        is_auth = rag_probe.get("is_auth", False)

        # 조건 D — auth 관련 청크 매칭
        if is_auth:
            return False, "rag_auth_signal"

        # 조건 C — 키워드 매칭 있으면 명확
        if matched_keywords:
            return False, "matched_keywords_present"

        # 조건 B — RAG 신호 강하면 명확
        if top_distance <= _AMBIGUITY_DISTANCE_THRESHOLD:
            return False, "rag_strong"

    # 모든 조건 충족 — 모호
    reason = "filler_only" if refined in _FILLER_ONLY_PATTERNS else "short_no_signal"
    return True, reason


def _force_escalation_if_clarify_exhausted(state: CallState) -> Optional[dict]:
    """clarify 턴이 MAX_CLARIFY_TURNS 이상이면 escalation 강제 반환."""
    clarify_count = state.get("session_view", {}).get("clarify_count", 0)
    if clarify_count >= MAX_CLARIFY_TURNS:
        logger.info(
            "query_refine clarify_exhausted call_id=%s clarify_count=%d/%d → escalation",
            state.get("call_id", "?"), clarify_count, MAX_CLARIFY_TURNS,
        )
        return {
            "primary_intent": "intent_escalation",
            "is_ambiguous": False,
            "ambiguity_reason": "clarify_exhausted",
        }
    return None


def _spawn_ack_stall(
    call_id: str,
    tenant_id: str,
    stall_messages: dict,
) -> None:
    """ack stall fire-and-forget. push_ack 는 turn-once 가드 없음 — stall 과 동시 방출 가능.

    stall_messages["understood"] 가 있으면 그 텍스트를 사용, 없으면 하드코딩 폴백.
    """
    text = stall_messages.get("understood") or _ACK_FALLBACK_TEXT
    asyncio.create_task(
        tts_channel.push_ack(
            call_id=call_id,
            text=text,
            audio_field="understood",
        ),
        name=f"ack:{call_id}",
    )


# ── 노드 진입점 ───────────────────────────────────────────────────────────────


async def query_refine_node(state: CallState) -> dict:
    """STT 정규화 + ambiguity gate 노드 (LLM 없음, ~50ms 목표).

    출력 필드:
        refined_text      정제된 텍스트
        normalized_text   refined_text 와 동일 (하위 노드 자동 반영)
        is_ambiguous      모호 여부
        ambiguity_reason  gate 통과/차단 사유 (디버깅용)

    escalation 강제 시:
        primary_intent    "intent_escalation"
        is_ambiguous      False
        ambiguity_reason  "clarify_exhausted"
    """
    t0 = time.monotonic()
    call_id = state["call_id"]

    # 1. clarify 소진 → escalation 강제
    forced = _force_escalation_if_clarify_exhausted(state)
    if forced is not None:
        return forced

    # 2. raw 텍스트 획득 (raw_transcript 우선, 없으면 normalized_text)
    raw = state.get("raw_transcript") or state.get("normalized_text") or ""

    # 3. tenant 키텀 로드
    tenant_keyterms: list[str] = state.get("session_view", {}).get("tenant_keyterms") or []

    # 4. 정규화
    refined = _normalize_text(raw, tenant_keyterms)

    # 5. rag_probe + auth_pending 읽기
    probe = state.get("rag_probe")
    auth_pending: bool = state.get("auth_pending", False)

    # 6. ambiguity gate
    is_amb, reason = _is_ambiguous(refined, probe, auth_pending)

    # 7. 명확한 경우 ack stall fire-and-forget
    if not is_amb:
        stall_messages: dict = state.get("stall_messages") or {}
        _spawn_ack_stall(call_id, state["tenant_id"], stall_messages)

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "query_refine call_id=%s refined=%r is_ambiguous=%s reason=%s elapsed=%.1fms",
        call_id, refined, is_amb, reason, elapsed_ms,
    )

    return {
        "refined_text": refined,
        "normalized_text": refined,
        "is_ambiguous": is_amb,
        "ambiguity_reason": reason,
    }

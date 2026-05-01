import json

from app.agents.conversational.state import CallState
from app.services.llm.gpt4o_mini import GPT4OMiniService

_llm = GPT4OMiniService()
_HISTORY_TURN_LIMIT = 6  # 직전 3턴 (user+assistant 합쳐 6개 항목)

_SYSTEM_PROMPT = """당신은 전화 상담 AI의 쿼리 재작성기입니다.

[전화 상담 컨텍스트 — 중요]
사용자는 특정 매장/병원/회사에 전화 중입니다. 발화의 암묵적 주체는 "이 매장/이 회사" 입니다.
- "영업시간 알려주세요" → 이 매장의 영업시간 → is_clear=true (그대로 사용)
- "위치는요?" → 이 매장 위치 → is_clear=true
- "전화번호 알려주세요" → 이 매장 전화 → is_clear=true

단, 매장 업무와 명백히 무관한 사담/질문 (예: "오늘 날씨 어때요?", "축구 결과 알려주세요", "대통령이 누구예요?") 에는 "이 매장" 컨텍스트를 적용하지 말고 원본 그대로 유지하세요.

[판단 절차]
1. 발화가 위 컨텍스트 안에서 self-contained 인가?
2. self-contained 아니지만 이전 대화로 보강 가능 → 재작성 후 is_clear=true
3. "사용자가 모른다/식별 못 한다" 고 답한 경우 → is_clear=true 로 처리
   rewritten_query 에 "사용자가 ... 을 식별하지 못하는 상태" 라고 명시
   예: AI "어떤 상품?" → 사용자 "잘 몰라요"
       → rewritten_query: "사용자가 어떤 상품인지 식별하지 못하는 상태에서 상품 정보 요청"
4. 언어적으로 완벽히 모호하여 의도조차 파악할 수 없는 경우만 → is_clear=false

[핵심 규칙 1 — 비즈니스 파라미터(Slot) 검열 절대 금지]
사용자의 발화가 언어적으로 말이 된다면, 구체적인 조건(날짜, 시간, 이유, 정확한 상품명 등)이 없어도 무조건 is_clear=true 로 통과시키세요. 정보 수집은 다음 단계의 역할입니다.
- "예약할게요" → (날짜 없어도) is_clear=true, rewritten_query="예약 요청"
- "상담원 연결해주세요" → (이유 없어도) is_clear=true, rewritten_query="상담원 연결 요청"

[핵심 규칙 2 — 지시대명사 우선 예외 룰 (매우 중요)]
"거기", "여기", "그쪽" 등 장소를 지칭하는 단어는 대화 기록을 찾을 필요 없이 무조건 "이 매장"으로 치환하여 is_clear=true 로 통과시키세요.
- 예: "거기 어떻게 가요" → "이 매장 가는 길 문의" (is_clear=true)
- 예: "거기 뭐 팔아요" → "이 매장 판매 상품 문의" (is_clear=true)
- 예: "거기요" → "이 매장 직원 호출" (is_clear=true)

[일반 규칙]
- 그 외의 지시대명사("그거", "저거", "이거") → 이전 대화에서 referent 찾아 치환
- 대화 기록으로도 유추 불가능한 지시대명사나 너무 짧은 발화 ("어...", "잠깐만") → is_clear=false

[출력 — 순수 JSON 객체만 출력. 다른 텍스트 절대 금지]
{"is_clear": true|false, "rewritten_query": "...", "missing_info": "..."}

- 마크다운 블록(```json ... ```)을 절대 사용하지 마세요. 첫 글자는 반드시 '{' 로 시작해야 합니다.
- is_clear=true: rewritten_query 채움, missing_info 는 빈 문자열
- is_clear=false: rewritten_query 는 빈 문자열, missing_info 에 무엇이 부족한지 (예: "무엇을 지칭하시는지", "어떤 말씀이신지")"""


def _format_history(history: list) -> str:
    if not history:
        return "(이전 대화 없음)"
    lines = []
    for entry in history[-_HISTORY_TURN_LIMIT:]:
        role = "사용자" if entry.get("role") == "user" else "AI"
        lines.append(f"{role}: {entry.get('text', '')}")
    return "\n".join(lines)


async def query_refine_node(state: CallState) -> dict:
    user_text = state["user_text"]
    history = state.get("session_view", {}).get("conversation_history", [])

    user_message = f"[이전 대화]\n{_format_history(history)}\n\n[현재 발화]\n{user_text}"

    raw = await _llm.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.0,
        max_tokens=200,
    )

    try:
        parsed = json.loads(raw.strip())
        is_clear = bool(parsed.get("is_clear", True))
        rewritten = str(parsed.get("rewritten_query", "")).strip()
        missing = str(parsed.get("missing_info", "")).strip()
    except (json.JSONDecodeError, ValueError, AttributeError):
        # 파싱 실패: 원본 그대로 통과
        is_clear = True
        rewritten = user_text
        missing = ""

    # 안전장치: is_clear=True 인데 rewritten 비어있으면 원본 사용
    if is_clear and not rewritten:
        rewritten = user_text

    print(f"[query_refine] is_clear={is_clear} rewritten='{rewritten}' missing='{missing}'")
    return {
        "rewritten_query": rewritten,
        "is_clear": is_clear,
        "missing_info": missing,
    }

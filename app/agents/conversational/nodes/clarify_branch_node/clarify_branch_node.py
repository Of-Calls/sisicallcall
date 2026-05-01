from app.agents.conversational.state import CallState
from app.services.llm.gpt4o_mini import GPT4OMiniService

_llm = GPT4OMiniService()

_FALLBACK_TEXT = "죄송하지만 다시 한 번 말씀해주시겠어요?"

_SYSTEM_PROMPT = """당신은 전화 상담 AI 입니다. 사용자 발화가 모호해서 정확한 응대를 위해 한 번 더 물어봐야 합니다.

[지침]
- 사용자에게 친절하고 자연스러운 한국어로 한 문장의 역질문을 만드세요
- "죄송하지만", "혹시" 같은 부드러운 어조 사용
- 너무 격식적이거나 행정적인 표현은 피하세요 ("지칭하시는지" 같은 어색한 표현 X)
- 출력은 역질문 한 문장만. 다른 설명, 따옴표, 머릿말 금지."""


async def clarify_branch_node(state: CallState) -> dict:
    user_text = state.get("user_text", "").strip()
    missing = state.get("missing_info", "").strip()

    if not missing and not user_text:
        print("[clarify_branch] 입력 없음 → fallback")
        return {"response_text": _FALLBACK_TEXT}

    user_message = (
        f"[사용자 발화]\n{user_text}\n\n"
        f"[부족한 정보]\n{missing or '발화가 모호함'}"
    )

    try:
        text = await _llm.generate(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.2,
            max_tokens=80,
        )
        text = text.strip().strip('"').strip("'")
        if not text:
            text = _FALLBACK_TEXT
    except Exception as exc:
        print(f"[clarify_branch] LLM 실패 → fallback: {exc}")
        text = _FALLBACK_TEXT

    print(f"[clarify_branch] missing='{missing}' generated='{text}'")
    return {"response_text": text}

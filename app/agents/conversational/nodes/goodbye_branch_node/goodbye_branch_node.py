"""goodbye 분기 — 사용자 작별/종료 의사 감지 시 짧은 작별 응답 + 자동 hangup 신호.

query_refine 의 is_goodbye=True 가 graph._route_by_clarity 에서 이 분기로 라우팅.
intent_router 거치지 않음 (의도 분류 책임 분리).

call.py 가 should_hangup=True 받으면 TTS 송출 후 Twilio REST API 로 통화 종료.
"""
from app.agents.conversational.state import CallState


_GOODBYE_MESSAGE = "네, 감사합니다. 좋은 하루 되세요."


async def goodbye_branch_node(state: CallState) -> dict:
    print(f"[goodbye_branch] 진입 user_text='{state['user_text']}'")
    return {
        "response_text": _GOODBYE_MESSAGE,
        "should_hangup": True,
    }

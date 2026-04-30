"""Auth 브랜치 — 인증 필요 안내 + LLM 긍정 판단 + Solapi SMS 발송.

Turn 1: RAG is_auth 신호 → 확인 질문 발화, auth_pending=True 설정.
Turn 2: 사용자 응답 → LLM 긍정/부정 판단 → 긍정 시 SMS 발송.

SMS 수신번호: settings.solapi_sender_number (테스트 고정).
auth_pending 은 call.py 가 session_view 경유로 턴 간 유지한다.
"""
import asyncio

from app.agents.conversational.state import CallState
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.sms.solapi import SolapiSMSService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_llm = GPT4OMiniService()
_sms = SolapiSMSService()

AUTH_CONFIRM_TIMEOUT_SEC = 3.0

_AUTH_QUESTION = "해당 작업은 인증이 필요한 작업입니다. 사용자 인증을 진행하겠습니까?"
_AUTH_SMS_SENT = "인증 문자를 발송했습니다. 문자를 확인하신 후 인증을 완료해 주세요."
_AUTH_CANCELLED = "알겠습니다. 다른 도움이 필요하시면 말씀해 주세요."
_AUTH_NO_PHONE = "인증 문자 발송에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."

_CONFIRM_SYSTEM_PROMPT = """사용자의 발화가 인증 진행에 대한 긍정 응답인지 판단하세요.
긍정 예시: "네", "응", "예", "그래", "해줘", "부탁해", "할게", "진행해줘", "좋아"
부정 예시: "아니요", "괜찮아", "됐어", "안 해", "필요 없어"
출력: "positive" 또는 "negative" 한 단어만."""


async def _is_positive_response(text: str) -> bool:
    try:
        result = await asyncio.wait_for(
            _llm.generate(
                system_prompt=_CONFIRM_SYSTEM_PROMPT,
                user_message=text,
                temperature=0.0,
                max_tokens=5,
            ),
            timeout=AUTH_CONFIRM_TIMEOUT_SEC,
        )
        return (result or "").strip().lower().startswith("positive")
    except Exception as exc:
        logger.warning("auth confirm LLM 실패: %s — negative fallback", exc)
        return False


async def auth_branch_node(state: CallState) -> dict:
    call_id = state["call_id"]
    auth_pending = state.get("auth_pending", False)

    # Turn 1: 최초 인증 안내
    if not auth_pending:
        logger.info("auth_branch Turn1 call_id=%s — 확인 질문 발화", call_id)
        return {
            "response_text": _AUTH_QUESTION,
            "response_path": "auth",
            "is_timeout": False,
            "is_fallback": False,
            "auth_pending": True,
        }

    # Turn 2: 사용자 응답 처리
    normalized_text = state.get("normalized_text", "")
    logger.info("auth_branch Turn2 call_id=%s text='%s'", call_id, normalized_text)

    positive = await _is_positive_response(normalized_text)

    if not positive:
        logger.info("auth_branch 부정 응답 call_id=%s", call_id)
        return {
            "response_text": _AUTH_CANCELLED,
            "response_path": "auth",
            "is_timeout": False,
            "is_fallback": False,
            "auth_pending": False,
        }

    # 긍정 → SMS 발송 (테스트: SOLAPI_SENDER_NUMBER 고정)
    sms_to = settings.solapi_sender_number
    sms_body = "[시시콜콜] 본인 인증을 진행해 주세요. 아래 링크를 확인하세요."
    success = await _sms.send_sms(to=sms_to, body=sms_body)

    if not success:
        logger.error("auth SMS 발송 실패 call_id=%s to=%s", call_id, sms_to)
        return {
            "response_text": _AUTH_NO_PHONE,
            "response_path": "auth",
            "is_timeout": False,
            "is_fallback": True,
            "auth_pending": False,
        }

    logger.info("auth SMS 발송 완료 call_id=%s to=%s", call_id, sms_to)
    return {
        "response_text": _AUTH_SMS_SENT,
        "response_path": "auth",
        "is_timeout": False,
        "is_fallback": False,
        "auth_pending": False,
        "auth_sms_sent": True,
    }

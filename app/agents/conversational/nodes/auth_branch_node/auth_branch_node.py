import os

from app.agents.conversational.nodes.task_branch_node.task_branch_node import (
    ask_for_missing,
    humanize_tool_result,
)
from app.agents.conversational.prompts.fallback_phrases import get_inquiry_phrase
from app.agents.conversational.state import CallState
from app.agents.conversational.tool_catalog import get_available_actions
from app.services.auth.session import AuthSessionService
from app.services.ocr.auth_link_service import OCRAuthLinkService
from app.services.mcp.client import mcp_client
from app.services.session.redis_session import RedisSessionService
from app.services.sms import get_sms_service
from app.utils.auth_sms import build_face_auth_sms

_auth_session_svc = AuthSessionService()
_call_session_svc = RedisSessionService()
_sms_svc = get_sms_service()
_ocr_link_svc = OCRAuthLinkService(session_service=_auth_session_svc, sms_service=_sms_svc)

_POLITE_SMS_FAILED = "인증 링크 발송에 문제가 생겼어요. 잠시 후 다시 시도해주세요."
_POLITE_SMS_SENT = (
    "본인 인증 링크를 두 건(얼굴/신분증)으로 나눠 보내드렸어요. "
    "각 링크에서 인증을 완료해주세요."
)
_POLITE_IN_PROGRESS = "본인 인증을 진행 중이에요. 휴대폰에서 인증을 완료해주세요."
_POLITE_OCR_PENDING = (
    "문자에 있는 신분증 OCR 전용 링크(/ocr-auth)에서 촬영·인식을 먼저 완료해주세요."
)
_POLITE_FACE_PENDING = (
    "신분증 인식이 완료됐어요. 얼굴 인증 페이지(/auth 링크)로 돌아가 정면 촬영을 마쳐주세요."
)
_POLITE_VERIFIED = "인증이 완료됐어요. 어떤 도움이 필요하신가요?"
_POLITE_BLOCKED = "인증이 여러 번 실패해 차단됐어요. 상담원으로 연결해드릴게요."


def _polite_no_phone(industry: str) -> str:
    return f"본인 인증 진행을 위한 정보가 부족해요. {get_inquiry_phrase(industry)}."


def _polite_tool_failed(industry: str) -> str:
    return f"처리 중 문제가 생겼어요. 잠시 후 다시 시도해주시거나 {get_inquiry_phrase(industry)}."


async def _create_new_auth(call_id: str, tenant_id: str, customer_phone: str) -> dict:
    """새 auth 세션 생성 → SMS 발송 → 통화 세션에 auth_id 저장."""
    auth_id = await _auth_session_svc.create_session(
        tenant_id=tenant_id,
        customer_ref=customer_phone,
        customer_phone=customer_phone,
        call_id=call_id,
    )
    print(f"[auth_branch] 세션 생성 auth_id={auth_id}")

    sent_face = await _sms_svc.send_sms(to=customer_phone, body=build_face_auth_sms(auth_id))
    sent_ocr = await _ocr_link_svc.send_link(auth_id, customer_phone)
    print(f"[auth_branch] SMS 발송 face={sent_face} ocr={sent_ocr} to={customer_phone}")

    if not (sent_face and sent_ocr):
        return {"response_text": _POLITE_SMS_FAILED}

    # SMS 발송 성공 시에만 auth_id 저장 — 실패 시 다음 진입 때 재시도하도록.
    await _call_session_svc.set_auth_id(call_id, auth_id)
    print(f"[auth_branch] 통화 세션에 auth_id 저장")
    return {"response_text": _POLITE_SMS_SENT}


async def _resume_pending_task(
    call_id: str, tenant_id: str, tenant_industry: str, pending: dict
) -> dict:
    """auth verified 직후 task_branch 가 저장한 pending_task 자동 재실행.

    spec 재조회 → required 검증:
      - missing 있으면 ask_for_missing (다음 turn 에 task 가 LLM history 기반으로 채움).
      - 완전하면 mcp_client.call_tool + humanize.
    pending_task 는 호출 측에서 이미 clear. 실패 시 polite_tool_failed (industry 기반).
    """
    action_type = pending.get("action_type", "")
    arguments = pending.get("arguments") or {}
    user_text = pending.get("user_text", "")
    print(f"[auth_branch] pending_task 자동 재실행 action={action_type} args={arguments}")

    # spec 재조회 — 매장 도구 disconnect 같은 edge case 안전망.
    available = await get_available_actions(tenant_id)
    spec = available.get(action_type)
    if spec is None:
        print(f"[auth_branch] pending action={action_type} 카탈로그 없음 → polite_verified")
        return {"response_text": _POLITE_VERIFIED}

    required = (spec.get("parameters") or {}).get("required") or []
    missing = [k for k in required if not arguments.get(k)]
    if missing:
        print(f"[auth_branch] resume args 부족 missing={missing} → ask_for_missing")
        text = await ask_for_missing(user_text, action_type, spec, missing)
        return {"response_text": text}

    try:
        mcp_result = await mcp_client.call_tool(
            pending["tool"],
            action_type,
            arguments,
            call_id=call_id,
            tenant_id=tenant_id,
        )
    except Exception as exc:
        print(f"[auth_branch] mcp 호출 실패: {exc}")
        return {"response_text": _polite_tool_failed(tenant_industry)}

    print(f"[auth_branch] mcp_result status={mcp_result.get('status')}")
    if mcp_result.get("status") != "success":
        return {"response_text": _polite_tool_failed(tenant_industry)}

    text = await humanize_tool_result(user_text, action_type, mcp_result, tenant_industry)
    return {"response_text": text}


async def auth_branch_node(state: CallState) -> dict:
    tenant_id = state["tenant_id"]
    tenant_industry = state.get("tenant_industry", "")
    call_id = state["call_id"]

    # 시연용: SMS_TEST_RECIPIENT 고정. 실서비스에서는 Twilio caller ID 매핑으로 대체.
    customer_phone = os.getenv("SMS_TEST_RECIPIENT", "")
    if not customer_phone:
        print("[auth_branch] SMS_TEST_RECIPIENT 미설정 → polite_no_phone")
        return {"response_text": _polite_no_phone(tenant_industry)}

    # ① 진행 중인 auth 가 있으면 status 분기 (재발송 방지)
    existing_auth_id = await _call_session_svc.get_auth_id(call_id)
    if existing_auth_id:
        auth_session = await _auth_session_svc.get_session(existing_auth_id)
        if auth_session is None:
            # auth 세션 TTL (10분) 만료 — 통화는 살아있지만 인증은 끊김. 재발급.
            print(f"[auth_branch] 기존 auth_id={existing_auth_id} TTL 만료 → 재발급")
            return await _create_new_auth(call_id, tenant_id, customer_phone)

        status = auth_session.get("status", "")
        liveness_ok = auth_session.get("liveness_passed") == "true"
        ocr_ok = auth_session.get("ocr_passed") == "true"
        face_ok = auth_session.get("face_verified") == "true"
        print(
            f"[auth_branch] 기존 auth_id={existing_auth_id} status={status} "
            f"liveness={liveness_ok} ocr={ocr_ok} face={face_ok}"
        )

        if status == "verified":
            pending = await _call_session_svc.get_pending_task(call_id)
            if pending:
                await _call_session_svc.clear_pending_task(call_id)
                return await _resume_pending_task(call_id, tenant_id, tenant_industry, pending)
            return {"response_text": _POLITE_VERIFIED}
        if status == "blocked":
            return {"response_text": _POLITE_BLOCKED}
        if liveness_ok and not ocr_ok:
            return {"response_text": _POLITE_OCR_PENDING}
        if liveness_ok and ocr_ok and not face_ok:
            return {"response_text": _POLITE_FACE_PENDING}
        return {"response_text": _POLITE_IN_PROGRESS}

    # ② 신규 진입 — D-A 와 동일
    return await _create_new_auth(call_id, tenant_id, customer_phone)

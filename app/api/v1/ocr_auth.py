"""신분증 OCR 전용 페이지·API (/ocr-auth/...). 얼굴 인증(/auth/...)과 경로 분리."""

import json
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.schemas.auth import AuthInitiateRequest, AuthInitiateResponse, AuthStatusResponse
from app.services.auth.session import AuthSessionService
from app.services.ocr.id_card_ocr_service import get_id_card_ocr_service
from app.services.sms import get_sms_service
from app.utils.auth_sms import build_ocr_auth_sms, ocr_auth_url
from app.utils.config import settings
from app.utils.logger import get_logger

_OCR_PAGE_HTML = (
    Path(__file__).parent.parent.parent / "static" / "auth_ocr.html"
).read_text(encoding="utf-8")

logger = get_logger(__name__)

router = APIRouter()

_session_svc = AuthSessionService()
_ocr = get_id_card_ocr_service()
_sms_svc = get_sms_service()
_ROI_PROFILE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "ocr_roi_profile.json"


@router.post("/verify", response_model=AuthInitiateResponse)
async def initiate_ocr_auth(body: AuthInitiateRequest):
    """인증 세션 생성 + 신분증 OCR 링크 SMS 발송."""
    auth_id = await _session_svc.create_session(
        tenant_id=body.tenant_id,
        customer_ref=body.customer_ref,
        customer_phone=body.customer_phone,
        call_id=body.call_id,
    )
    ocr_link = ocr_auth_url(auth_id)
    if settings.auth_skip_sms:
        return AuthInitiateResponse(
            auth_id=auth_id,
            status="pending",
            message=f"SMS 스킵 모드 — OCR 인증 링크: {ocr_link}",
        )

    sent = await _sms_svc.send_sms(to=body.customer_phone, body=build_ocr_auth_sms(auth_id))
    if not sent:
        logger.error("OCR 인증 SMS 발송 실패 auth_id=%s phone=%s", auth_id, body.customer_phone)
    return AuthInitiateResponse(
        auth_id=auth_id,
        status="pending",
        message="OCR 인증 SMS 발송 완료" if sent else "OCR 인증 SMS 발송 실패 — 인증 세션은 유효",
    )


@router.get("/{auth_id}/status", response_model=AuthStatusResponse)
async def get_ocr_auth_status(auth_id: str):
    """인증 세션 상태 — 얼굴 플로우와 동일 Redis 해시."""
    session = await _session_svc.get_session(auth_id)
    if not session:
        raise HTTPException(status_code=404, detail="인증 세션이 없거나 만료됨")
    return AuthStatusResponse(
        auth_id=auth_id,
        status=session.get("status", "unknown"),
        liveness_passed=session.get("liveness_passed") == "true",
        ocr_passed=session.get("ocr_passed") == "true",
        face_verified=session.get("face_verified") == "true",
        created_at=session.get("created_at"),
    )


@router.get("/config/roi-profile")
async def get_roi_profile():
    if not _ROI_PROFILE_PATH.exists():
        return {"exists": False}
    try:
        data = json.loads(_ROI_PROFILE_PATH.read_text(encoding="utf-8"))
        return {"exists": True, "profile": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ROI profile read failed: {exc}")


@router.post("/config/roi-profile")
async def save_roi_profile(payload: dict = Body(...)):
    required = ("roi_x1", "roi_x2", "roi_y1", "roi_y2")
    missing = [k for k in required if k not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing keys: {missing}")
    data = {k: float(payload[k]) for k in required}
    _ROI_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ROI_PROFILE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("OCR ROI profile saved path=%s", _ROI_PROFILE_PATH)
    return {"saved": True, "path": str(_ROI_PROFILE_PATH)}


@router.post("/{auth_id}/capture")
async def capture_id_card(
    auth_id: str,
    file: UploadFile = File(...),
    roi_x1: float | None = Form(default=None),
    roi_x2: float | None = Form(default=None),
    roi_y1: float | None = Form(default=None),
    roi_y2: float | None = Form(default=None),
):
    """신분증 이미지 OCR — 성공 시 ocr_passed 반영."""
    session = await _session_svc.get_session(auth_id)
    if not session:
        raise HTTPException(status_code=404, detail="인증 세션이 없거나 만료됨")

    image_bytes = await file.read()
    roi_override = None
    if None not in (roi_x1, roi_x2, roi_y1, roi_y2):
        roi_override = {
            "roi_x1": float(roi_x1),
            "roi_x2": float(roi_x2),
            "roi_y1": float(roi_y1),
            "roi_y2": float(roi_y2),
        }
    result = await _ocr.process_image(image_bytes, roi_override=roi_override)
    if result.get("status") == "success":
        await _session_svc.set_ocr_passed(auth_id)
    return result


@router.get("/{auth_id}", response_class=HTMLResponse)
async def ocr_auth_page(auth_id: str) -> HTMLResponse:
    """신분증 OCR 전용 페이지."""
    return HTMLResponse(content=_OCR_PAGE_HTML)

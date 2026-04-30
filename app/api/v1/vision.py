from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.sms import get_sms_service
from app.services.vision.session_store import (
    VISION_STATUS_WAITING_UPLOAD,
    VisionUploadSessionStore,
)
from app.services.vision.token import create_upload_token, hash_upload_token
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_session_store = VisionUploadSessionStore()
_sms_svc = get_sms_service()


class VisionUploadSessionRequest(BaseModel):
    call_id: str
    tenant_id: str
    caller_number: str


class VisionUploadSessionResponse(BaseModel):
    upload_url: str
    expires_in_sec: int
    sms_sent: bool
    status: str


def _build_upload_url(token: str) -> str:
    base_url = settings.vision_upload_base_url.rstrip("/")
    path_prefix = settings.vision_upload_path_prefix.strip("/")
    if path_prefix:
        return f"{base_url}/{path_prefix}/{token}"
    return f"{base_url}/{token}"


@router.post("/upload-sessions", response_model=VisionUploadSessionResponse)
async def create_upload_session(body: VisionUploadSessionRequest):
    token = create_upload_token()
    token_hash = hash_upload_token(token)
    expires_in_sec = settings.vision_upload_ttl_sec
    upload_url = _build_upload_url(token)

    try:
        await _session_store.create_upload_session(
            token_hash=token_hash,
            call_id=body.call_id,
            tenant_id=body.tenant_id,
            caller_number=body.caller_number,
            expires_in_sec=expires_in_sec,
        )
    except Exception as e:
        logger.exception(
            "vision upload session create failed call_id=%s tenant_id=%s: %s",
            body.call_id,
            body.tenant_id,
            e,
        )
        raise HTTPException(status_code=500, detail="vision upload session create failed")

    await _session_store.set_call_vision_status(
        call_id=body.call_id,
        status=VISION_STATUS_WAITING_UPLOAD,
    )

    sms_body = f"[sisicallcall] Product photo upload link: {upload_url}"
    try:
        sms_sent = await _sms_svc.send_sms(to=body.caller_number, body=sms_body)
    except Exception as e:
        logger.warning(
            "vision upload SMS send failed call_id=%s phone=%s: %s",
            body.call_id,
            body.caller_number,
            e,
        )
        sms_sent = False

    return VisionUploadSessionResponse(
        upload_url=upload_url,
        expires_in_sec=expires_in_sec,
        sms_sent=bool(sms_sent),
        status=VISION_STATUS_WAITING_UPLOAD,
    )

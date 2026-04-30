from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.sms import get_sms_service
from app.services.vision.session_store import (
    VISION_STATUS_PROCESSING,
    VISION_STATUS_WAITING_UPLOAD,
    VisionUploadSessionAlreadyUsedError,
    VisionUploadSessionNotFoundError,
    VisionUploadSessionStore,
)
from app.services.vision.storage import VisionImageStorageError, save_vision_image
from app.services.vision.token import create_upload_token, hash_upload_token
from app.services.vision.validation import (
    EmptyImageFileError,
    ImageFileTooLargeError,
    UnsupportedImageMimeTypeError,
    validate_image_upload,
)
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
_SLOWAPI_EMPTY_CONFIG = str(Path(__file__).resolve().parents[2] / "__init__.py")
limiter = Limiter(key_func=get_remote_address, config_filename=_SLOWAPI_EMPTY_CONFIG)

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


class VisionUploadSessionStatusResponse(BaseModel):
    status: str
    expires_at: str
    used: bool


class VisionImageUploadResponse(BaseModel):
    status: str
    message: str


def _parse_redis_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


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


@router.get("/upload-sessions/{token}", response_model=VisionUploadSessionStatusResponse)
async def get_upload_session(token: str):
    token_hash = hash_upload_token(token)

    try:
        session = await _session_store.get_upload_session(token_hash)
    except Exception as e:
        logger.exception(
            "vision upload session lookup failed token_hash=%s: %s",
            token_hash,
            e,
        )
        raise HTTPException(status_code=500, detail="vision upload session lookup failed")

    if not session:
        raise HTTPException(
            status_code=404,
            detail="vision upload session not found or expired",
        )

    used = _parse_redis_bool(session.get("used", "false"))
    if used:
        raise HTTPException(
            status_code=409,
            detail="vision upload session already used",
        )

    return VisionUploadSessionStatusResponse(
        status=session.get("status", VISION_STATUS_WAITING_UPLOAD),
        expires_at=session.get("expires_at", ""),
        used=used,
    )


@router.post("/upload", response_model=VisionImageUploadResponse)
@limiter.limit("5/minute")
async def upload_vision_image(
    request: Request,
    token: str | None = Query(None),
    file: UploadFile = File(...),
):
    if not token:
        raise HTTPException(
            status_code=404,
            detail="vision upload session not found or expired",
        )

    token_hash = hash_upload_token(token)

    try:
        validated = await validate_image_upload(
            file,
            settings.vision_upload_max_bytes,
        )
    except EmptyImageFileError:
        raise HTTPException(status_code=400, detail="empty image file")
    except UnsupportedImageMimeTypeError:
        raise HTTPException(status_code=400, detail="unsupported image MIME type")
    except ImageFileTooLargeError:
        raise HTTPException(status_code=413, detail="image file too large")
    except Exception as e:
        logger.exception(
            "vision image validation failed token_hash=%s: %s",
            token_hash,
            e,
        )
        raise HTTPException(status_code=500, detail="vision image validation failed")

    try:
        session = await _session_store.reserve_upload_session_for_processing(token_hash)
    except VisionUploadSessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="vision upload session not found or expired",
        )
    except VisionUploadSessionAlreadyUsedError:
        raise HTTPException(
            status_code=409,
            detail="vision upload session already used",
        )
    except Exception as e:
        logger.exception(
            "vision upload session reserve failed token_hash=%s: %s",
            token_hash,
            e,
        )
        raise HTTPException(
            status_code=500,
            detail="vision upload session update failed",
        )

    call_id = session.get("call_id")
    if not call_id:
        logger.error("vision upload session missing call_id token_hash=%s", token_hash)
        raise HTTPException(
            status_code=500,
            detail="vision upload session update failed",
        )

    try:
        image_path = await save_vision_image(
            content=validated.content,
            extension=validated.extension,
            call_id=call_id,
            upload_dir=settings.vision_upload_dir,
        )
    except VisionImageStorageError as e:
        # TODO: storage 실패 시 FAILED 상태 전환 또는 cleanup 필요.
        logger.exception(
            "vision image save failed token_hash=%s call_id=%s: %s",
            token_hash,
            call_id,
            e,
        )
        raise HTTPException(status_code=500, detail="vision image save failed")

    try:
        await _session_store.attach_upload_file_info(
            token_hash=token_hash,
            image_path=image_path,
            mime_type=validated.mime_type,
            size_bytes=validated.size_bytes,
        )
    except Exception as e:
        logger.exception(
            "vision upload file info attach failed token_hash=%s call_id=%s: %s",
            token_hash,
            call_id,
            e,
        )
        raise HTTPException(
            status_code=500,
            detail="vision upload session update failed",
        )

    await _session_store.set_call_vision_status(
        call_id=call_id,
        status=VISION_STATUS_PROCESSING,
    )

    return VisionImageUploadResponse(
        status=VISION_STATUS_PROCESSING,
        message="사진을 받았습니다. 제품을 분석하고 있습니다.",
    )

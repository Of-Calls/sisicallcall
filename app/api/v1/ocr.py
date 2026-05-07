from fastapi import APIRouter, File, UploadFile

from app.services.ocr.id_card_ocr_service import get_id_card_ocr_service

router = APIRouter()

_svc = get_id_card_ocr_service()


@router.post("/ocr")
async def ocr_card(file: UploadFile = File(...)):
    """로컬/테스트용 신분증 OCR — 인증 세션 없이 이미지만 검사."""
    raw = await file.read()
    return await _svc.process_image(raw)

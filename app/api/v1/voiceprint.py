"""in-memory voiceprint 관리 (ONNX 싱글톤). DB 없음."""

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse

from app.services.speaker_verify import enrollment as voice_enrollment
from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service
from app.services.speaker_verify.titanet_compare import cleanup_compare_for_call
from app.utils.logger import get_logger

router = APIRouter()
_logger = get_logger(__name__)


@router.delete("/{call_id}", response_model=None)
async def delete_voiceprint(call_id: str) -> Response | JSONResponse:
    """특정 통화(call_id / streamSid)의 finetuned voiceprint 및 enrollment 상태 삭제."""
    removed_f = get_finetuned_onnx_service().cleanup(call_id)
    cleanup_compare_for_call(call_id)
    voice_enrollment.cleanup(call_id)
    if not removed_f:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "해당 call_id voiceprint 없음"},
        )
    _logger.info(
        "voiceprint DELETE call_id=%s finetuned=%s",
        call_id,
        removed_f,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

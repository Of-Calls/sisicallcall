"""레거시 진입점 — NeMo 백엔드 없음. 모두 onnxruntime(medium ONNX)와 동일 싱글톤.

`get_titanet_service()` · enrollment 등은 `[medium]` 파이프라인과 같은 모델을 쓴다:
  기본 `app/models/speaker_verification/titanet_small_medium_5epoch_lr5e5.onnx`
  (`settings.titanet_pipeline_onnx_path` 로 덮어쓰기 가능)
"""
from __future__ import annotations

from app.services.speaker_verify.onnx_pipeline import (
    OnnxMelSpeakerVerifyService,
    get_onnx_pipeline_service,
)


def get_titanet_service() -> OnnxMelSpeakerVerifyService:
    """enrollment·레거시 노드 호환."""
    return get_onnx_pipeline_service()


class TitaNetSpeakerVerifyService:
    """스크립트 호환: `TitaNetSpeakerVerifyService()` → onnx 파이프라인 싱글톤."""

    def __new__(cls) -> OnnxMelSpeakerVerifyService:
        return get_onnx_pipeline_service()

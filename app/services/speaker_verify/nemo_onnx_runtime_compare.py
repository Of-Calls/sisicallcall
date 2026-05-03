"""통화 중(발화 단위) 파인튜닝 ONNX vs NeMo `forward_for_export` 임베딩 코사인 로그.

동일 mel(파인튜닝 ONNX 서비스의 `_pcm_to_mel_np`)을 ONNX Runtime과 NeMo에 넣어
`scripts/compare_nemo_vs_onnx_embeddings.py` 와 같은 의미의 cos 를 찍는다.

켜기: `SPEAKER_VERIFY_NEMO_ONNX_COMPARE_ON_CALL=true` + `TITANET_SPEAKER_NEMO_PATH`(.nemo).
NeMo는 첫 비교 시 restore(CPU 고정); 이후 발화는 동일 모델 재사용. 무거우므로 상시 true 비권장.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)

_nemo_lock = threading.Lock()
_nemo_model: Any = None
_nemo_device: Any = None
_warned_missing_nemo_path = False

_compare_chain_lock = threading.Lock()


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x) + 1e-12)
    return x / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2n(a), _l2n(b)))


def _get_nemo_model():
    global _nemo_model, _nemo_device
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    with _nemo_lock:
        if _nemo_model is None:
            raw = (settings.titanet_speaker_nemo_path or "").strip()
            if not raw:
                raise RuntimeError("TITANET_SPEAKER_NEMO_PATH 비어 있음")
            path = Path(raw)
            if not path.is_file():
                raise RuntimeError(f".nemo 없음: {path}")
            _nemo_device = torch.device("cpu")
            _logger.info("nemo_vs_onnx: NeMo restore_from 시작 path=%s device=cpu …", path)
            t0 = time.monotonic()
            _nemo_model = EncDecSpeakerLabelModel.restore_from(restore_path=str(path))
            _nemo_model.eval()
            _nemo_model.to(_nemo_device)
            elapsed = time.monotonic() - t0
            _logger.info(
                "nemo_vs_onnx: NeMo restore_from 완료 elapsed=%.2fs device=cpu",
                elapsed,
            )
        return _nemo_model, _nemo_device


def compare_finetuned_onnx_vs_nemo_sync(
    pcm16: bytes,
) -> tuple[float | None, str | None, int]:
    """동기: (cosine, err, T_frames). cos 는 L2 정규화 후 내적."""
    from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service

    ft = get_finetuned_onnx_service()
    if ft.load_error or ft._ort_sess is None:
        return None, "finetuned ONNX 미로드", 0
    # mel 은 지연 로드 — _pcm_to_mel_np 가 첫 호출 시 _ensure_mel_frontend 수행
    mel = ft._pcm_to_mel_np(pcm16)
    t_frames = int(mel.shape[2])
    onnx_emb = ft._embedding_from_mel_np(mel)

    model, device = _get_nemo_model()
    import torch

    mel_t = torch.from_numpy(mel).float().to(device)
    ln = torch.tensor([t_frames], device=device, dtype=torch.int64)
    with torch.no_grad():
        out = model.forward_for_export(mel_t, ln)
    if isinstance(out, (tuple, list)):
        emb_t = out[1] if len(out) > 1 else out[0]
    else:
        emb_t = out
    nemo_emb = emb_t.detach().float().cpu().numpy().reshape(-1)

    if onnx_emb.shape != nemo_emb.shape:
        return None, f"dim 불일치 onnx={onnx_emb.shape} nemo={nemo_emb.shape}", t_frames
    return _cos(onnx_emb, nemo_emb), None, t_frames


def compare_finetuned_onnx_vs_nemo_sync_safe(
    pcm16: bytes,
) -> tuple[float | None, str | None, int]:
    """한 번에 하나의 비교만 실행(통화 중 중첩 방지)."""
    with _compare_chain_lock:
        try:
            return compare_finetuned_onnx_vs_nemo_sync(pcm16)
        except Exception as e:
            return None, str(e), 0


async def log_nemo_vs_onnx_after_utterance(
    pcm16: bytes,
    *,
    call_id: str,
    utt_seq: int,
) -> None:
    """이벤트 루프에서 호출: NeMo+ONNX 비교는 스레드에서 실행 후 로그."""
    global _warned_missing_nemo_path

    if not settings.speaker_verify_nemo_onnx_compare_on_call:
        return
    raw = (settings.titanet_speaker_nemo_path or "").strip()
    if not raw:
        if not _warned_missing_nemo_path:
            _warned_missing_nemo_path = True
            _logger.warning(
                "nemo_vs_onnx: SPEAKER_VERIFY_NEMO_ONNX_COMPARE_ON_CALL 이 true 인데 "
                "TITANET_SPEAKER_NEMO_PATH 가 비어 있음 — 비교 생략"
            )
        return

    loop = asyncio.get_running_loop()
    try:
        cos, err, t_frames = await loop.run_in_executor(
            None, compare_finetuned_onnx_vs_nemo_sync_safe, pcm16
        )
    except Exception:
        _logger.exception(
            "nemo_vs_onnx streamSid=%s utt=%d executor 실패",
            call_id,
            utt_seq,
        )
        return

    if err:
        _logger.warning(
            "nemo_vs_onnx streamSid=%s utt=%d 실패: %s",
            call_id,
            utt_seq,
            err,
        )
        return
    if cos is None:
        return
    _logger.info(
        "nemo_vs_onnx streamSid=%s utt=%d T=%d cos(ONNX,NeMo_same_mel)=%.6f "
        "(~1 export 정합, 낮으면 체크포인트·mel·동적축 의심)",
        call_id,
        utt_seq,
        t_frames,
        cos,
    )


def spawn_nemo_vs_onnx_compare_task(
    pcm16: bytes,
    *,
    call_id: str,
    utt_seq: int,
) -> None:
    """통화 핫패스에서 논블로킹으로 비교 태스크만 붙인다."""
    if not settings.speaker_verify_nemo_onnx_compare_on_call:
        return
    if not (settings.titanet_speaker_nemo_path or "").strip():
        return

    async def _run() -> None:
        try:
            await log_nemo_vs_onnx_after_utterance(pcm16, call_id=call_id, utt_seq=utt_seq)
        except Exception:
            _logger.exception(
                "nemo_vs_onnx streamSid=%s utt=%d task 실패",
                call_id,
                utt_seq,
            )

    try:
        asyncio.create_task(_run())
    except RuntimeError:
        # 이벤트 루프 없음(유닛 테스트 등)
        _logger.debug("nemo_vs_onnx: no running loop, skip")

"""Voiceprint enrollment — 통화당 STT 성공 발화 N건으로 finetuned 임베딩 수집 후 평균·L2 저장.

PCM 은 발화 단위로 넘어오며, **임베딩 추출·저장은 finetuned ONNX**만 사용한다.
`cleanup()` 은 call.py 통화 종료 finally 에서 호출.
"""
from __future__ import annotations

import asyncio
import functools

import numpy as np

from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service_async
from app.services.speaker_verify.titanet_compare import (
    get_titanet_compare_speaker_verify_service,
)
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# per-call: 발화별 임베딩 리스트 (finalize 전까지)
_emb_finetuned: dict[str, list[np.ndarray]] = {}
_done: dict[str, bool] = {}


def _enroll_n() -> int:
    return max(1, int(settings.enroll_utt_count))


def _l2_normalize_1d(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return np.asarray(x, dtype=np.float32)
    return np.asarray(x / n, dtype=np.float32)


async def accumulate(call_id: str, audio_chunk: bytes, transcript: str) -> bool:
    """STT 성공 발화마다 finetuned 임베딩 1개 수집. N개 도달 시 평균→L2→저장.

    Returns:
        True: 등록 완료(이번 또는 이전). False: 미완료.
    """
    if not transcript:
        return _done.get(call_id, False)
    if _done.get(call_id, False):
        return True

    n_need = _enroll_n()
    finetuned = await get_finetuned_onnx_service_async()

    try:
        emb_f: np.ndarray | None = None
        if not finetuned.load_error:
            emb_f = await finetuned.extract_embedding_for_enroll(audio_chunk)

        if emb_f is not None:
            lf = _emb_finetuned.setdefault(call_id, [])
            lf.append(emb_f)
            logger.info(
                "[enrollment] finetuned 발화 수집 %d/%d call_id=%s",
                len(lf),
                n_need,
                call_id,
            )
        if emb_f is not None and settings.speaker_verify_compare_enabled:
            cmp = get_titanet_compare_speaker_verify_service()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                cmp.accumulate_enrollment_utterance_sync,
                call_id,
                audio_chunk,
            )
    except Exception as e:
        logger.error("enrollment 임베딩 추출 실패 call_id=%s: %s", call_id, e)
        return _done.get(call_id, False)

    ready_f = bool(finetuned.load_error) or len(_emb_finetuned.get(call_id, [])) >= n_need
    if not ready_f:
        return _done.get(call_id, False)

    try:
        if not finetuned.load_error:
            stack_f = np.stack(_emb_finetuned[call_id][:n_need], axis=0)
            vp_f = _l2_normalize_1d(np.mean(stack_f, axis=0))
            finetuned.store_voiceprint_vector(call_id, vp_f)
            logger.info("[enrollment] finetuned 완료 — %d발화 평균", n_need)
        if settings.speaker_verify_compare_enabled:
            cmp = get_titanet_compare_speaker_verify_service()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                functools.partial(cmp.finalize_enrollment_sync, call_id),
            )
    except Exception as e:
        logger.error("enrollment 최종 저장 실패 call_id=%s: %s", call_id, e)
        return _done.get(call_id, False)

    ok_finetuned = bool(finetuned.load_error) or finetuned.has_voiceprint(call_id)
    if ok_finetuned:
        _done[call_id] = True
        logger.info("voiceprint enrollment 완료 call_id=%s", call_id)
        logger.info(
            "finetuned voiceprint[:4]: %s",
            finetuned.voiceprint_slice_first_n(call_id, 4),
        )
        _emb_finetuned.pop(call_id, None)
    else:
        logger.error(
            "enrollment 후 voiceprint 없음 call_id=%s finetuned_ok=%s",
            call_id,
            ok_finetuned,
        )

    return _done.get(call_id, False)


def cleanup(call_id: str) -> None:
    """통화 종료 시 per-call enrollment·임베딩 버퍼 정리."""
    buf_f = _emb_finetuned.pop(call_id, None) is not None
    done_removed = _done.pop(call_id, None) is not None
    if buf_f or done_removed:
        logger.info("enrollment 상태 삭제 call_id=%s", call_id)


def reset_all_enrollment_state() -> None:
    """기동 시 등록 버퍼·완료 플래그 전역 비움."""
    nb_f = len(_emb_finetuned)
    nd = len(_done)
    _emb_finetuned.clear()
    _done.clear()
    if settings.speaker_verify_compare_enabled:
        try:
            get_titanet_compare_speaker_verify_service().reset_enrollment_buffers_only()
        except Exception:
            logger.exception("titanet_compare: enrollment 버퍼 초기화 실패")
    logger.info(
        "enrollment 전역 초기화 (finetuned_buffers=%d, done=%d)",
        nb_f,
        nd,
    )

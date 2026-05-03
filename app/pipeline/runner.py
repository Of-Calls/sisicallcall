"""Twilio μ-law 청크를 3개 VAD 상태에 동기 fan-out → 발화 단위로 3갈래 STT·로그.

각 파이프라인에 `asyncio.gather`로 청크를 동시에 넣어 VAD 경계를 일치시킨다.
"""
from __future__ import annotations

import audioop
import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

from app.services.speaker_verify import enrollment as voice_enrollment
from app.services.speaker_verify.nemo_onnx_runtime_compare import (
    spawn_nemo_vs_onnx_compare_task,
)
from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service_async
from app.services.speaker_verify.onnx_pipeline import get_onnx_pipeline_service
from app.services.stt.deepgram import DeepgramSTTService
from app.services.vad.silero_vad import SileroVADService
from app.utils.config import settings
from app.utils.logger import get_logger

_logger = get_logger(__name__)

_VAD_FRAME_BYTES = 1024
_SILENCE_THRESHOLD = 30


def mulaw_to_pcm16_16k(mulaw: bytes) -> bytes:
    if not mulaw:
        return b""
    pcm_8k = audioop.ulaw2lin(mulaw, 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return pcm_16k


@dataclass
class VADPipelineState:
    audio_buffer: bytearray = field(default_factory=bytearray)
    pcm_buffer: bytearray = field(default_factory=bytearray)
    ratecv_state: Any = None
    silence_count: int = 0
    had_speech: bool = False

    def reset(self) -> None:
        self.audio_buffer.clear()
        self.pcm_buffer.clear()
        self.ratecv_state = None
        self.silence_count = 0
        self.had_speech = False

    async def feed_chunk(self, mulaw: bytes, vad: SileroVADService) -> bytes | None:
        self.audio_buffer.extend(mulaw)
        pcm_8k = audioop.ulaw2lin(mulaw, 2)
        pcm_16k, self.ratecv_state = audioop.ratecv(
            pcm_8k, 2, 1, 8000, 16000, self.ratecv_state
        )
        self.pcm_buffer.extend(pcm_16k)
        while len(self.pcm_buffer) >= _VAD_FRAME_BYTES:
            frame = bytes(self.pcm_buffer[:_VAD_FRAME_BYTES])
            del self.pcm_buffer[:_VAD_FRAME_BYTES]
            is_speech = await vad.detect(frame)
            if is_speech:
                self.silence_count = 0
                self.had_speech = True
            else:
                self.silence_count += 1
                if self.silence_count >= _SILENCE_THRESHOLD and self.had_speech:
                    mulaw_utt = bytes(self.audio_buffer)
                    self.audio_buffer.clear()
                    self.silence_count = 0
                    self.had_speech = False
                    return mulaw_utt
        return None


def _stt_text(result: str | BaseException) -> str:
    if isinstance(result, BaseException):
        _logger.error("STT 실패: %s", result)
        return f"STT 실패: {result!s}"
    return result or ""


def _onnx_branch_ready(svc: Any, call_id: str) -> bool:
    """로드 실패 갈래는 제외하고, 활성 ONNX는 voiceprint가 있어야 검증·STT 단계로 진입."""
    if svc.load_error:
        return True
    return svc.has_voiceprint(call_id)


@dataclass
class SttLatencyAccumulator:
    """발화 경계(파이프라인 진입)부터 해당 갈래 STT `transcribe` 완료까지 초 단위 샘플."""

    finetuned: list[float] = field(default_factory=list)
    medium: list[float] = field(default_factory=list)
    no_verify: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.finetuned.clear()
        self.medium.clear()
        self.no_verify.clear()


def log_call_stt_latency_summary(acc: SttLatencyAccumulator, *, stream_sid: str) -> None:
    """통화당 한 번 — 갈래별 평균 STT 레이턴시(실제 `transcribe` 호출이 있었던 발화만 집계)."""

    def part(name: str, samples: list[float]) -> str:
        if not samples:
            return f"{name}: 표본 없음"
        avg = sum(samples) / len(samples)
        return f"{name}: 평균 {avg:.3f}s (n={len(samples)})"

    _logger.info(
        "[WS] 통화 종료 STT 레이턴시 요약 streamSid=%s — %s | %s | %s",
        stream_sid,
        part("finetuned", acc.finetuned),
        part("medium", acc.medium),
        part("no_verify", acc.no_verify),
    )


async def _timed_transcribe(
    stt: DeepgramSTTService,
    mulaw: bytes,
    *,
    t0: float,
    bucket: list[float] | None,
) -> str:
    try:
        return await stt.transcribe(mulaw)
    except Exception as e:
        return f"STT 실패: {e!s}"
    finally:
        if bucket is not None:
            bucket.append(time.perf_counter() - t0)


async def on_utterance_triple(
    mulaw_utt: bytes,
    call_id: str,
    stt_finetuned: DeepgramSTTService,
    stt_medium: DeepgramSTTService,
    stt_no_verify: DeepgramSTTService,
    latency_acc: SttLatencyAccumulator | None = None,
    *,
    utt_seq: int = 0,
) -> None:
    # VAD 발화 경계 직후(파이프라인 진입) — 각 갈래 STT 완료까지 동일 기준점
    t0 = time.perf_counter()
    finetuned = await get_finetuned_onnx_service_async()
    medium_svc = get_onnx_pipeline_service()
    pcm_utt = mulaw_to_pcm16_16k(mulaw_utt)
    gated = bool(settings.speaker_verify_enabled)
    b_ft = latency_acc.finetuned if latency_acc is not None else None
    b_md = latency_acc.medium if latency_acc is not None else None
    b_nv = latency_acc.no_verify if latency_acc is not None else None

    def enrollment_done() -> bool:
        return _onnx_branch_ready(finetuned, call_id) and _onnx_branch_ready(
            medium_svc, call_id
        )

    async def branch_finetuned(done: bool) -> str:
        # [no verify] 갈래와 무관: finetuned 는 검증·등록·오류 시 STT 호출 안 함.
        if finetuned.load_error:
            _logger.info("[finetuned]: 모델 오류 — STT 생략")
            return ""
        if gated and not done:
            _logger.info("[finetuned]: enrollment 미완료 — STT 생략")
            return ""
        if not gated:
            return await _timed_transcribe(stt_finetuned, mulaw_utt, t0=t0, bucket=b_ft)
        if not finetuned.has_voiceprint(call_id):
            _logger.info("[finetuned]: voiceprint 미등록 — STT 생략")
            return ""
        try:
            ok, sim = await finetuned.verify(pcm_utt, call_id)
        except Exception as e:
            _logger.info("[finetuned]: 검증 예외 — STT 생략 (%s)", e)
            return ""
        if not ok:
            if isinstance(sim, float) and math.isnan(sim):
                _logger.info("[finetuned]: 화자 불일치 (score=nan)")
            else:
                _logger.info("[finetuned]: 화자 불일치 (score=%.3f)", sim)
            return ""
        return await _timed_transcribe(stt_finetuned, mulaw_utt, t0=t0, bucket=b_ft)

    async def branch_medium(done: bool) -> str:
        if medium_svc.load_error:
            _logger.info("[medium]: 모델 오류 — STT 생략")
            return ""
        if gated and not done:
            _logger.info("[medium]: enrollment 미완료 — STT 생략")
            return ""
        if not gated:
            return await _timed_transcribe(stt_medium, mulaw_utt, t0=t0, bucket=b_md)
        if not medium_svc.has_voiceprint(call_id):
            _logger.info("[medium]: voiceprint 미등록 — STT 생략")
            return ""
        try:
            ok, sim = await medium_svc.verify(pcm_utt, call_id)
        except Exception as e:
            _logger.info("[medium]: 검증 예외 — STT 생략 (%s)", e)
            return ""
        if not ok:
            if isinstance(sim, float) and math.isnan(sim):
                _logger.info("[medium]: 화자 불일치 (score=nan)")
            else:
                _logger.info("[medium]: 화자 불일치 (score=%.3f)", sim)
            return ""
        return await _timed_transcribe(stt_medium, mulaw_utt, t0=t0, bucket=b_md)

    async def branch_no_verify_transcribe() -> str:
        return await _timed_transcribe(stt_no_verify, mulaw_utt, t0=t0, bucket=b_nv)

    done_flag = enrollment_done()

    if gated and not done_flag:
        t3 = await branch_no_verify_transcribe()
        if t3 and not t3.startswith("STT 실패"):
            await voice_enrollment.accumulate(call_id, pcm_utt, t3)
        done_flag = enrollment_done()
        r1, r2 = await asyncio.gather(
            branch_finetuned(done_flag),
            branch_medium(done_flag),
            return_exceptions=True,
        )
        t1 = _stt_text(r1)
        t2 = _stt_text(r2)
    else:
        async def branch_no_verify() -> str:
            t = await branch_no_verify_transcribe()
            if gated and not enrollment_done():
                await voice_enrollment.accumulate(call_id, pcm_utt, t)
            return t

        r1, r2, r3 = await asyncio.gather(
            branch_finetuned(done_flag),
            branch_medium(done_flag),
            branch_no_verify(),
            return_exceptions=True,
        )
        t1 = _stt_text(r1)
        t2 = _stt_text(r2)
        t3 = _stt_text(r3)

    _logger.info("[utt=%d] [finetuned]: %s", utt_seq, t1)
    _logger.info("[utt=%d] [medium]: %s", utt_seq, t2)
    _logger.info(
        "[utt=%d] [no verify, 화자검증 없음·항상 STT]: %s",
        utt_seq,
        t3,
    )
    spawn_nemo_vs_onnx_compare_task(pcm_utt, call_id=call_id, utt_seq=utt_seq)


@dataclass
class TripleStreamContext:
    """한 WebSocket 연결에 대해 3 VAD + 3 STT (인스턴스 분리로 STT race 방지)."""

    states: tuple[VADPipelineState, VADPipelineState, VADPipelineState] = field(
        default_factory=lambda: (VADPipelineState(), VADPipelineState(), VADPipelineState())
    )
    vads: tuple[SileroVADService, SileroVADService, SileroVADService] = field(
        default_factory=lambda: (
            SileroVADService(),
            SileroVADService(),
            SileroVADService(),
        )
    )
    stt_finetuned: DeepgramSTTService = field(default_factory=DeepgramSTTService)
    stt_medium: DeepgramSTTService = field(default_factory=DeepgramSTTService)
    stt_no_verify: DeepgramSTTService = field(default_factory=DeepgramSTTService)
    stt_latency: SttLatencyAccumulator = field(default_factory=SttLatencyAccumulator)
    utterance_seq: int = 0

    def reset(self) -> None:
        for s in self.states:
            s.reset()
        self.stt_latency.clear()
        self.utterance_seq = 0

    async def feed_media_chunk(self, mulaw: bytes, call_id: str) -> None:
        o1, o2, o3 = await asyncio.gather(
            self.states[0].feed_chunk(mulaw, self.vads[0]),
            self.states[1].feed_chunk(mulaw, self.vads[1]),
            self.states[2].feed_chunk(mulaw, self.vads[2]),
        )
        utt = o1 if o1 is not None else o2 if o2 is not None else o3
        if utt is None:
            return
        if o1 != o2 or o2 != o3:
            _logger.warning(
                "VAD 발화 경계 불일치 (비트 동일 아님) — 첫 비어있지 않은 값 사용"
            )
            utt = o1 or o2 or o3
        self.utterance_seq += 1
        await on_utterance_triple(
            utt,
            call_id,
            self.stt_finetuned,
            self.stt_medium,
            self.stt_no_verify,
            self.stt_latency,
            utt_seq=self.utterance_seq,
        )

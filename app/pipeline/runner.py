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
from app.services.stt.deepgram_streaming import DeepgramLiveMulawSession
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
        _logger.warning("STT 예외: %s", result)
        return f"STT 실패: {result!s}"
    return result or ""


def _pair_from_branch_gather(res: Any) -> tuple[str, str | None]:
    """branch_finetuned / branch_medium 의 gather 결과 → (stt_text, 검증_스킵_사유)."""
    if isinstance(res, BaseException):
        return "", f"처리 예외: {res!s}"
    if isinstance(res, tuple) and len(res) == 2:
        a, b = res[0], res[1]
        return (a if isinstance(a, str) else ""), (
            b if isinstance(b, str) or b is None else None
        )
    if isinstance(res, str):
        return res, None
    return "", "알 수 없는 branch 반환"


def _onnx_branch_ready(svc: Any, call_id: str) -> bool:
    """로드 실패 갈래는 제외하고, 활성 ONNX는 voiceprint가 있어야 검증·STT 단계로 진입."""
    if svc.load_error:
        return True
    return svc.has_voiceprint(call_id)


@dataclass
class SttLatencyAccumulator:
    """STT 레이턴시 초 단위 샘플.

    vad_prerecorded: 발화 경계부터 `transcribe` 완료까지.
    deepgram_stream: 해당 발화 구간의 추정 종료 시각(첫 오디오 wall + start+duration)부터
    `is_final` 수신까지(Deepgram 꼬리 지연·엔드포인팅 대략치).
    """

    finetuned: list[float] = field(default_factory=list)
    medium: list[float] = field(default_factory=list)
    no_verify: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.finetuned.clear()
        self.medium.clear()
        self.no_verify.clear()


def log_call_stt_latency_summary(acc: SttLatencyAccumulator, *, stream_sid: str) -> None:
    """통화당 한 번 — 갈래별 평균 STT 레이턴시(스트림/VAD 모드별 의미는 SttLatencyAccumulator 주석 참고)."""

    def part(name: str, samples: list[float]) -> str:
        if not samples:
            return f"{name}: 표본 없음"
        avg = sum(samples) / len(samples)
        return f"{name}: 평균 {avg:.3f}s (n={len(samples)})"

    mode = "deepgram_stream" if settings.deepgram_use_streaming else "vad_prerecorded"
    _logger.info(
        "[WS] 통화 종료 STT 레이턴시 요약 mode=%s streamSid=%s — %s | %s | %s",
        mode,
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
    transcript_override: str | None = None,
    stream_stt_latency_sec: float | None = None,
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

    if (
        transcript_override is not None
        and stream_stt_latency_sec is not None
        and latency_acc is not None
    ):
        for bucket in (latency_acc.finetuned, latency_acc.medium, latency_acc.no_verify):
            bucket.append(stream_stt_latency_sec)

    def enrollment_done() -> bool:
        return _onnx_branch_ready(finetuned, call_id) and _onnx_branch_ready(
            medium_svc, call_id
        )

    async def _text_or_transcribe(
        stt: DeepgramSTTService, mulaw: bytes, bucket: list[float] | None
    ) -> str:
        if transcript_override is not None:
            return transcript_override
        return await _timed_transcribe(stt, mulaw, t0=t0, bucket=bucket)

    async def branch_finetuned(done: bool) -> tuple[str, str | None]:
        """(STT 텍스트 또는 빈 문자열, 검증·모델 사유 — None 이면 STT 경로까지 진행)."""
        if finetuned.load_error:
            return "", "모델 로드 실패"
        if gated and not done:
            return "", "enrollment 미완료"
        if not gated:
            t = await _text_or_transcribe(stt_finetuned, mulaw_utt, b_ft)
            return t, None
        if not finetuned.has_voiceprint(call_id):
            return "", "voiceprint 미등록"
        try:
            ok, sim = await finetuned.verify(pcm_utt, call_id)
        except Exception as e:
            return "", f"검증 예외: {e}"
        if not ok:
            if isinstance(sim, float) and math.isnan(sim):
                return "", "화자 불일치 (similarity=nan)"
            return "", f"화자 불일치 (similarity={float(sim):.4f})"
        t = await _text_or_transcribe(stt_finetuned, mulaw_utt, b_ft)
        return t, None

    async def branch_medium(done: bool) -> tuple[str, str | None]:
        if medium_svc.load_error:
            return "", "모델 로드 실패"
        if gated and not done:
            return "", "enrollment 미완료"
        if not gated:
            t = await _text_or_transcribe(stt_medium, mulaw_utt, b_md)
            return t, None
        if not medium_svc.has_voiceprint(call_id):
            return "", "voiceprint 미등록"
        try:
            ok, sim = await medium_svc.verify(pcm_utt, call_id)
        except Exception as e:
            return "", f"검증 예외: {e}"
        if not ok:
            if isinstance(sim, float) and math.isnan(sim):
                return "", "화자 불일치 (similarity=nan)"
            return "", f"화자 불일치 (similarity={float(sim):.4f})"
        t = await _text_or_transcribe(stt_medium, mulaw_utt, b_md)
        return t, None

    async def branch_no_verify_transcribe() -> str:
        return await _text_or_transcribe(stt_no_verify, mulaw_utt, b_nv)

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
        t1, skip1 = _pair_from_branch_gather(r1)
        t2, skip2 = _pair_from_branch_gather(r2)
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
        t1, skip1 = _pair_from_branch_gather(r1)
        t2, skip2 = _pair_from_branch_gather(r2)
        t3 = _stt_text(r3)

    def _log_model_branch(tag: str, text: str, skip: str | None) -> None:
        if skip:
            _logger.info("[utt=%d] [%s] 검증 실패 — %s", utt_seq, tag, skip)
            return
        if text.startswith("STT 실패"):
            _logger.info("[utt=%d] [%s] %s", utt_seq, tag, text)
            return
        _logger.info("[utt=%d] [%s] %s", utt_seq, tag, text if text.strip() else "(빈 인식)")

    _log_model_branch("finetuned", t1, skip1)
    _log_model_branch("medium", t2, skip2)
    if t3.startswith("STT 실패"):
        _logger.info("[utt=%d] [no_verify] %s", utt_seq, t3)
    else:
        _logger.info(
            "[utt=%d] [no_verify] %s",
            utt_seq,
            t3 if t3.strip() else "(빈 인식)",
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
    _dg_live: DeepgramLiveMulawSession | None = field(default=None, repr=False)
    _stream_call_id: str | None = field(default=None, repr=False)
    _dg_start_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def reset(self) -> None:
        for s in self.states:
            s.reset()
        self.stt_latency.clear()
        self.utterance_seq = 0
        # Deepgram 세션은 async 로 닫음 — prepare / 첫 media 에서 열고 shutdown 에서 정리

    async def shutdown_deepgram_streaming(self) -> None:
        if self._dg_live is not None:
            await self._dg_live.close()
            self._dg_live = None
        self._stream_call_id = None

    async def prepare_deepgram_streaming(self, call_id: str) -> None:
        """Twilio `start` 직후 — call_id 만 저장. Deepgram WS 는 첫 `media` 에서 연다(연결~첫 오디오 공백 타임아웃 방지)."""
        if not settings.deepgram_use_streaming:
            return
        await self.shutdown_deepgram_streaming()
        self._stream_call_id = call_id

    async def _ensure_deepgram_live(self) -> None:
        """첫 오디오 직전에 Deepgram live 세션 1회 생성."""
        if not settings.deepgram_use_streaming or self._stream_call_id is None:
            return
        if self._dg_live is not None:
            return
        async with self._dg_start_lock:
            if self._dg_live is not None:
                return
            triple = self
            cid0 = self._stream_call_id

            async def _on_final(
                mulaw_seg: bytes, transcript: str, _start: float, _dur: float, e2e: float
            ) -> None:
                cid = triple._stream_call_id or cid0 or "no-stream"
                triple.utterance_seq += 1
                await on_utterance_triple(
                    mulaw_seg,
                    cid,
                    triple.stt_finetuned,
                    triple.stt_medium,
                    triple.stt_no_verify,
                    triple.stt_latency,
                    utt_seq=triple.utterance_seq,
                    transcript_override=transcript,
                    stream_stt_latency_sec=e2e,
                )

            sess = DeepgramLiveMulawSession(on_final=_on_final)
            await sess.start()
            self._dg_live = sess

    async def feed_media_chunk(self, mulaw: bytes, call_id: str) -> None:
        if settings.deepgram_use_streaming:
            await self._ensure_deepgram_live()
            if self._dg_live is not None:
                await self._dg_live.feed(mulaw)
                return

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

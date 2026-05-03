"""Twilio μ-law 8k → Deepgram live listen WebSocket (nova-3).

`is_final` 전사마다 콜백으로 오디오 구간(버퍼 슬라이스)·텍스트·타이밍을 넘긴다.
"""
from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from deepgram.options import DeepgramClientOptions

from app.utils.config import settings
from app.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_MULAW_SR = 8000


class DeepgramLiveMulawSession:
    """한 통화(streamSid)당 하나 — μ-law 누적 버퍼 + Deepgram async websocket."""

    def __init__(
        self,
        *,
        on_final: Callable[[bytes, str, float, float, float], Awaitable[None]],
    ) -> None:
        self._on_final = on_final
        self._buffer = bytearray()
        self._client: DeepgramClient | None = None
        self._conn: object | None = None
        self._open_monotonic: float | None = None
        # 첫 μ-law 수신 시각 — Deepgram `start`/`duration`(스트림 초)을 wall-clock 에 매핑할 때 사용
        self._first_audio_wall_mono: float | None = None

    @property
    def mulaw_total_bytes(self) -> int:
        return len(self._buffer)

    def _slice_mulaw(self, start_sec: float, duration_sec: float) -> bytes:
        i0 = max(0, int(start_sec * _MULAW_SR))
        i1 = min(len(self._buffer), int(math.ceil((start_sec + duration_sec) * _MULAW_SR)))
        if i1 <= i0:
            return b""
        return bytes(self._buffer[i0:i1])

    async def start(self) -> None:
        # keep_alive: 오디오 청크 사이에도 Deepgram 타임아웃(net0001) 방지
        dg_cfg = DeepgramClientOptions(
            api_key=settings.deepgram_api_key,
            options={"keep_alive": "true"},
        )
        self._client = DeepgramClient(settings.deepgram_api_key, config=dg_cfg)
        conn = self._client.listen.asyncwebsocket.v("1")

        # SDK: (1) start() 직후 synthetic Open 은 OpenResponse 를 *위치* 인자로 넘김
        #     (2) 수신 루프에서는 open=OpenResponse 키워드로 넘김 — 둘 다 수용
        async def _on_open(_dg, *args, **kwargs) -> None:
            self._open_monotonic = time.perf_counter()
            self._first_audio_wall_mono = None

        async def _on_error(_dg, **kwargs) -> None:
            err = kwargs.get("error")
            logger.warning("Deepgram 스트림 오류: %s", err)

        async def _on_transcript(_dg, result, **_kwargs) -> None:
            if not getattr(result, "is_final", False):
                return
            try:
                ch = result.channel
                alts = getattr(ch, "alternatives", None) or []
                if not alts:
                    return
                sentence = (alts[0].transcript or "").strip()
            except Exception as e:
                logger.debug("Deepgram Transcript 파싱 생략: %s", e)
                return
            if not sentence:
                return
            start = float(getattr(result, "start", 0.0) or 0.0)
            duration = float(getattr(result, "duration", 0.0) or 0.0)
            mulaw_seg = self._slice_mulaw(start, duration)
            now = time.perf_counter()
            # 이전 식(now - open - start - duration)은 스트림 타임(초)과 wall-clock 을 섞어 대부분 0으로 클램프됨.
            # Twilio 가 대략 실시간으로 보낸다고 가정: 스트림 t=start+duration 에 해당하는 wall 시각 ≈
            #   첫 오디오 수신 시각 + (start+duration) 초
            if self._first_audio_wall_mono is not None:
                est_utterance_end_wall = self._first_audio_wall_mono + start + duration
                e2e = max(0.0, now - est_utterance_end_wall)
            elif self._open_monotonic is not None:
                est_utterance_end_wall = self._open_monotonic + start + duration
                e2e = max(0.0, now - est_utterance_end_wall)
            else:
                e2e = 0.0
            await self._on_final(mulaw_seg, sentence, start, duration, e2e)

        conn.on(LiveTranscriptionEvents.Open, _on_open)
        conn.on(LiveTranscriptionEvents.Error, _on_error)
        conn.on(LiveTranscriptionEvents.Transcript, _on_transcript)

        options = LiveOptions(
            model="nova-3",
            language="ko",
            encoding="mulaw",
            sample_rate=8000,
            smart_format=True,
            punctuate=True,
            interim_results=True,
            endpointing=400,
        )

        ok = await conn.start(options)
        if not ok:
            raise RuntimeError("Deepgram live listen start 실패")
        self._conn = conn
        logger.info("Deepgram live listen 연결됨")

    async def feed(self, mulaw: bytes) -> None:
        if not mulaw:
            return
        if self._first_audio_wall_mono is None:
            self._first_audio_wall_mono = time.perf_counter()
        self._buffer.extend(mulaw)
        if self._conn is None:
            return
        await self._conn.send(mulaw)

    async def close(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is None:
            return
        try:
            await conn.finalize()
        except Exception as e:
            logger.debug("Deepgram finalize 생략: %s", e)
        try:
            await conn.finish()
        except Exception as e:
            logger.debug("Deepgram finish: %s", e)
        logger.info("Deepgram live listen 종료 (mulaw 누적=%d bytes)", len(self._buffer))

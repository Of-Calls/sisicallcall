"""XTTS v2 (Coqui TTS) — 팀원 목소리 zero-shot voice cloning.

GPU 자동 감지: CUDA 가용 시 GPU(권장), 아니면 CPU(데모/오프라인 합성용).
실시간 통화에서는 CPU로는 5초 하드컷 위반 가능성 높음 — GPU 환경에서만 운영 권장.

reference audio:
    settings.xtts_reference_path 의 WAV 파일을 1회 로드 후 voice embedding 캐싱.
    매 합성 시 reference 재처리 없이 캐시된 embedding 사용 → 합성 속도 최적화.

audio pipeline:
    XTTS v2 출력 (24kHz float [-1, 1]) → int16 PCM → 8kHz μ-law (Twilio 호환)
"""
import asyncio
import audioop
import os
import re

from app.services.tts.base import BaseTTSService
from app.utils.config import settings
from app.utils.logger import get_logger

# Coqui TTS 모델 라이선스(CPML) 자동 동의 — 비대화형 환경(서버 등)에서
# input() 프롬프트로 EOFError 발생 방지. XTTS v2 사용은 본 동의를 전제로 함.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

logger = get_logger(__name__)


class XTTSService(BaseTTSService):
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    LANGUAGE = "ko"
    NATIVE_SAMPLE_RATE = 24000  # XTTS v2 출력 샘플레이트
    TARGET_SAMPLE_RATE = 8000   # Twilio Media Stream 입력 샘플레이트
    MAX_KO_CHARS = 90           # XTTS v2 한국어 입력 한계 95자, 안전 마진 적용
    INTER_CHUNK_SILENCE_MS = 600  # 청크 사이 무음 — 자연스러운 호흡감 (~0.6초)

    # 한국어 시간 표현은 시(時)를 순우리말로 읽는 게 자연스러움.
    # 09:00 → "아홉 시" (○), "구 시" (X)
    _KOREAN_HOUR = {
        1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯",
        7: "일곱", 8: "여덟", 9: "아홉", 10: "열", 11: "열한", 12: "열두",
        13: "열세", 14: "열네", 15: "열다섯", 16: "열여섯", 17: "열일곱",
        18: "열여덟", 19: "열아홉", 20: "스물", 21: "스물한", 22: "스물두",
        23: "스물세", 24: "스물네",
    }

    def __init__(self):
        # 지연 초기화 — 앱 import 시점에 1.8GB 모델 로딩 회피.
        self._model = None
        self._device = None
        self._gpt_cond_latent = None
        self._speaker_embedding = None

    def _ensure_model(self):
        """첫 호출 시 모델 로드 + reference voice embedding 추출 + 캐싱."""
        if self._model is not None:
            return

        import torch
        from TTS.api import TTS as CoquiTTS

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            logger.warning(
                "XTTS v2 가 CPU에서 실행됩니다 — 합성 시간 3~8초 예상. "
                "실시간 통화에는 GPU 필수."
            )

        logger.info("XTTS v2 모델 로딩 중 device=%s ...", self._device)
        model = CoquiTTS(self.MODEL_NAME, progress_bar=False)
        model.to(self._device)
        self._model = model

        # Voice embedding 캐싱 — reference WAV 1회 처리
        ref_path = settings.xtts_reference_path
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"XTTS reference audio not found: {ref_path} "
                f"(.env 의 XTTS_REFERENCE_PATH 확인)"
            )

        logger.info("XTTS voice embedding 추출 중 reference=%s", ref_path)
        gpt_cond, speaker_emb = (
            self._model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[ref_path]
            )
        )
        self._gpt_cond_latent = gpt_cond
        self._speaker_embedding = speaker_emb
        logger.info("XTTS v2 모델 로딩 + voice embedding 캐싱 완료")

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """TTS 입력 전처리 — 시간 표기를 자연스러운 한국어 12시간제로 변환.

        XTTS 가 콜론(:) 등 부호를 발음 가이드로 인식하지 못해 어색하게 읽는 문제 해결.
        24시간제는 한국어 일상 표현에 어색하므로 오전/오후 + 순우리말 12시간제로 변환.
        예:
            "09:00" → "오전 아홉 시"
            "12:30" → "오후 열두 시 30분"
            "17:30" → "오후 다섯 시 30분"
            "00:00" → "오전 열두 시"
        """
        def time_replacer(m: re.Match) -> str:
            h = int(m.group(1))
            mm = int(m.group(2))

            # 24시간제 → 12시간제 변환
            if h == 0:
                period, h12 = "오전", 12
            elif h < 12:
                period, h12 = "오전", h
            elif h == 12:
                period, h12 = "오후", 12
            else:
                period, h12 = "오후", h - 12

            h_text = cls._KOREAN_HOUR.get(h12, str(h12))
            if mm == 0:
                return f"{period} {h_text} 시"
            return f"{period} {h_text} 시 {mm}분"
        return re.sub(r"(\d{1,2}):(\d{2})", time_replacer, text)

    def _split_for_synthesis(self, text: str) -> list[str]:
        """긴 텍스트를 합성 가능 청크로 분할.

        1차: 마침표/물음표/느낌표 단위
        2차: 청크가 한계 초과 시 쉼표 단위 추가 분할
        """
        max_chars = self.MAX_KO_CHARS
        sentences = re.split(r"(?<=[.!?。])\s*", text)
        chunks: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) <= max_chars:
                chunks.append(s)
                continue
            # 쉼표 단위 2차 분할 + 누적 버퍼링
            parts = re.split(r"(?<=[,，、])\s*", s)
            buf = ""
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                joined = (buf + " " + p).strip() if buf else p
                if len(joined) <= max_chars:
                    buf = joined
                else:
                    if buf:
                        chunks.append(buf)
                    buf = p
            if buf:
                chunks.append(buf)
        return chunks

    def _synthesize_sync(self, text: str) -> bytes:
        """동기 합성 — 텍스트 정규화 → 청크 분할 → 순차 합성 → 8kHz μ-law 변환."""
        self._ensure_model()
        import numpy as np

        normalized = self._normalize_text(text)
        chunks = self._split_for_synthesis(normalized)
        if not chunks:
            return b""

        wavs: list[np.ndarray] = []
        for chunk in chunks:
            out = self._model.synthesizer.tts_model.inference(
                text=chunk,
                language=self.LANGUAGE,
                gpt_cond_latent=self._gpt_cond_latent,
                speaker_embedding=self._speaker_embedding,
                temperature=0.7,
            )
            wavs.append(np.asarray(out["wav"], dtype=np.float32))

        if len(wavs) == 1:
            full_wav = wavs[0]
        else:
            # 청크 사이에 무음 패딩 — 자연스러운 호흡감 부여
            silence_samples = int(self.NATIVE_SAMPLE_RATE * self.INTER_CHUNK_SILENCE_MS / 1000)
            silence = np.zeros(silence_samples, dtype=np.float32)
            interleaved: list[np.ndarray] = []
            for i, w in enumerate(wavs):
                if i > 0:
                    interleaved.append(silence)
                interleaved.append(w)
            full_wav = np.concatenate(interleaved)

        # float [-1.0, 1.0] → int16 PCM
        wav_int16 = (np.clip(full_wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        # 24kHz → 8kHz 다운샘플링
        wav_8k, _ = audioop.ratecv(
            wav_int16, 2, 1,
            self.NATIVE_SAMPLE_RATE, self.TARGET_SAMPLE_RATE,
            None,
        )

        # 16-bit PCM → μ-law (Twilio Media Stream 포맷)
        return audioop.lin2ulaw(wav_8k, 2)

    async def synthesize(self, text: str) -> bytes:
        """텍스트 → μ-law 8kHz mono bytes (Twilio Media Stream 호환).

        XTTS는 동기 코드라서 run_in_executor로 비동기 래핑.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

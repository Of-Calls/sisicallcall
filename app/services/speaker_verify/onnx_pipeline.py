"""ONNXRuntime(mel 입력) 화자 검증 — 병렬 파이프라인용.

파인튜닝·medium 등 **모델 파일만 다르고** 추론 계약은 동일:
  mel [B, n_mels, T] + length → logits / embedding (`titanet.py` ONNX 분기와 동일).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import torch

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.services.speaker_verify.titanet_mel import pcm16_bytes_to_float_mono
from app.services.speaker_verify.titanet_mel_nemo import build_onnx_mel_frontend
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEVICE_CPU = torch.device("cpu")
_MIN_MEL_FRAMES = 8
# 구 export ONNX 등 그래프 내부 T 가 ~1200 으로 박힐 때 Where 브로드캐스트(1200×1201) 방지용 안전 상한.
_FINETUNED_ONNX_SAFE_DEFAULT_MEL_FRAMES = 1200


def _effective_finetuned_mel_frame_cap() -> int:
    """TITANET_FINETUNED_ONNX_MAX_MEL_FRAMES: >0 이면 그 값, 0 이면 안전 기본 1200 (length 와 mel 슬라이스 T 일치)."""
    v = int(settings.titanet_finetuned_onnx_max_mel_frames)
    return v if v > 0 else _FINETUNED_ONNX_SAFE_DEFAULT_MEL_FRAMES


def _app_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _onnx_embedding_from_output(out: np.ndarray) -> np.ndarray:
    """ONNX 출력 텐서 → 1D 임베딩 (`titanet.py`와 동일).

    ORT 출력은 다음 `run()` 시 버퍼가 재사용될 수 있으므로 항상 독립 ndarray 로 복사한다.
    """
    a = np.asarray(out, dtype=np.float32)
    if a.ndim == 1:
        vec = a
    elif a.ndim == 2:
        vec = a[0]
    elif a.ndim == 3:
        if a.shape[1] <= a.shape[2]:
            vec = a[0].mean(axis=-1)
        else:
            vec = a[0].mean(axis=1)
    else:
        vec = a.reshape(-1)
    return np.array(vec, dtype=np.float32, copy=True).reshape(-1)


def _resolve_onnx_path(settings_field: str, default_filename: str) -> Path | None:
    raw = (getattr(settings, settings_field, "") or "").strip()
    p = (
        Path(raw)
        if raw
        else _app_root() / "models" / "speaker_verification" / default_filename
    )
    if p.is_file():
        return p
    if raw:
        logger.error("%s: 설정 경로에 파일 없음 path=%s", settings_field, p)
    else:
        logger.error("%s: 기본 경로에 파일 없음 path=%s", settings_field, p)
    return None


class OnnxMelSpeakerVerifyService(BaseSpeakerVerifyService):
    """단일 ONNX 파일 + mel 전처리 + 코사인 검증.

    ONNX `InferenceSession` 은 **인스턴스 필드** `self._ort_sess` 만 사용한다.
    클래스 변수로 세션을 두지 않는다(파이프라인 간 공유 방지).
    """

    def __init__(
        self,
        *,
        settings_field: str,
        default_filename: str,
        log_tag: str,
        infer_max_pcm_sec: float | None = None,
    ) -> None:
        self._settings_field = settings_field
        self._default_filename = default_filename
        self._log_tag = log_tag
        self._infer_max_pcm_bytes: int | None = None
        if infer_max_pcm_sec is not None and infer_max_pcm_sec > 0:
            self._infer_max_pcm_bytes = int(16000 * 2 * infer_max_pcm_sec)
        self._voiceprints: dict[str, np.ndarray] = {}
        self._ort_sess = None
        self._mel = None
        self._mel_init_lock = threading.Lock()
        self._onnx_in_audio = ""
        self._onnx_in_length: str | None = None
        self._onnx_out_emb_idx = 0
        self.load_error: str | None = None
        self.onnx_resolved_path: str = ""

        path = _resolve_onnx_path(settings_field, default_filename)
        if path is None:
            self.load_error = "모델 로드 실패"
            return
        self.onnx_resolved_path = str(path.resolve())
        try:
            import onnxruntime as ort

            t0 = time.monotonic()
            logger.info("[%s] ONNX 로딩 path=%s …", log_tag, path)
            providers: list[str] = []
            if torch.cuda.is_available():
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            # 세션마다 독립 SessionOptions — ORT 내부 캐시/공유 이슈 회피
            sess_opt = ort.SessionOptions()
            self._ort_sess = ort.InferenceSession(
                str(path), sess_options=sess_opt, providers=providers
            )
            ins = self._ort_sess.get_inputs()
            if not ins:
                raise RuntimeError("ONNX 모델에 입력이 없습니다.")
            self._onnx_in_audio = ins[0].name
            self._onnx_in_length = ins[1].name if len(ins) > 1 else None
            outs_meta = self._ort_sess.get_outputs()
            self._onnx_out_emb_idx = 1 if len(outs_meta) > 1 else 0
            # mel(NeMo 포함)은 첫 임베딩·검증 시 로드 — 기동 시 restore 분리
            logger.info(
                "[%s] ONNX Runtime 준비 완료 elapsed=%.2fs emb_out_idx=%s (mel 은 첫 사용 시)",
                log_tag,
                time.monotonic() - t0,
                self._onnx_out_emb_idx,
            )
        except Exception as e:
            self.load_error = "모델 로드 실패"
            logger.exception("[%s] 로드 실패: %s", log_tag, e)

    def _ensure_mel_frontend(self) -> None:
        """torchaudio 또는 NeMo mel 모듈 지연 초기화 (NeMo restore 는 여기서 최초 1회)."""
        if self.load_error or self._ort_sess is None:
            return
        if self._mel is not None:
            return
        with self._mel_init_lock:
            if self._mel is not None:
                return
            try:
                t0 = time.monotonic()
                self._mel = build_onnx_mel_frontend(_DEVICE_CPU, log_tag=self._log_tag)
                logger.info(
                    "[%s] mel 프론트엔드 준비 완료 elapsed=%.2fs",
                    self._log_tag,
                    time.monotonic() - t0,
                )
            except Exception as e:
                self.load_error = "모델 로드 실패"
                self._mel = None
                logger.exception("[%s] mel 프론트 로드 실패: %s", self._log_tag, e)

    @property
    def session(self):
        """onnxruntime InferenceSession (기동 시 medium≠finetuned 세션 id 확인용)."""
        return self._ort_sess

    @property
    def onnx_path(self) -> str:
        """로드에 사용한 ONNX 파일 절대 경로(미로드 시 빈 문자열)."""
        return self.onnx_resolved_path or ""

    def _similarity_threshold(self) -> float:
        if self._settings_field == "titanet_pipeline_onnx_path":
            v = settings.speaker_verify_medium_threshold
            return settings.speaker_verify_threshold if v is None else float(v)
        if self._settings_field == "titanet_finetuned_onnx_path":
            v = settings.speaker_verify_finetuned_threshold
            return settings.speaker_verify_threshold if v is None else float(v)
        return settings.speaker_verify_threshold

    def _mel_pad_if_needed(self, mel: np.ndarray) -> np.ndarray:
        n_mels = self._mel.params.n_mels if self._mel is not None else 80
        if mel.ndim != 3 or mel.shape[1] != n_mels:
            return mel
        tdim = mel.shape[2]
        if tdim >= _MIN_MEL_FRAMES:
            return mel
        pad = _MIN_MEL_FRAMES - tdim
        return np.pad(mel, ((0, 0), (0, 0), (0, pad)), mode="constant")

    def _cap_mel_time_finetuned_onnx(self, mel: np.ndarray) -> np.ndarray:
        """finetuned 전용: mel [1,n_mels,T] 의 T 를 ONNX 가 견딜 상한 이하로 자름. length 는 run 시 shape[2] 와 동일."""
        if self._settings_field != "titanet_finetuned_onnx_path":
            return mel
        mx = _effective_finetuned_mel_frame_cap()
        tdim = int(mel.shape[2])
        if tdim <= mx:
            return mel
        logger.info(
            "[%s] finetuned ONNX mel 시간축 자름 T=%d→%d (적용상한=%d, TITANET_FINETUNED_ONNX_MAX_MEL_FRAMES=%d, 0이면 기본 %d)",
            self._log_tag,
            tdim,
            mx,
            mx,
            int(settings.titanet_finetuned_onnx_max_mel_frames),
            _FINETUNED_ONNX_SAFE_DEFAULT_MEL_FRAMES,
        )
        return np.ascontiguousarray(mel[:, :, :mx], dtype=np.float32)

    def _pcm_to_mel_np(self, audio_chunk: bytes) -> np.ndarray:
        """PCM16 mono → mel [1,n_mels,T] (pad 포함). `_extract_embedding_sync` 와 동일 전처리."""
        self._ensure_mel_frontend()
        if self._mel is None:
            raise RuntimeError(self.load_error or "ONNX 미초기화")
        pcm = audio_chunk
        # PCM 을 초로 자르도 STFT center/hop 정렬 때문에 mel 프레임 T 가 1200·1201 경계를 넘을 수 있음
        # → finetuned 경로는 _cap_mel_time_finetuned_onnx 로 별도 T 상한 적용.
        if (
            self._infer_max_pcm_bytes is not None
            and len(pcm) > self._infer_max_pcm_bytes
        ):
            logger.info(
                "[%s] 긴 발화 임베딩: 앞 %d bytes(%.2fs)만 사용 (원본 %d bytes)",
                self._log_tag,
                self._infer_max_pcm_bytes,
                self._infer_max_pcm_bytes / 32000.0,
                len(pcm),
            )
            pcm = pcm[: self._infer_max_pcm_bytes]
        samples = pcm16_bytes_to_float_mono(pcm)
        mel = self._mel(samples)
        mel = self._mel_pad_if_needed(mel)
        mel = self._cap_mel_time_finetuned_onnx(mel)
        return np.ascontiguousarray(mel, dtype=np.float32)

    def _embedding_from_mel_np(self, mel: np.ndarray) -> np.ndarray:
        """mel [1,n_mels,T] → ONNX 임베딩 1D (`_extract_embedding_sync` 후반)."""
        if self._ort_sess is None:
            raise RuntimeError(self.load_error or "ONNX 미초기화")
        if self._settings_field == "titanet_finetuned_onnx_path":
            mel = self._cap_mel_time_finetuned_onnx(mel)
        mel = np.ascontiguousarray(mel, dtype=np.float32)
        feeds: dict[str, np.ndarray] = {self._onnx_in_audio: mel}
        if self._onnx_in_length is not None:
            t_frames = int(mel.shape[2])
            feeds[self._onnx_in_length] = np.array([t_frames], dtype=np.int64)
        outs = self._ort_sess.run(None, feeds)
        if not outs or len(outs) <= self._onnx_out_emb_idx:
            raise RuntimeError("ONNX 출력 없음")
        raw = outs[self._onnx_out_emb_idx]
        return _onnx_embedding_from_output(raw)

    def _extract_embedding_sync(self, audio_chunk: bytes) -> np.ndarray:
        if self._ort_sess is None:
            raise RuntimeError(self.load_error or "ONNX 미초기화")
        mel = self._pcm_to_mel_np(audio_chunk)
        return self._embedding_from_mel_np(mel)

    async def _extract_embedding(self, audio_chunk: bytes) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._extract_embedding_sync, audio_chunk
        )

    async def extract_embedding_for_enroll(self, audio_chunk: bytes) -> np.ndarray:
        """다발화 등록: PCM 당 1회 임베딩만 추출(저장 없음). ORT 버퍼와 분리된 소유 복사."""
        if self.load_error or self._ort_sess is None:
            raise RuntimeError(self.load_error or "ONNX 미초기화")
        emb = await self._extract_embedding(audio_chunk)
        return np.asarray(emb, dtype=np.float32).reshape(-1).copy()

    def store_voiceprint_vector(self, call_id: str, vector: np.ndarray) -> None:
        """평균·L2 정규화 등 최종 등록 벡터 저장(항상 소유 복사)."""
        if self.load_error:
            return
        self._voiceprints[call_id] = np.asarray(vector, dtype=np.float32).reshape(-1).copy()

    def voiceprint_slice_first_n(self, call_id: str, n: int = 4) -> list[float] | None:
        """등록 완료 후 로그용 — 앞 n 스칼라(없으면 None)."""
        vp = self._voiceprints.get(call_id)
        if vp is None:
            return None
        return [float(x) for x in np.asarray(vp, dtype=np.float32).ravel()[:n]]

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        if self.load_error:
            return
        try:
            embedding = await self._extract_embedding(audio_chunk)
            # 파이프라인·세션 간 ORT 버퍼/뷰 공유 방지 — 등록 벡터는 반드시 소유 복사
            self._voiceprints[call_id] = np.asarray(
                embedding, dtype=np.float32
            ).reshape(-1).copy()
            logger.info(
                "[%s] voiceprint 등록 call_id=%s dim=%d arr_id=%d",
                self._log_tag,
                call_id,
                int(self._voiceprints[call_id].shape[0]),
                id(self._voiceprints[call_id]),
            )
        except Exception as e:
            logger.error("[%s] extract_and_store 실패: %s", self._log_tag, e)

    async def verify(self, audio_chunk: bytes, call_id: str) -> tuple[bool, float]:
        if self.load_error or self._ort_sess is None:
            return False, 0.0
        if call_id not in self._voiceprints:
            logger.info(
                "[%s] voiceprint 미등록 → 검증 불가 call_id=%s",
                self._log_tag,
                call_id,
            )
            return False, 0.0
        try:
            embedding = await self._extract_embedding(audio_chunk)
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()
            voiceprint = self._voiceprints[call_id]
            similarity = float(
                np.dot(embedding, voiceprint)
                / (np.linalg.norm(embedding) * np.linalg.norm(voiceprint))
            )
            thr = self._similarity_threshold()
            is_verified = similarity >= thr
            logger.info(
                "[%s] 검증 call_id=%s similarity=%.4f threshold=%.2f verified=%s",
                self._log_tag,
                call_id,
                similarity,
                thr,
                is_verified,
            )
            return is_verified, similarity
        except Exception as e:
            logger.error("[%s] verify 실패 call_id=%s: %s", self._log_tag, call_id, e)
            return False, 0.0

    def has_voiceprint(self, call_id: str) -> bool:
        """등록 완료 여부(검증 전제)."""
        return call_id in self._voiceprints

    def debug_voiceprint_array_id(self, call_id: str) -> int | None:
        """등록된 voiceprint ndarray 의 id — medium vs finetuned 공유 참조 진단용."""
        vp = self._voiceprints.get(call_id)
        return id(vp) if vp is not None else None

    def debug_voiceprint_head(self, call_id: str, n: int = 4) -> str:
        """등록 벡터 앞 n 값(로그용). 공유·동일 모델 여부 확인."""
        vp = self._voiceprints.get(call_id)
        if vp is None:
            return "(none)"
        flat = np.asarray(vp, dtype=np.float32).ravel()[:n]
        return str([float(x) for x in flat])

    def voiceprint_store_dict_id(self) -> int:
        """call_id → ndarray 저장 dict 객체 id (medium vs finetuned 별도 dict 확인)."""
        return id(self._voiceprints)

    def cleanup(self, call_id: str) -> bool:
        """해당 통화 voiceprint 제거. 제거했으면 True."""
        return self._voiceprints.pop(call_id, None) is not None

    def ort_inference_session_id(self) -> int | None:
        """onnxruntime InferenceSession 객체 id (기동 시 medium≠finetuned 확인용)."""
        return id(self._ort_sess) if self._ort_sess is not None else None

    def clear_all_voiceprints(self) -> int:
        """메모리에 있는 모든 call_id voiceprint 삭제. 삭제 건수 반환."""
        n = len(self._voiceprints)
        self._voiceprints.clear()
        logger.info("[%s] voiceprint 전체 초기화 (%d건)", self._log_tag, n)
        return n


_singleton_lock = threading.Lock()
_singleton_medium: OnnxMelSpeakerVerifyService | None = None
_singleton_finetuned: OnnxMelSpeakerVerifyService | None = None


def get_onnx_pipeline_service() -> OnnxMelSpeakerVerifyService:
    """medium 학습 ONNX (`titanet_small_medium_5epoch_lr5e5.onnx` 기본)."""
    global _singleton_medium, _singleton_finetuned
    if _singleton_medium is None:
        with _singleton_lock:
            if _singleton_medium is None:
                _singleton_medium = OnnxMelSpeakerVerifyService(
                    settings_field="titanet_pipeline_onnx_path",
                    default_filename="titanet_small_medium_5epoch_lr5e5.onnx",
                    log_tag="medium ONNX",
                )
                if _singleton_finetuned is not None and _singleton_medium is _singleton_finetuned:
                    raise RuntimeError(
                        "ONNX 싱글톤 버그: medium 이 finetuned 와 동일 인스턴스입니다."
                    )
    return _singleton_medium


def get_finetuned_onnx_service() -> OnnxMelSpeakerVerifyService:
    """파인튜닝 ONNX (`titanet_small_finetuned_final.onnx` 기본)."""
    global _singleton_finetuned, _singleton_medium
    if _singleton_finetuned is None:
        with _singleton_lock:
            if _singleton_finetuned is None:
                _singleton_finetuned = OnnxMelSpeakerVerifyService(
                    settings_field="titanet_finetuned_onnx_path",
                    default_filename="titanet_small_finetuned_final.onnx",
                    log_tag="finetuned ONNX",
                    infer_max_pcm_sec=settings.titanet_finetuned_infer_max_sec,
                )
                if _singleton_medium is not None and _singleton_finetuned is _singleton_medium:
                    raise RuntimeError(
                        "ONNX 싱글톤 버그: finetuned 가 medium 과 동일 인스턴스입니다."
                    )
    return _singleton_finetuned


async def get_finetuned_onnx_service_async() -> OnnxMelSpeakerVerifyService:
    """첫 로드 시 이벤트 루프 블로킹 방지."""
    if _singleton_finetuned is not None:
        return _singleton_finetuned
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_finetuned_onnx_service)


def clear_all_onnx_voiceprints() -> None:
    """medium·finetuned 싱글톤의 in-memory voiceprint 전부 삭제 (기동 옵션·관리 API)."""
    get_onnx_pipeline_service().clear_all_voiceprints()
    get_finetuned_onnx_service().clear_all_voiceprints()


def _session_id_log(sess: object | None) -> str:
    """None 이면 (none) — id(None) 과 구분 (기동 로그 혼동 방지)."""
    if sess is None:
        return "(none)"
    return str(id(sess))


def log_onnx_inference_session_check() -> None:
    """medium·finetuned 가 서로 다른 InferenceSession·파일을 쓰는지 기동 시 한 번 로그."""
    md = get_onnx_pipeline_service()
    ft = get_finetuned_onnx_service()
    sid_m = md.ort_inference_session_id()
    sid_f = ft.ort_inference_session_id()
    path_m = md.onnx_resolved_path or "(미로드)"
    path_f = ft.onnx_resolved_path or "(미로드)"

    if md is ft:
        logger.error(
            "ONNX 싱글톤 치명 오류: medium 과 finetuned 가 동일 파이썬 객체(id=%s)입니다.",
            id(md),
        )
    logger.info("finetuned onnx_path: %s", ft.onnx_path or path_f)
    logger.info("medium    onnx_path: %s", md.onnx_path or path_m)
    logger.info("finetuned session id: %s", _session_id_log(ft.session))
    logger.info("medium    session id: %s", _session_id_log(md.session))
    logger.info(
        "ONNX verifier 인스턴스 id: medium=%d finetuned=%d (같으면 동일 객체)",
        id(md),
        id(ft),
    )
    logger.info("medium InferenceSession id:    %s", sid_m)
    logger.info("finetuned InferenceSession id: %s", sid_f)
    logger.info(
        "startup ONNX Runtime: medium session_id=%s path=%s | finetuned session_id=%s path=%s",
        sid_m,
        path_m,
        sid_f,
        path_f,
    )
    if md is not ft and md.session is not None and ft.session is not None and md.session is ft.session:
        logger.error(
            "medium·finetuned 가 서로 다른 서비스 객체인데 동일 InferenceSession 을 공유합니다."
        )
    if sid_m is not None and sid_f is not None and sid_m == sid_f:
        logger.error(
            "medium·finetuned 가 동일 InferenceSession(id)를 공유합니다. "
            "싱글톤/초기화 버그를 확인하세요."
        )
    if path_m == path_f and path_m not in ("", "(미로드)"):
        logger.warning(
            "medium·finetuned ONNX 파일 경로가 동일합니다 (%s). "
            "유사도가 항상 같게 나올 수 있습니다.",
            path_m,
        )
    logger.info(
        "voiceprint 저장 dict id: medium=%d finetuned=%d (다르면 별도 저장소)",
        md.voiceprint_store_dict_id(),
        ft.voiceprint_store_dict_id(),
    )


# 하위 호환(이전 클래스명)
OnnxPipelineSpeakerVerifyService = OnnxMelSpeakerVerifyService
# 문서·이슈에서 쓰는 이름 (동일 클래스 — 세션은 인스턴스당 self._ort_sess)
OnnxSpeakerVerifier = OnnxMelSpeakerVerifyService

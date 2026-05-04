"""순정(baseline) ONNX vs 파인튜닝 ONNX 화자검증 동시 비교·CSV 기록.

동일 PCM → 동일 mel(`torchaudio` 프론트)을 두 `onnxruntime.InferenceSession`에 넣어
각각 등록 voiceprint 와 코사인 유사도를 계산한다. baseline·finetuned 임베딩 간 코사인은
진단용으로 INFO 로그에 남긴다(스냅샷 필드는 기존 gate·CSV 스키마 유지).
"""

from __future__ import annotations

import asyncio
import csv
import functools
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.services.speaker_verify.onnx_pipeline import (
    _MIN_MEL_FRAMES,
    _effective_finetuned_mel_frame_cap,
    _onnx_embedding_from_output,
)
from app.services.speaker_verify.titanet_mel import pcm16_bytes_to_float_mono
from app.services.speaker_verify.titanet_mel_nemo import build_onnx_mel_frontend
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_csv_lock = threading.Lock()
_service: TitaNetCompareSpeakerVerifyService | None = None


def _app_root() -> Path:
    """`app/` 디렉터리 (서비스 패키지 루트)."""
    return Path(__file__).resolve().parent.parent.parent


def _repo_root() -> Path:
    """레포지토리 루트 (`app/` 의 상위). `models/speech_verification/` 등이 여기에 둔다."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _resolve_compare_csv_path() -> Path:
    raw = (settings.speaker_verify_compare_log_path or "").strip()
    p = Path(raw)
    if p.is_absolute():
        return p
    return _app_root() / p


def _resolve_baseline_onnx_path() -> Path:
    raw = (settings.speaker_verify_compare_baseline_onnx_path or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
        for base in (_repo_root(), _app_root()):
            p2 = base / raw
            if p2.is_file():
                return p2
        raise FileNotFoundError(
            f"SPEAKER_VERIFY_COMPARE_BASELINE_ONNX_PATH 에 해당하는 파일이 없습니다: {raw!r}"
        )
    default = _repo_root() / "models" / "speech_verification" / "titanet-s.onnx"
    if not default.is_file():
        raise FileNotFoundError(
            f"baseline ONNX 기본 경로에 파일이 없습니다: {default} "
            "(레포 `models/speech_verification/titanet-s.onnx` 또는 SPEAKER_VERIFY_COMPARE_BASELINE_ONNX_PATH)"
        )
    return default


def _l2_normalize_1d(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return np.asarray(x, dtype=np.float32)
    return np.asarray(x / n, dtype=np.float32)


def _threshold_for_gate_model(gate: str) -> float:
    g = (gate or "finetuned").strip().lower()
    if g == "baseline":
        return float(settings.speaker_verify_threshold)
    v = settings.speaker_verify_finetuned_threshold
    return settings.speaker_verify_threshold if v is None else float(v)


def _similarity_passes_threshold(sim: float, thr: float) -> bool:
    if not isinstance(sim, (int, float)) or not math.isfinite(float(sim)):
        return False
    if float(sim) < 0.0:
        return False
    return float(sim) >= float(thr)


_COMPARE_CSV_FIELDNAMES: tuple[str, ...] = (
    "timestamp",
    "call_id",
    "turn_index",
    "audio_bytes",
    "audio_sec",
    "baseline_has_voiceprint",
    "finetuned_has_voiceprint",
    "bypass",
    "baseline_similarity",
    "finetuned_similarity",
    "threshold",
    "gate_model",
    "baseline_ok",
    "finetuned_ok",
    "verified",
    "stt_executed",
    "transcript",
    "stt_finetuned_executed",
    "stt_baseline_executed",
    "transcript_finetuned",
    "transcript_baseline",
    "transcript_no_verify",
)


def _cosine_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = _l2_normalize_1d(a).astype(np.float64, copy=False)
    bb = _l2_normalize_1d(b).astype(np.float64, copy=False)
    return float(np.dot(aa, bb))


@dataclass(frozen=True)
class CompareSnapshot:
    """검증 직후 스냅샷 — STT 후 transcript / stt_executed 만 채워 CSV 기록.

    - ``verified``: ``gate_model``이 가리키는 모델 유사도만 ``threshold``와 비교해 STT 진행 여부를 결정.
    - ``baseline_ok`` / ``finetuned_ok``: 각 모델이 **자기 임계값**을 넘었는지(로그·CSV·개선도 분석용).
      순정이 낮아도 파인튜닝 게이트가 통과하면 ``verified``는 True일 수 있음.
    """

    call_id: str
    turn_index: int
    audio_bytes: int
    audio_sec: float
    baseline_has_voiceprint: bool
    finetuned_has_voiceprint: bool
    bypass: bool
    baseline_similarity: float
    finetuned_similarity: float
    threshold: float
    gate_model: str
    baseline_ok: bool
    finetuned_ok: bool
    verified: bool
    inter_emb_cosine: float | None = None
    bypass_reason: str | None = None

    @property
    def is_baseline_passed(self) -> bool:
        """순정(baseline) 모델이 전용 임계값을 넘었는지 — STT 게이트와 별개(비교·지표용)."""
        return self.baseline_ok


def format_compare_snapshot_log_block(snap: CompareSnapshot) -> str:
    """runner 한 줄(멀티라인) 로그용 — 사용자 발화 텍스트는 caller 가 이어붙임(% 이스케이프 불필요)."""
    parts: list[str] = []
    if snap.inter_emb_cosine is not None:
        parts.append(
            "titanet_compare: 동일 mel 기준 baseline_emb vs finetuned_emb 코사인(L2)="
            f"{snap.inter_emb_cosine:.4f}"
        )
    if snap.bypass and snap.bypass_reason:
        parts.extend(
            [
                "[VERIFY_COMPARE]",
                f"call_id={snap.call_id}",
                f"baseline={snap.baseline_similarity:.4f}  ok={snap.baseline_ok}",
                f"finetuned={snap.finetuned_similarity:.4f} ok={snap.finetuned_ok}",
                f"gate={snap.gate_model}",
                "verified=True",
                f"reason={snap.bypass_reason}",
            ]
        )
    elif not snap.bypass:
        parts.extend(
            [
                "[VERIFY_COMPARE]",
                f"call_id={snap.call_id}",
                f"baseline={snap.baseline_similarity:.4f}  ok={snap.baseline_ok}",
                f"finetuned={snap.finetuned_similarity:.4f} ok={snap.finetuned_ok}",
                f"gate={snap.gate_model}",
                f"threshold={snap.threshold:.4f}",
                f"verified={snap.verified}",
            ]
        )
    return "\n".join(parts)


class TitaNetCompareSpeakerVerifyService:
    """baseline ONNX(순정) + 파인튜닝 ONNX를 동일 mel로 점수 산출.

    STT 진행(``verified``)은 ``settings.speaker_verify_gate_model``이 가리키는 쪽만
    ``threshold``와 비교한다. 반대쪽 점수는 ``baseline_ok`` / ``finetuned_ok``로
    로그·CSV에 남겨 순정 대비 개선 여부를 본다.
    """

    def __init__(self) -> None:
        self._baseline_ort_sess: Any = None
        self._baseline_onnx_in_audio: str = ""
        self._baseline_onnx_in_length: str | None = None
        self._baseline_emb_out_idx: int = 0
        self._baseline_onnx_resolved: str = ""
        self._baseline_load_error: str | None = None
        self._baseline_voiceprints: dict[str, np.ndarray] = {}
        self._emb_baseline: dict[str, list[np.ndarray]] = {}
        self._infer_lock = threading.Lock()
        self._baseline_init_lock = threading.Lock()
        self._mel: Any = None
        self._mel_init_lock = threading.Lock()

    @property
    def load_error(self) -> str | None:
        """baseline ONNX 로드 실패 시 메시지(파인튜닝 ONNX는 별도)."""
        return self._baseline_load_error

    def _ensure_mel_frontend(self) -> None:
        if self._mel is not None:
            return
        with self._mel_init_lock:
            if self._mel is not None:
                return
            t0 = time.monotonic()
            self._mel = build_onnx_mel_frontend(
                torch.device("cpu"), log_tag="verify.compare"
            )
            logger.info(
                "titanet_compare: mel 프론트엔드 준비 elapsed=%.2fs",
                time.monotonic() - t0,
            )

    def _mel_pad_if_needed(self, mel: np.ndarray) -> np.ndarray:
        n_mels = self._mel.params.n_mels if self._mel is not None else 80
        if mel.ndim != 3 or mel.shape[1] != n_mels:
            return mel
        tdim = mel.shape[2]
        if tdim >= _MIN_MEL_FRAMES:
            return mel
        pad = _MIN_MEL_FRAMES - tdim
        return np.pad(mel, ((0, 0), (0, 0), (0, pad)), mode="constant")

    def _cap_mel_time(self, mel: np.ndarray) -> np.ndarray:
        mx = _effective_finetuned_mel_frame_cap()
        tdim = int(mel.shape[2])
        if tdim <= mx:
            return mel
        return np.ascontiguousarray(mel[:, :, :mx], dtype=np.float32)

    def _pcm_to_mel_np(self, pcm16: bytes) -> np.ndarray:
        self._ensure_mel_frontend()
        if self._mel is None:
            raise RuntimeError("mel 미초기화")
        pcm = pcm16
        mx_sec = float(settings.titanet_finetuned_infer_max_sec)
        if mx_sec > 0:
            max_bytes = int(16000 * 2 * mx_sec)
            if len(pcm) > max_bytes:
                pcm = pcm[:max_bytes]
        samples = pcm16_bytes_to_float_mono(pcm)
        mel = self._mel(samples)
        mel = self._mel_pad_if_needed(mel)
        mel = self._cap_mel_time(mel)
        return np.ascontiguousarray(mel, dtype=np.float32)

    def _embedding_from_mel_baseline(self, mel: np.ndarray) -> np.ndarray:
        if self._baseline_ort_sess is None:
            raise RuntimeError(self._baseline_load_error or "baseline ONNX 미초기화")
        mel = self._cap_mel_time(np.ascontiguousarray(mel, dtype=np.float32))
        feeds: dict[str, np.ndarray] = {self._baseline_onnx_in_audio: mel}
        if self._baseline_onnx_in_length is not None:
            feeds[self._baseline_onnx_in_length] = np.array(
                [int(mel.shape[2])], dtype=np.int64
            )
        outs = self._baseline_ort_sess.run(None, feeds)
        if not outs or len(outs) <= self._baseline_emb_out_idx:
            raise RuntimeError("baseline ONNX 출력 없음")
        return _onnx_embedding_from_output(outs[self._baseline_emb_out_idx])

    def load_baseline_model(self) -> None:
        """baseline ONNX 세션 로드. `run_in_executor` 등 동기 스레드에서 호출."""
        if not settings.speaker_verify_compare_enabled:
            return
        self._ensure_baseline_loaded()

    def _ensure_baseline_loaded(self) -> None:
        if not settings.speaker_verify_compare_enabled:
            return
        if self._baseline_ort_sess is not None or self._baseline_load_error is not None:
            return
        with self._baseline_init_lock:
            if self._baseline_ort_sess is not None or self._baseline_load_error is not None:
                return
            try:
                path = _resolve_baseline_onnx_path()
                self._baseline_onnx_resolved = str(path.resolve())
                import onnxruntime as ort

                t0 = time.monotonic()
                logger.info(
                    "titanet_compare: baseline ONNX 로딩 path=%s …",
                    self._baseline_onnx_resolved,
                )
                providers: list[str] = []
                if torch.cuda.is_available():
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")
                sess_opt = ort.SessionOptions()
                self._baseline_ort_sess = ort.InferenceSession(
                    str(path), sess_options=sess_opt, providers=providers
                )
                ins = self._baseline_ort_sess.get_inputs()
                if not ins:
                    raise RuntimeError("baseline ONNX 입력 없음")
                self._baseline_onnx_in_audio = ins[0].name
                self._baseline_onnx_in_length = ins[1].name if len(ins) > 1 else None
                outs_meta = self._baseline_ort_sess.get_outputs()
                self._baseline_emb_out_idx = 1 if len(outs_meta) > 1 else 0
                self._ensure_mel_frontend()
                logger.info(
                    "titanet_compare: baseline ONNX 준비 완료 elapsed=%.2fs emb_out_idx=%s",
                    time.monotonic() - t0,
                    self._baseline_emb_out_idx,
                )
            except FileNotFoundError as e:
                self._baseline_load_error = str(e)
                logger.error("titanet_compare: baseline ONNX 경로 오류 — %s", e)
            except Exception as e:
                self._baseline_load_error = str(e)
                logger.exception("titanet_compare: baseline ONNX 로드 실패: %s", e)

    @property
    def models_ready(self) -> bool:
        if not settings.speaker_verify_compare_enabled:
            return False
        return self._baseline_ort_sess is not None and self._baseline_load_error is None

    def compare_gate_ready(self, call_id: str, onnx_finetuned: Any) -> bool:
        if not self.models_ready:
            return False
        with self._infer_lock:
            bl = call_id in self._baseline_voiceprints
        try:
            ft = bool(onnx_finetuned.has_voiceprint(call_id))
        except Exception:
            ft = False
        return bl and ft

    def _enroll_n(self) -> int:
        return max(1, int(settings.enroll_utt_count))

    def accumulate_enrollment_utterance_sync(self, call_id: str, pcm16: bytes) -> None:
        if not settings.speaker_verify_compare_enabled:
            return
        self._ensure_baseline_loaded()
        if self._baseline_ort_sess is None:
            return
        with self._infer_lock:
            try:
                mel = self._pcm_to_mel_np(pcm16)
                eb = self._embedding_from_mel_baseline(mel)
            except Exception as e:
                logger.error(
                    "titanet_compare: baseline enrollment 임베딩 실패 call_id=%s: %s",
                    call_id,
                    e,
                )
                return
            lb = self._emb_baseline.setdefault(call_id, [])
            lb.append(np.asarray(eb, dtype=np.float32).reshape(-1).copy())
            n_lb = len(lb)
        logger.info(
            "titanet_compare: [enrollment] baseline 발화 수집 %d/%d call_id=%s",
            n_lb,
            self._enroll_n(),
            call_id,
        )

    def finalize_enrollment_sync(self, call_id: str) -> None:
        if not settings.speaker_verify_compare_enabled:
            return
        self._ensure_baseline_loaded()
        if self._baseline_ort_sess is None:
            return
        n_need = self._enroll_n()
        with self._infer_lock:
            lb = self._emb_baseline.get(call_id, [])
            if len(lb) < n_need:
                logger.warning(
                    "titanet_compare: baseline finalize 스킵 call_id=%s (have=%d need=%d)",
                    call_id,
                    len(lb),
                    n_need,
                )
                return
            try:
                stack_b = np.stack(lb[:n_need], axis=0)
                vp_b = _l2_normalize_1d(np.mean(stack_b, axis=0))
                self._baseline_voiceprints[call_id] = vp_b
                self._emb_baseline.pop(call_id, None)
            except Exception as e:
                logger.error(
                    "titanet_compare: baseline finalize 실패 call_id=%s: %s", call_id, e
                )
                return
        logger.info(
            "titanet_compare: baseline ONNX enrollment 완료 call_id=%s (%d발화 평균·L2)",
            call_id,
            n_need,
        )

    def _extract_baseline_embedding(self, pcm16: bytes) -> np.ndarray:
        self._ensure_baseline_loaded()
        if self._baseline_ort_sess is None:
            raise RuntimeError("baseline ONNX 미로드")
        mel = self._pcm_to_mel_np(pcm16)
        emb = self._embedding_from_mel_baseline(mel)
        return _l2_normalize_1d(emb).astype(np.float32, copy=True)

    def _enroll_baseline_sync(self, call_id: str, pcm16: bytes) -> None:
        for _ in range(self._enroll_n()):
            self.accumulate_enrollment_utterance_sync(call_id, pcm16)
        self.finalize_enrollment_sync(call_id)

    async def enroll_baseline_async(self, call_id: str, pcm16: bytes) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            functools.partial(self._enroll_baseline_sync, call_id, pcm16),
        )

    def _baseline_similarity_sync(self, mel: np.ndarray, call_id: str) -> float:
        with self._infer_lock:
            vb = self._baseline_voiceprints.get(call_id)
            if vb is None:
                raise RuntimeError("baseline voiceprint 없음")
            vb = np.asarray(vb, dtype=np.float32).reshape(-1).copy()
        emb_b = self._embedding_from_mel_baseline(mel)
        return _cosine_l2(emb_b, vb)

    async def verify_compare_async(
        self,
        pcm16: bytes,
        call_id: str,
        turn_index: int,
        onnx_finetuned: Any,
    ) -> CompareSnapshot:
        gate = (settings.speaker_verify_gate_model or "finetuned").strip().lower()
        if gate not in ("baseline", "finetuned"):
            gate = "finetuned"
        thr = _threshold_for_gate_model(gate)
        thr_bl = _threshold_for_gate_model("baseline")
        thr_ft = _threshold_for_gate_model("finetuned")
        audio_bytes = len(pcm16)
        audio_sec = audio_bytes / 32000.0 if audio_bytes else 0.0

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_baseline_loaded)
        bl_has = call_id in self._baseline_voiceprints if self._baseline_ort_sess is not None else False
        try:
            ft_has = bool(onnx_finetuned.has_voiceprint(call_id))
        except Exception:
            ft_has = False

        if not settings.speaker_verify_compare_enabled or self._baseline_ort_sess is None:
            return CompareSnapshot(
                call_id=call_id,
                turn_index=turn_index,
                audio_bytes=audio_bytes,
                audio_sec=audio_sec,
                baseline_has_voiceprint=bl_has,
                finetuned_has_voiceprint=ft_has,
                bypass=True,
                baseline_similarity=-1.0,
                finetuned_similarity=-1.0,
                threshold=thr,
                gate_model=gate,
                baseline_ok=False,
                finetuned_ok=False,
                verified=False,
            )

        if not (bl_has and ft_has):
            return CompareSnapshot(
                call_id=call_id,
                turn_index=turn_index,
                audio_bytes=audio_bytes,
                audio_sec=audio_sec,
                baseline_has_voiceprint=bl_has,
                finetuned_has_voiceprint=ft_has,
                bypass=True,
                baseline_similarity=-1.0,
                finetuned_similarity=-1.0,
                threshold=thr,
                gate_model=gate,
                baseline_ok=False,
                finetuned_ok=False,
                verified=True,
                bypass_reason="no_voiceprint",
            )

        def _verify_dual_sync() -> tuple[float, float, float | None]:
            mel = self._pcm_to_mel_np(pcm16)
            emb_bl = self._embedding_from_mel_baseline(mel)
            with self._infer_lock:
                vp_bl = self._baseline_voiceprints.get(call_id)
                if vp_bl is None:
                    raise RuntimeError("baseline voiceprint 없음")
                vp_bl = np.asarray(vp_bl, dtype=np.float32).reshape(-1).copy()
            bl_sim = _cosine_l2(emb_bl, vp_bl)
            _ok_ft, ft_sim = onnx_finetuned.verify_with_precomputed_mel(mel, call_id)
            _ = _ok_ft
            inter: float | None = None
            try:
                emb_ft_np = onnx_finetuned._embedding_from_mel_np(mel)
                if emb_bl.size == emb_ft_np.size:
                    inter = float(_cosine_l2(emb_bl, emb_ft_np))
            except Exception:
                pass
            return bl_sim, float(ft_sim), inter

        try:
            bl_sim, ft_sim, inter_cos = await loop.run_in_executor(None, _verify_dual_sync)
        except Exception as e:
            logger.error("titanet_compare: dual verify 실패 call_id=%s: %s", call_id, e)
            return CompareSnapshot(
                call_id=call_id,
                turn_index=turn_index,
                audio_bytes=audio_bytes,
                audio_sec=audio_sec,
                baseline_has_voiceprint=True,
                finetuned_has_voiceprint=True,
                bypass=True,
                baseline_similarity=-1.0,
                finetuned_similarity=-1.0,
                threshold=thr,
                gate_model=gate,
                baseline_ok=False,
                finetuned_ok=False,
                verified=False,
            )

        gate_sim = bl_sim if gate == "baseline" else float(ft_sim)
        baseline_ok = _similarity_passes_threshold(bl_sim, thr_bl)
        finetuned_ok = _similarity_passes_threshold(float(ft_sim), thr_ft)
        verified = bool(gate_sim >= thr)

        return CompareSnapshot(
            call_id=call_id,
            turn_index=turn_index,
            audio_bytes=audio_bytes,
            audio_sec=audio_sec,
            baseline_has_voiceprint=True,
            finetuned_has_voiceprint=True,
            bypass=False,
            baseline_similarity=bl_sim,
            finetuned_similarity=float(ft_sim),
            threshold=thr,
            gate_model=gate,
            baseline_ok=baseline_ok,
            finetuned_ok=finetuned_ok,
            verified=verified,
            inter_emb_cosine=inter_cos,
        )

    def flush_csv_row_sync(
        self,
        snap: CompareSnapshot,
        *,
        transcript_finetuned: str,
        transcript_baseline: str,
        transcript_no_verify: str,
        stt_finetuned_executed: bool,
        stt_baseline_executed: bool,
    ) -> None:
        if not settings.speaker_verify_compare_enabled:
            return
        path = _resolve_compare_csv_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tf = transcript_finetuned.replace("\r\n", " ").replace("\n", " ") if transcript_finetuned else ""
        tb = transcript_baseline.replace("\r\n", " ").replace("\n", " ") if transcript_baseline else ""
        tn = transcript_no_verify.replace("\r\n", " ").replace("\n", " ") if transcript_no_verify else ""
        stt_any = bool(stt_finetuned_executed or stt_baseline_executed)
        legacy_t = tf if tf else (tb if tb else tn)
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "call_id": snap.call_id,
            "turn_index": snap.turn_index,
            "audio_bytes": snap.audio_bytes,
            "audio_sec": f"{snap.audio_sec:.6f}",
            "baseline_has_voiceprint": snap.baseline_has_voiceprint,
            "finetuned_has_voiceprint": snap.finetuned_has_voiceprint,
            "bypass": snap.bypass,
            "baseline_similarity": f"{snap.baseline_similarity:.6f}",
            "finetuned_similarity": f"{snap.finetuned_similarity:.6f}",
            "threshold": f"{snap.threshold:.6f}",
            "gate_model": snap.gate_model,
            "baseline_ok": snap.baseline_ok,
            "finetuned_ok": snap.finetuned_ok,
            "verified": snap.verified,
            "stt_executed": stt_any,
            "transcript": legacy_t,
            "stt_finetuned_executed": stt_finetuned_executed,
            "stt_baseline_executed": stt_baseline_executed,
            "transcript_finetuned": tf,
            "transcript_baseline": tb,
            "transcript_no_verify": tn,
        }
        with _csv_lock:
            new_file = not path.is_file()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(_COMPARE_CSV_FIELDNAMES))
                if new_file:
                    w.writeheader()
                w.writerow(row)

    def cleanup(self, call_id: str) -> None:
        with self._infer_lock:
            self._baseline_voiceprints.pop(call_id, None)
            self._emb_baseline.pop(call_id, None)

    def reset_enrollment_buffers_only(self) -> None:
        with self._infer_lock:
            self._emb_baseline.clear()

    def clear_all_voiceprints(self) -> int:
        with self._infer_lock:
            n = len(self._baseline_voiceprints)
            self._baseline_voiceprints.clear()
            self._emb_baseline.clear()
        logger.info("titanet_compare: baseline ONNX voiceprint·enrollment 버퍼 전체 삭제")
        return n


def get_titanet_compare_speaker_verify_service() -> TitaNetCompareSpeakerVerifyService:
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = TitaNetCompareSpeakerVerifyService()
    return _service


async def get_titanet_compare_speaker_verify_service_async() -> TitaNetCompareSpeakerVerifyService:
    if _service is not None:
        return _service
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_titanet_compare_speaker_verify_service)


def clear_all_compare_voiceprints() -> None:
    global _service
    if _service is not None:
        _service.clear_all_voiceprints()


def cleanup_compare_for_call(call_id: str) -> None:
    global _service
    if _service is not None:
        _service.cleanup(call_id)

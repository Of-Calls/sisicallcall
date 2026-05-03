"""파인튜닝 TitaNet-Small ONNX 화자 검증 — mel 입력 [B,80,T], 출력 embeddings outputs[1] [B,192].

raw wav 는 ONNX에 넣지 않음. NeMo `get_embedding` 과 동일하게 mel 전처리 후 추론한다.

CLI 스모크 테스트:
    python -m app.services.speaker_verify.titanet_onnx_verifier ^
      --model-path app/models/speaker_verification/titanet_small_medium_5epoch_lr5e5.onnx ^
      --enroll-wav path/to/enroll.wav ^
      --verify-wav path/to/verify.wav ^
      --threshold 0.48
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)

_TARGET_SR = 16000
_EMB_DIM = 192
_EPS = 1e-12


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터 간 cosine similarity (미정규화 입력도 허용)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + _EPS)
    return float(np.dot(a, b) / denom)


def load_wav_mono_float32(path: str | Path) -> tuple[np.ndarray, int]:
    """WAV 파일 로드 → mono float32, 원본 샘플레이트."""
    import torchaudio

    path = Path(path)
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    return wav.numpy().astype(np.float32), int(sr)


def resample_to_16k_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """1D float32 오디오를 16kHz mono 로 맞춤."""
    import torchaudio

    if audio.ndim != 1:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate == _TARGET_SR:
        return audio.astype(np.float32)
    t = torch.from_numpy(audio).unsqueeze(0)
    out = torchaudio.functional.resample(t, sample_rate, _TARGET_SR)
    return out.squeeze(0).numpy().astype(np.float32)


class TitaNetOnnxVerifier:
    """ONNXRuntime 기반 TitaNet 화자 검증 — outputs[1] 임베딩만 사용 (outputs[0] logits 무시)."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float | None = None,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX 파일 없음: {self.model_path}")

        prov = (
            list(providers)
            if providers is not None
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(self.model_path), providers=prov)

        self._in_audio_name = self._session.get_inputs()[0].name
        self._in_length_name = self._session.get_inputs()[1].name
        outs = self._session.get_outputs()
        if len(outs) < 2:
            raise ValueError(
                f"ONNX 출력 개수 부족: {len(outs)} (logits+embedding 2개 필요)"
            )
        self._out_logits_idx = 0
        self._out_emb_idx = 1

        from app.services.speaker_verify.titanet_mel_nemo import build_onnx_mel_frontend

        self._mel_frontend = build_onnx_mel_frontend(torch.device("cpu"), log_tag="titanet_onnx_verifier")

        logger.info(
            "TitaNet ONNX 로드 OK path=%s providers=%s",
            self.model_path,
            self._session.get_providers(),
        )
        logger.info("inputs: %s", [i.name for i in self._session.get_inputs()])
        logger.info(
            "outputs[0] logits (분류, 미사용): %s",
            outs[self._out_logits_idx].name if outs else "?",
        )
        logger.info(
            "outputs[1] embedding (사용): %s shape_hint=%s",
            outs[self._out_emb_idx].name if len(outs) > 1 else "?",
            outs[self._out_emb_idx].shape if len(outs) > 1 else None,
        )

    def preprocess_audio_to_mel(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """16kHz mono float32 → log-mel [1, n_mels, T] 및 length [1] (프레임 수 T).

        Mel 파라미터는 `TitaNetMelFrontend`·`app.utils.config`(`TITANET_ONNX_MEL_*`)와 동일.
        """
        samples = resample_to_16k_mono(audio, sample_rate)
        if samples.size == 0:
            raise ValueError("empty audio")

        mel = self._mel_frontend(samples)
        n_m = self._mel_frontend.params.n_mels
        if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != n_m:
            raise ValueError(f"unexpected mel shape {mel.shape}, expected [1,{n_m},T]")

        t_frames = int(mel.shape[2])
        length = np.array([t_frames], dtype=np.int64)

        logger.debug(
            "mel shape=%s min=%.4f max=%.4f length=%s",
            mel.shape,
            float(mel.min()),
            float(mel.max()),
            length,
        )
        return mel.astype(np.float32), length

    def extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """mel 생성 → ONNX 추론 → outputs[1] → L2 정규화 → shape [192]."""
        mel, length_arr = self.preprocess_audio_to_mel(audio, sample_rate)

        feeds = {
            self._in_audio_name: np.ascontiguousarray(mel, dtype=np.float32),
            self._in_length_name: np.ascontiguousarray(length_arr, dtype=np.int64),
        }
        outputs = self._session.run(None, feeds)
        logits = outputs[self._out_logits_idx]
        embedding = outputs[self._out_emb_idx]

        _ = logits

        emb = np.asarray(embedding, dtype=np.float32).reshape(embedding.shape[0], -1)
        if emb.shape[-1] != _EMB_DIM:
            logger.warning(
                "embedding 마지막 차원 기대=%s 실제=%s",
                _EMB_DIM,
                emb.shape[-1],
            )
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + _EPS)
        return emb[0].astype(np.float32)

    def enroll(
        self,
        audio_list: Sequence[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        """여러 발화에서 임베딩 추출 → 평균 → L2 정규화 → [192]."""
        if not audio_list:
            raise ValueError("audio_list 비어 있음")
        vecs = [
            self.extract_embedding(np.asarray(a, dtype=np.float32), sample_rate)
            for a in audio_list
        ]
        stacked = np.stack(vecs, axis=0)
        mean = stacked.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + _EPS)
        return mean.astype(np.float32)

    def verify(
        self,
        enrollment_embedding: np.ndarray,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, bool | None]:
        """등록 임베딩과 현재 발화 임베딩 코사인 유사도. threshold 없으면 passed=None."""
        cur = self.extract_embedding(audio, sample_rate)
        enr = np.asarray(enrollment_embedding, dtype=np.float32).reshape(-1)
        score = float(np.dot(enr, cur))
        if self.threshold is None:
            return score, None
        return score, score >= self.threshold

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """인스턴스 메서드 — `compute_cosine_similarity` 위임."""
        return compute_cosine_similarity(a, b)


def _cli_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="TitaNet ONNX 화자검증 스모크 테스트")
    ap.add_argument(
        "--model-path",
        type=Path,
        default=Path("app/models/speaker_verification/titanet_small_medium_5epoch_lr5e5.onnx"),
        help="ONNX 파일 경로",
    )
    ap.add_argument("--enroll-wav", type=Path, required=True, help="등록용 WAV")
    ap.add_argument("--verify-wav", type=Path, required=True, help="검증용 WAV")
    ap.add_argument("--threshold", type=float, default=None, help="코사인 임계값 (선택)")
    args = ap.parse_args()

    root = Path.cwd()
    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve()

    print("model loaded")
    ver = TitaNetOnnxVerifier(model_path, threshold=args.threshold)
    print("inputs:")
    for i in ver._session.get_inputs():
        print(f"- {i.name}")
    print("outputs:")
    outs = ver._session.get_outputs()
    print(f"- {outs[0].name}: logits (미사용)")
    print(f"- {outs[1].name}: embedding")

    ae, sr_e = load_wav_mono_float32(args.enroll_wav)
    av, sr_v = load_wav_mono_float32(args.verify_wav)

    mel_e, len_e = ver.preprocess_audio_to_mel(ae, sr_e)
    mel_v, len_v = ver.preprocess_audio_to_mel(av, sr_v)
    print(f"enroll mel shape: {mel_e.shape} length: {len_e}")
    print(f"verify mel shape: {mel_v.shape} length: {len_v}")

    enroll_emb = ver.enroll([ae], sr_e)
    verify_emb = ver.extract_embedding(av, sr_v)

    print(f"enroll_embedding_shape: {enroll_emb.shape}")
    print(f"verify_embedding_shape: {verify_emb.shape}")

    score, passed = ver.verify(enroll_emb, av, sr_v)
    print(f"score: {score:.4f}")
    if passed is None:
        print("passed: (threshold 미설정)")
    else:
        print(f"passed: {passed}")


if __name__ == "__main__":
    _cli_main()

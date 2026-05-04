#!/usr/bin/env python3
"""TitaNetCompareSpeakerVerifyService 로컬 검증 (Twilio WebSocket 없음).

실행 (레포 루트, venv):

  set SPEAKER_VERIFY_COMPARE_ENABLED=true   # PowerShell: $env:SPEAKER_VERIFY_COMPARE_ENABLED=\"true\"
  venv\\Scripts\\python.exe scripts\\test_titanet_compare_local.py

테스트 1은 자식 프로세스에서 compare_enabled=false 로만 돌려 NeMo 미import 를 확인합니다.
테스트 2~6은 위 환경변수 true(또는 .env) + ONNX 모델 파일이 있어야 합니다.
"""
from __future__ import annotations

import asyncio
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _snap_dict(s: object) -> dict:
    return {
        "bypass": s.bypass,
        "baseline_similarity": s.baseline_similarity,
        "finetuned_similarity": s.finetuned_similarity,
        "baseline_ok": s.baseline_ok,
        "finetuned_ok": s.finetuned_ok,
    }


def _l2_mean_stack(embs: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.asarray(e, dtype=np.float32).ravel() for e in embs], axis=0)
    m = np.mean(stack, axis=0)
    n = float(np.linalg.norm(m))
    if n < 1e-12:
        return m.astype(np.float32)
    return (m / n).astype(np.float32)


async def _onnx_enroll_same_pcm(call_id: str, pcm: bytes, n_need: int, onnx_ft) -> None:
    embs: list[np.ndarray] = []
    for _ in range(n_need):
        e = await onnx_ft.extract_embedding_for_enroll(pcm)
        embs.append(np.asarray(e, dtype=np.float32).ravel().copy())
    vp = _l2_mean_stack(embs)
    onnx_ft.store_voiceprint_vector(call_id, vp)


def run_test_1_subprocess() -> tuple[str, str]:
    """자식 프로세스: SPEAKER_VERIFY_COMPARE_ENABLED=false, NeMo 미로드·baseline None."""
    code = f"""
import os, sys
os.environ["SPEAKER_VERIFY_COMPARE_ENABLED"] = "false"
sys.path.insert(0, {repr(str(_ROOT))})
from app.services.speaker_verify.titanet_compare import get_titanet_compare_speaker_verify_service
svc = get_titanet_compare_speaker_verify_service()
nemo_like = any(k == "nemo" or k.startswith("nemo.") for k in sys.modules)
print("NEMO_LIKE", nemo_like)
print("BASELINE_NONE", svc._baseline_model is None)
"""
    env = os.environ.copy()
    env["SPEAKER_VERIFY_COMPARE_ENABLED"] = "false"
    p = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        return "FAIL", f"subprocess exit={p.returncode}\n{out}"
    nemo_like = "NEMO_LIKE True" in out
    baseline_none = "BASELINE_NONE True" in out
    if nemo_like or not baseline_none:
        return "FAIL", out
    return "PASS", out.strip()


async def async_main() -> None:
    # compare + CSV 경로 (load_dotenv 이전에 고정하면 .env 가 덮어쓰지 않음)
    os.environ["SPEAKER_VERIFY_COMPARE_ENABLED"] = "true"
    if not os.environ.get("SPEAKER_VERIFY_COMPARE_LOG_PATH"):
        os.environ["SPEAKER_VERIFY_COMPARE_LOG_PATH"] = str(
            _ROOT / "logs" / "titanet_compare_local_test.csv"
        )

    from dotenv import load_dotenv

    load_dotenv(override=False)

    from app.services.speaker_verify.compare_runtime_bootstrap import (
        bootstrap_speaker_compare_runtime_env_before_imports,
    )

    bootstrap_speaker_compare_runtime_env_before_imports(log=True)

    lines: list[str] = []

    # ----- 테스트 1 -----
    t1, detail1 = run_test_1_subprocess()
    lines.append(f"테스트 1 — NeMo lazy load:     {t1}")
    if t1 == "FAIL":
        lines.append(f"  상세: {detail1}")

    # ----- 이하 compare_enabled=true 가정 -----
    from app.services.speaker_verify.onnx_pipeline import get_finetuned_onnx_service
    from app.services.speaker_verify.titanet_compare import (
        get_titanet_compare_speaker_verify_service,
        _resolve_compare_csv_path,
    )
    from app.utils.config import settings

    dummy_pcm = np.zeros(16000, dtype=np.int16).tobytes()
    onnx_ft = get_finetuned_onnx_service()
    svc = get_titanet_compare_speaker_verify_service()
    # `models_ready` 는 로드를 트리거하지 않음 — 서버와 동일하게 명시 로드.
    svc.load_baseline_model()

    # ----- 테스트 2 -----
    t2 = "FAIL"
    t2_extra = ""
    try:
        if not svc.models_ready:
            t2_extra = f"models_ready=False load_error={svc.load_error!r}"
        else:
            emb = svc._extract_baseline_embedding(dummy_pcm)
            assert emb is not None
            assert emb.ndim == 1
            nrm = float(np.linalg.norm(emb))
            assert abs(nrm - 1.0) < 1e-5, f"L2 norm={nrm}"
            t2 = "PASS"
            t2_extra = f"embedding shape={emb.shape} norm={nrm:.6f}"
    except Exception as e:
        t2_extra = str(e)
    lines.append(f"테스트 2 — embedding 추출:     {t2}  {t2_extra}")

    # ----- 테스트 3 -----
    t3 = "FAIL"
    t3_extra = ""
    cid3 = "test-call-001-local"
    try:
        if not svc.models_ready:
            t3_extra = "baseline ONNX 미로드"
        else:
            svc.cleanup(cid3)
            await svc.enroll_baseline_async(cid3, dummy_pcm)
            assert cid3 in svc._baseline_voiceprints
            vp = svc._baseline_voiceprints[cid3]
            assert vp.ndim == 1
            vn = float(np.linalg.norm(vp))
            assert abs(vn - 1.0) < 1e-5, f"voiceprint norm={vn}"
            t3 = "PASS"
            t3_extra = f"voiceprint shape={vp.shape} norm={vn:.6f}"
    except Exception as e:
        t3_extra = str(e)
    lines.append(f"테스트 3 — enrollment 등록:    {t3}  {t3_extra}")

    # ----- 테스트 4 -----
    t4 = "FAIL"
    t4_extra = ""
    cid4 = "test-call-no-enroll-local"
    try:
        svc.cleanup(cid4)
        snap = await svc.verify_compare_async(
            dummy_pcm,
            cid4,
            0,
            onnx_ft,
        )
        d = _snap_dict(snap)
        assert d["bypass"] is True
        assert d["baseline_similarity"] == -1.0
        assert d["finetuned_similarity"] == -1.0
        t4 = "PASS"
    except Exception as e:
        t4_extra = str(e)
    lines.append(f"테스트 4 — bypass 동작:        {t4}" + (f"  {t4_extra}" if t4_extra else ""))

    # ----- 테스트 5 -----
    t5_line = "테스트 5 — 본인/타인 score:    SKIP (ONNX 미로드)"
    cid5 = "test-call-002-local"
    other_pcm = np.random.default_rng(42).integers(-1000, 1000, 16000, dtype=np.int16).tobytes()
    try:
        if onnx_ft.load_error or not svc.models_ready:
            if onnx_ft.load_error:
                t5_line = f"테스트 5 — 본인/타인 score:    SKIP (ONNX load_error={onnx_ft.load_error!r})"
        else:
            svc.cleanup(cid5)
            onnx_ft.cleanup(cid5)
            n_need = max(1, int(settings.enroll_utt_count))
            await svc.enroll_baseline_async(cid5, dummy_pcm)
            await _onnx_enroll_same_pcm(cid5, dummy_pcm, n_need, onnx_ft)
            r_same = await svc.verify_compare_async(dummy_pcm, cid5, 1, onnx_ft)
            r_other = await svc.verify_compare_async(other_pcm, cid5, 2, onnx_ft)
            bs, bo = r_same.baseline_similarity, r_other.baseline_similarity
            fs, fo = r_same.finetuned_similarity, r_other.finetuned_similarity
            t5_line = (
                f"테스트 5 — 본인/타인 score:    baseline 본인={bs:.4f} 타인={bo:.4f} / "
                f"finetuned 본인={fs:.4f} 타인={fo:.4f}"
            )
            ok5 = (bs > bo) and (fs > fo) and abs(bs - 1.0) > 1e-6 and abs(fs - 1.0) > 1e-6
            if not ok5:
                t5_line += "  (참고: dummy silence·랜덤 조합에서 항상 본인>타인은 보장되지 않을 수 있음)"
    except Exception as e:
        t5_line = f"테스트 5 — 본인/타인 score:    FAIL {e}"
    lines.append(t5_line)

    # ----- 테스트 6 -----
    t6 = "FAIL"
    t6_extra = ""
    log_path = _resolve_compare_csv_path()
    try:
        svc.cleanup("test-csv-row")
        onnx_ft.cleanup("test-csv-row")
        await svc.enroll_baseline_async("test-csv-row", dummy_pcm)
        await _onnx_enroll_same_pcm("test-csv-row", dummy_pcm, max(1, int(settings.enroll_utt_count)), onnx_ft)
        # 등록과 동일 무음으로 verify 하면 코사인≈1.0이 정상이라, CSV 검증용으로만 약간 변형한 PCM 사용
        _pcm_arr = np.frombuffer(dummy_pcm, dtype=np.int16).copy()
        _pcm_arr[len(_pcm_arr) // 2] = np.int16(
            min(max(int(_pcm_arr[len(_pcm_arr) // 2]) + 300, -32768), 32767)
        )
        pcm_verify_for_csv = _pcm_arr.tobytes()
        snap6 = await svc.verify_compare_async(pcm_verify_for_csv, "test-csv-row", 3, onnx_ft)
        svc.flush_csv_row_sync(
            snap6,
            transcript_finetuned="local-ft",
            transcript_baseline="local-bl",
            transcript_no_verify="local-nv",
            stt_finetuned_executed=True,
            stt_baseline_executed=True,
        )
        if not log_path.is_file():
            t6_extra = f"CSV 없음 path={log_path}"
        else:
            with log_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ours = [r for r in rows if r.get("call_id") == "test-csv-row"]
            if not ours:
                t6_extra = "call_id=test-csv-row 인 row 없음"
            else:
                last = ours[-1]
                bl = float(last["baseline_similarity"])
                ft = float(last["finetuned_similarity"])
                exp_cols = {
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
                    "baseline_ok",
                    "finetuned_ok",
                    "stt_executed",
                    "transcript",
                    "stt_finetuned_executed",
                    "stt_baseline_executed",
                    "transcript_finetuned",
                    "transcript_baseline",
                    "transcript_no_verify",
                }
                assert set(last.keys()) == exp_cols, f"컬럼 불일치: {set(last.keys()) ^ exp_cols}"
                bypass = last.get("bypass", "").lower() == "true"
                if bypass:
                    assert bl == -1.0 and ft == -1.0
                else:
                    # 무음·가짜 음성이 아닌 이상 1.0으로만 고정되는 버그 방지(자기대칭 제외 후 검증)
                    assert not (abs(bl - 1.0) < 1e-6 and abs(ft - 1.0) < 1e-6)
                t6 = "PASS"
                t6_extra = f"test-csv-row rows={len(ours)} total_csv_rows={len(rows)} bl={bl:.4f} ft={ft:.4f}"
    except Exception as e:
        t6_extra = str(e)
    lines.append(f"테스트 6 — CSV 저장:           {t6}  {t6_extra}")

    lines.append("")
    lines.append("확인된 문제:")
    lines.append("  (스크립트 실행 중 예외는 위 FAIL 항목 참고)")
    lines.append("")
    lines.append("다음 단계:")
    lines.append("  - threshold 재보정 필요 여부: FAIL·score 범위 보고 판단")
    lines.append("  - 실제 음성 WAV 로 테스트 5 재실행 권장")

    print("\n".join(lines))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

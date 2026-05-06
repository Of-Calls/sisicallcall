"""Dev server launcher — uvicorn 실행 + 콘솔/파일 동시 로깅.

- logs/{YYYY-MM-DD}/stdout_{HHMMSS}.log 자동 생성
- print/logger/uvicorn 출력 통째 캡처 (stderr → stdout 통합)
- 콘솔에도 그대로 출력 (Tee-like)

사용 (cmd):
    venv\\Scripts\\python.exe scripts\\run_dev.py
또는:
    scripts\\run_dev.bat

종료: Ctrl+C
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    # 프로젝트 루트로 이동 (스크립트 위치 기준)
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    now = datetime.datetime.now()
    log_dir = Path("logs") / now.strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"stdout_{now.strftime('%H%M%S')}.log"

    print(f"[run_dev] log file: {log_path}")
    print(f"[run_dev] starting server — Ctrl+C to stop")
    print(flush=True)

    # uvicorn — venv python 직접 호출 (현재 인터프리터)
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,        # stderr → stdout 통합 (NativeCommandError 우회)
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.stdout is not None
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as f:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
    except KeyboardInterrupt:
        print("\n[run_dev] terminating...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())

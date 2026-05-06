@echo off
REM scripts/run_dev.bat — cmd 창에서 한 줄로 dev 서버 실행.
REM
REM 동작:
REM   - chcp 65001 로 cmd 코드페이지 UTF-8 (한글 깨짐 방지)
REM   - venv python 으로 scripts/run_dev.py 호출
REM     → logs/{YYYY-MM-DD}/stdout_{HHMMSS}.log 자동 생성 + 콘솔 동시 출력
REM
REM 사용:
REM   cmd> scripts\run_dev.bat

chcp 65001 >nul
cd /d "%~dp0\.."
venv\Scripts\python.exe -X utf8 scripts\run_dev.py

import datetime
import glob
import logging
import logging.handlers
import os
import sys

# 크로스플랫폼 ANSI 색상 지원 — Windows cmd/PowerShell 모두에서 색상 출력.
# colorama 가 없을 때를 대비한 fallback 으로 os.system("") 도 실행.
try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    if os.name == "nt":
        os.system("")

# Windows 콘솔 한글 깨짐 방지 — stdout 을 UTF-8 로 강제 재구성 (Python 3.7+).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


class _ColorFormatter(logging.Formatter):
    """터미널 가독성 향상 — 레벨별 색상 + 모듈명 축약 + 시각만 표시."""

    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",      # cyan
        "INFO":     "\033[32m",      # green
        "WARNING":  "\033[33m",      # yellow
        "ERROR":    "\033[31m",      # red
        "CRITICAL": "\033[1;31m",    # bold red
    }
    DIM = "\033[2m"
    RESET = "\033[0m"

    # 자주 쓰는 모듈은 짧은 별칭으로 — 가독성 + 정렬 폭 절약.
    SHORT_ALIASES = {
        "app.api.v1.call":                                 "api.call",
        "app.services.tts.twilio_channel":                 "tts.twilio",
        "app.services.tts.azure":                          "tts.azure",
        "app.services.stt.deepgram_streaming":             "stt.stream",
        "app.services.stt.deepgram_prerecorded":           "stt.prerec",
        "app.services.speaker_verify.titanet":             "verify.titanet",
        "app.services.cache.semantic_cache":               "cache.sem",
        "app.services.embedding.local":                    "embed.local",
        "app.services.rag.chroma":                         "rag.chroma",
        "app.services.session.redis_session":              "session.redis",
        "app.agents.conversational.graph":                 "graph",
    }

    @classmethod
    def _short_name(cls, full: str) -> str:
        if full in cls.SHORT_ALIASES:
            return cls.SHORT_ALIASES[full]
        # 노드 모듈은 'node:name' 형태로 — graph 단계 식별 용이
        if ".nodes." in full:
            tail = full.rsplit(".", 1)[-1].removesuffix("_node")
            return f"node.{tail}"
        # 그 외에는 마지막 segment 만
        return full.rsplit(".", 1)[-1]

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        record.colored_level = f"{color}{record.levelname:<7}{self.RESET}"
        record.short_name = f"{self._short_name(record.name):<16}"
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    """파일 출력용 — ANSI 색상 제거, 날짜 포함, 모듈명 축약은 유지."""

    def format(self, record: logging.LogRecord) -> str:
        record.short_name = f"{_ColorFormatter._short_name(record.name):<16}"
        return super().format(record)


# 루트 로거 파일 핸들러 1회 설정 — 여러 get_logger() 호출에서도 중복 방지.
_root_file_handler_added = False

# 서버 실행 단위로 로그 파일을 분리할 때 보관할 최대 일수 (이보다 오래된 파일 자동 삭제).
_LOG_RETENTION_DAYS = 7


def _expand_log_path(raw_path: str, started_at: datetime.datetime) -> str:
    """LOG_FILE 경로에 서버 시작 timestamp 를 끼워 실행 단위 파일명 생성.

    예시:
        logs/server.log         → logs/server_20260423_233458.log
        logs/app                → logs/app_20260423_233458
        logs/foo/bar.txt        → logs/foo/bar_20260423_233458.txt
    """
    ts = started_at.strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(raw_path)
    return f"{base}_{ts}{ext or '.log'}"


def _cleanup_old_logs(raw_path: str, retention_days: int) -> None:
    """LOG_FILE 베이스 패턴(`logs/server_*.log`) 중 retention 초과 파일 삭제.

    같은 prefix(`logs/server_`) + suffix(`.log`) 인 파일만 대상으로 하므로
    다른 시스템의 로그를 실수로 지우지 않는다.
    """
    base, ext = os.path.splitext(raw_path)
    pattern = f"{base}_*{ext or '.log'}"
    cutoff = datetime.datetime.now().timestamp() - retention_days * 86400
    for path in glob.glob(pattern):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass  # 동시 접근/삭제 race 무시


def _ensure_root_file_handler() -> None:
    """LOG_FILE 환경변수가 설정되어 있으면 루트 로거에 파일 핸들러 부착.

    각 named logger 는 자체 스트림 핸들러(컬러) 를 가지고, 레코드를 루트로
    propagate 하여 파일에도 함께 기록한다.

    파일명에 서버 시작 timestamp 를 부착하여 실행 단위로 파일이 분리된다:
        logs/server.log → logs/server_20260423_233458.log
    7일 이상 된 파일은 자동 정리.

    활성화 방법:
        LOG_FILE=logs/server.log uvicorn app.main:app --reload
        (PowerShell)  $env:LOG_FILE="logs/server.log"; uvicorn app.main:app --reload
    """
    global _root_file_handler_added
    if _root_file_handler_added:
        return

    raw_log_file = os.getenv("LOG_FILE")
    if not raw_log_file:
        _root_file_handler_added = True  # 비활성 상태도 1회만 결정
        return

    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        _root_file_handler_added = True
        return

    # 대상 디렉토리 자동 생성 (예: logs/ 가 없어도 동작)
    log_dir = os.path.dirname(raw_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    started_at = datetime.datetime.now()
    actual_path = _expand_log_path(raw_log_file, started_at)

    # 오래된 로그 정리 (best-effort)
    _cleanup_old_logs(raw_log_file, _LOG_RETENTION_DAYS)

    file_handler = logging.FileHandler(actual_path, encoding="utf-8")
    file_handler.setFormatter(
        _PlainFormatter(
            "%(asctime)s %(levelname)-7s %(short_name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))

    _root_file_handler_added = True
    root.info(
        "로그 파일 기록 시작 — path=%s (실행 단위 timestamp, %d일 보관)",
        actual_path, _LOG_RETENTION_DAYS,
    )


def get_logger(name: str) -> logging.Logger:
    _ensure_root_file_handler()

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColorFormatter(
            "%(asctime)s %(colored_level)s %(short_name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    # LOG_FILE 설정 시 루트의 파일 핸들러로 전달하기 위해 propagate=True,
    # 아니면 False 로 차단해 외부 라이브러리 핸들러와의 중복 출력을 방지.
    logger.propagate = bool(os.getenv("LOG_FILE"))
    return logger

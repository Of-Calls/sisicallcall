from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Twilio — REST/검증에 쓰는 키 이름(대문자)은 pydantic-settings 가 필드명에서 유도: twilio_account_sid → TWILIO_ACCOUNT_SID
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    base_url: str = "http://localhost:8000"

    # OpenAI
    openai_api_key: str = ""

    # Deepgram
    deepgram_api_key: str = ""
    # True: Twilio μ-law → Deepgram live WebSocket (nova-3), is_final 마다 화자 검증·로그.
    # False: Silero VAD 발화 경계 + asyncprerecorded 파일 전사(기존).
    deepgram_use_streaming: bool = Field(
        default=False,
        validation_alias=AliasChoices("DEEPGRAM_USE_STREAMING"),
    )

    # PostgreSQL
    postgres_user: str = "sisicallcall"
    postgres_password: str = "changeme"
    postgres_db: str = "sisicallcall"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = "postgresql://sisicallcall:changeme@localhost:5432/sisicallcall"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str = "redis://localhost:6379"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # App
    env: str = "development"
    log_level: str = "INFO"

    # TTS Output Channel 모드 — "mock" (기본, 테스트/유닛) | "twilio" (프로덕션 WebSocket)
    tts_channel_mode: str = "mock"

    # TitaNet 화자 검증 — onnxruntime + 로컬 ONNX. mel 은 torchaudio 기본, 선택 시 NeMo preprocessor(.nemo).
    # 코사인 유사도 ≥ threshold 일 때만 검증 통과. 짧은 발화·전화 음질에 따라 본인 거절(FRR)↑ 가능.
    speaker_verify_threshold: float = Field(
        default=0.40,
        validation_alias=AliasChoices(
            "SPEAKER_VERIFY_THRESHOLD",
            "TITANET_SIMILARITY_THRESHOLD",
        ),
    )
    speaker_verify_finetuned_threshold: float | None = Field(
        default=None,
        validation_alias=AliasChoices("SPEAKER_VERIFY_FINETUNED_THRESHOLD"),
    )
    titanet_enrollment_sec: float = 3.0  # 레거시(초 단위 PCM 누적). 등록 로직은 enroll_utt_count 발화 기준.
    # enrollment: STT 성공 발화마다 finetuned 임베딩 수집 → N개 도달 시 평균·L2 정규화 후 저장.
    enroll_utt_count: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices("ENROLL_UTT_COUNT"),
    )
    # ONNX mel: "torchaudio"(기본) | "nemo" — nemo 시 학습과 동일 preprocessor(.nemo) 사용.
    titanet_mel_backend: str = Field(
        default="torchaudio",
        validation_alias=AliasChoices("TITANET_MEL_BACKEND"),
    )
    titanet_speaker_nemo_path: str = Field(
        default="",
        validation_alias=AliasChoices("TITANET_SPEAKER_NEMO_PATH"),
    )
    # finetuned ONNX만: 긴 발화 PCM 상한(초). mel 계산·메모리 완화. 0 이하면 상한 없음.
    # STFT center·hop 정렬 때문에 이 한도만으로 mel T 가 ONNX 내부 상한과 일치하지 않을 수 있음 → TITANET_FINETUNED_ONNX_MAX_MEL_FRAMES.
    titanet_finetuned_infer_max_sec: float = 12.0
    # finetuned ONNX mel 시간축 T 상한(프레임). 0 이면 런타임 안전 기본 1200 사용(Where 1200×1201 방지).
    # 더 큰 T 가 필요하면 명시적으로 큰 값(예: 2000)을 두고, 재export 로 그래프를 고치는 것이 근본 해결.
    titanet_finetuned_onnx_max_mel_frames: int = Field(
        default=0,
        validation_alias=AliasChoices("TITANET_FINETUNED_ONNX_MAX_MEL_FRAMES"),
    )
    # 병렬 통화 / enrollment — 빈 문자열이면 레포 models/speech_verification/ 기본 ONNX
    titanet_finetuned_onnx_path: str = Field(
        default="",
        validation_alias=AliasChoices("TITANET_FINETUNED_ONNX_PATH"),
    )
    # True면 기동 시 파인튜닝 ONNX까지 동시 로드. False면 NeMo mel(설정 시)만 즉시, 파인튜닝은 백그라운드.
    # (환경 변수명 PRELOAD_FINETUNED_NEMO_AT_STARTUP는 기존 .env 호환용)
    preload_finetuned_nemo_at_startup: bool = False
    # TITANET_MEL_BACKEND=nemo 일 때만 해당. False(기본)=NeMo restore 는 첫 mel 사용 시(기동 빠름).
    # True=기존처럼 기동 시 restore(첫 통화 지연 없음, 대신 startup 이 길어짐). CPU만 쓸 때 특히 false 권장.
    warmup_nemo_mel_at_startup: bool = Field(
        default=False,
        validation_alias=AliasChoices("WARMUP_NEMO_MEL_AT_STARTUP"),
    )
    # ONNX mel 전처리 (TitaNet 16kHz 스펙트로그 관례; 학습 yaml과 다르면 덮어쓰기)
    titanet_onnx_mel_n_fft: int = 512
    titanet_onnx_mel_n_mels: int = 80
    titanet_onnx_mel_win_length: int = 400  # 25 ms @ 16 kHz
    titanet_onnx_mel_hop_length: int = 160  # 10 ms @ 16 kHz
    titanet_onnx_mel_fmin: float = 0.0
    titanet_onnx_mel_fmax: float = 8000.0
    speaker_verify_enabled: bool = True
    # 통화 발화마다 동일 mel로 NeMo forward_for_export vs 파인튜닝 ONNX 임베딩 코사인 로그 (NeMo+CPU/GPU 부하).
    speaker_verify_nemo_onnx_compare_on_call: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "SPEAKER_VERIFY_NEMO_ONNX_COMPARE_ON_CALL",
        ),
    )
    # 순정·파인튜닝 ONNX 동시 추론·CSV. 경로별 STT는 각 모델 임계값(baseline_ok / finetuned_ok)만 사용.
    speaker_verify_compare_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("SPEAKER_VERIFY_COMPARE_ENABLED"),
    )
    speaker_verify_gate_model: Literal["baseline", "finetuned"] = Field(
        default="finetuned",
        description="레거시·.env 호환. 이중 ONNX compare 스냅샷·CSV에서는 미사용.",
        validation_alias=AliasChoices("SPEAKER_VERIFY_GATE_MODEL"),
    )
    speaker_verify_compare_log_path: str = Field(
        default="logs/speaker_verify_compare.csv",
        validation_alias=AliasChoices("SPEAKER_VERIFY_COMPARE_LOG_PATH"),
    )
    # 비교 baseline: 순정(또는 비파인튜닝) TitaNet ONNX. 비우면 models/speech_verification/titanet-s.onnx
    speaker_verify_compare_baseline_onnx_path: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SPEAKER_VERIFY_COMPARE_BASELINE_ONNX_PATH",
        ),
    )
    # 비어 있으면 from_pretrained("titanet_small") — 캐시/허브. 로컬 .nemo 절대·상대 경로면 다운로드 없이 복원.
    speaker_verify_nemo_baseline_nemo_path: str = Field(
        default="",
        validation_alias=AliasChoices("SPEAKER_VERIFY_NEMO_BASELINE_NEMO_PATH"),
    )
    # True면 원격 체크포인트 캐시 갱신(이름 로드 시). 로컬 .nemo 경로일 때는 보통 불필요.
    speaker_verify_nemo_baseline_refresh_cache: bool = Field(
        default=False,
        validation_alias=AliasChoices("SPEAKER_VERIFY_NEMO_BASELINE_REFRESH_CACHE"),
    )
    # True: NeMo 로드 전 CUDA_VISIBLE_DEVICES=-1 (CPU 전용·torch CUDA 미초기화 시에만 효과). GPU 서버에서는 false 유지.
    speaker_verify_nemo_force_cpu: bool = Field(
        default=False,
        validation_alias=AliasChoices("SPEAKER_VERIFY_NEMO_FORCE_CPU"),
    )

    @field_validator("speaker_verify_gate_model", mode="before")
    @classmethod
    def _normalize_speaker_verify_gate_model(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    # True면 서버 기동 시 in-memory voiceprint·enrollment 전역을 비움 (오염 임베딩 제거).
    reset_voiceprint_on_startup: bool = False
    # Twilio 로컬 테스트 시 화자검증 CSV·웹훅 안내용 디버그 GET (운영에서는 false).
    call_debug_routes_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("CALL_DEBUG_ROUTES_ENABLED"),
    )

    # Silero VAD (v6.2+, 2026-04-30 채택 — 짧은 발화 + 긴 trailing silence reject 해결).
    # logs/2026-04-30/server_100651.log Turn 4/5 사례: "예약은어떻게해요" 0.5s + trailing 1.3s
    # → WebRTC bulk ratio 28~38% reject → graph END. Silero per-frame 누적으로 해결.
    silero_threshold: float = 0.5  # speech 확률 임계값 (Silero 기본). 낮추면 잡음 통과↑
    silero_min_speech_frames: int = 3  # 청크 내 speech frame (32ms each) 최소 개수.
    # 3 = 96ms — 짧은 단어 ("응", "예") 까지 통과
    silero_use_onnx: bool = False  # ONNX runtime 가속 (~2x faster). PyTorch JIT 가
    # default — 첫 운영 안정화 후 True 전환 검토.

    # TTS 합성 엔진 — "azure" (Azure Speech SDK, μ-law 8kHz 네이티브 출력) 단일화
    tts_provider: str = "azure"
    # Azure Speech (TTS) — Korean Neural Voice
    azure_speech_key: str = ""
    azure_speech_region: str = ""  # e.g. "koreacentral", "eastus"
    azure_tts_voice: str = "ko-KR-SunHiNeural"

    # TTS Throttle (barge-in 정확도용)
    # Twilio jitter buffer 가 11~14초간 음성 재생하므로 송신 속도 ≈ 재생 속도 로 맞춰
    # cancel 즉시 효과 + is_speaking 정확도 자연 확보. 음성 끊김 발생 시 enabled=False
    # 로 즉시 끄고 재시작 가능. Linux 운영 정상, Windows 개발은 chunk_interval 0.015 보정.
    tts_throttle_enabled: bool = True
    tts_preroll_chunks: int = 20  # 처음 N 청크는 즉시 송신 (시작 latency + 지터 흡수)
    tts_chunk_interval_sec: float = 0.020  # 청크 사이 throttle (160B / 8kHz = 20ms)
    tts_play_tail_margin_sec: float = 0.15  # 송신 후 jitter buffer 잔여 재생 마진

    # Barge-in verify (Phase B — VAD + 화자검증 게이트)
    # TTS 송신 중 사용자 발화로 보이는 신호가 들어오면 첫 0.8초를 추출해
    # WebRTC VAD (음성 vs 잡음) + TitaNet (등록 화자 vs 타인/echo) 통과한 경우에만
    # BARGE-IN 트리거. enrollment 미완료 시 TitaNet 가 자동 bypass(True) 반환 →
    # RMS-only 동작으로 자연스럽게 fallback. 문제 시 enabled=false 로 즉시 PR1~3 동작.
    bargein_verify_enabled: bool = True
    bargein_rms_pre_threshold: int = (
        1500  # verify 게이트 진입 RMS (echo 임계값 2400 보다 낮음)
    )
    bargein_verify_chunk_bytes: int = 25600  # 0.8s × 16kHz × 2byte (PCM16 mono)
    bargein_verify_chunk_sec: float = 0.8  # 디버그/로그용

    # SMS Provider — "solapi" (기본) | "twilio"
    sms_provider: str = "solapi"
    solapi_api_key: str = ""
    solapi_api_secret: str = ""
    solapi_sender_number: str = ""

    # Face Auth (M3+)
    arcface_model_name: str = "buffalo_l"
    arcface_similarity_threshold: float = 0.6
    arcface_max_retries: int = 3
    liveness_instruction_count: int = 3
    liveness_hmac_secret: str = "change-me-in-production"
    auth_session_ttl_sec: int = 600
    auth_enable_test_register: bool = False
    auth_web_base_url: str = "http://localhost:3000"

    # extra="ignore" — .env 에 코드에서 제거된 잔여 키(예: 과거 GOOGLE_APPLICATION_CREDENTIALS)
    # 가 있어도 ValidationError 로 죽지 않게. 신규 키는 위 클래스 필드로 명시 정의 필요.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

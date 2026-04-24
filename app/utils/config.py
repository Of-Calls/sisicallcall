from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    base_url: str = "http://localhost:8000"

    # OpenAI
    openai_api_key: str = ""

    # Deepgram
    deepgram_api_key: str = ""

    # Google Cloud TTS
    google_application_credentials: str = ""

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

    # KNN Router (신용 연구 완료 후 확정)
    knn_confidence_threshold: float = 0.85

    # WebRTC VAD (주미 연구 결과 적용)
    webrtc_energy_fallback_threshold: int = 1200
    webrtc_mode: int = 3
    webrtc_frame_ms: int = 20
    webrtc_speech_ratio_threshold: float = 0.3

    # TTS Output Channel 모드 — "mock" (기본, 테스트/유닛) | "twilio" (프로덕션 WebSocket)
    tts_channel_mode: str = "mock"

    # TitaNet 화자 검증 (대영 R-01 연구 결과 — titanet_large 채택)
    titanet_model_name: str = "titanet_large"
    titanet_similarity_threshold: float = 0.40
    titanet_enrollment_sec: float = 3.0  # voiceprint 등록에 사용할 첫 발화 누적 시간

    # TTS 합성 엔진 — "google" (Cloud TTS) | "xtts" (Coqui XTTS v2 로컬, 팀원 목소리)
    tts_provider: str = "google"
    # XTTS reference audio 경로 (zero-shot voice cloning용 WAV 파일)
    xtts_reference_path: str = "voice/speaker_reference.wav"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

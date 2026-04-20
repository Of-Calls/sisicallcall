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
    database_url: str = "postgresql://sisicollcoll:password@localhost:5432/sisicollcoll"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # App
    env: str = "development"
    log_level: str = "INFO"

    # KNN Router (신용 연구 완료 후 확정)
    knn_confidence_threshold: float = 0.85

    # VAD (KDT-30 threshold 연구 완료 후 확정)
    vad_threshold: float = 0.5

    # Speaker Enrollment (KDT-40 실험용 — 3.0 / 5.0 / 10.0 전환 가능)
    enrollment_target_sec: float = 5.0

    # TitaNet (KDT-39 실험용 — threshold sweep: .env의 TITANET_SIMILARITY_THRESHOLD로 덮어쓰기 가능)
    titanet_model_name: str = "titanet_small"
    titanet_similarity_threshold: float = 0.40

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

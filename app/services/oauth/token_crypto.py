"""
Fernet 대칭 암호화로 OAuth 토큰을 암/복호화한다.

env: TOKEN_ENCRYPTION_KEY — base64url-safe 32바이트 Fernet 키
     생성: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken

from app.utils.logger import get_logger

logger = get_logger(__name__)

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = os.getenv("TOKEN_ENCRYPTION_KEY", "")
        if not key:
            raise RuntimeError(
                "TOKEN_ENCRYPTION_KEY env 가 설정되지 않았습니다. "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
                "로 키를 생성하고 .env 에 설정하세요."
            )
        _fernet = Fernet(key.encode())
    return _fernet


def encrypt_token(plaintext: str) -> str:
    """평문 토큰 → Fernet 암호화 str (base64url)."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    """Fernet 암호화 str → 평문 토큰. 복호화 실패 시 ValueError."""
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("토큰 복호화 실패: 키가 다르거나 데이터가 변조되었습니다.") from exc


def reset_fernet_cache() -> None:
    """테스트 격리용 — Fernet 캐시를 초기화한다."""
    global _fernet
    _fernet = None

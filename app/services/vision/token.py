import hashlib
import secrets


def create_upload_token() -> str:
    return secrets.token_urlsafe(24)


def hash_upload_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

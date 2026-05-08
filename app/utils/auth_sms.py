"""본인인증 SMS 본문 빌더."""

from app.utils.config import settings


def auth_public_base() -> str:
    """AUTH_WEB_BASE_URL 정규화 (스킴 없으면 https 붙임)."""
    b = (settings.auth_web_base_url or "").strip().rstrip("/")
    if not b:
        return ""
    if not b.lower().startswith(("http://", "https://")):
        b = "https://" + b.lstrip("/")
    return b


def face_auth_url(auth_id: str) -> str:
    base = auth_public_base()
    return f"{base}/auth/{auth_id}"


def ocr_auth_url(auth_id: str) -> str:
    base = auth_public_base()
    return f"{base}/ocr-auth/{auth_id}"


def build_face_auth_sms(auth_id: str) -> str:
    """얼굴 인증 링크 단독 발송용."""
    url = face_auth_url(auth_id)
    return (
        "[시시콜콜] 얼굴 인증 링크\n"
        "아래 주소를 눌러 얼굴 인증을 진행해주세요.\n\n"
        f"{url}\n"
    )


def build_ocr_auth_sms(auth_id: str) -> str:
    """신분증 OCR 링크 단독 발송용."""
    url = ocr_auth_url(auth_id)
    return (
        "[시시콜콜] 신분증 OCR 링크\n"
        "아래 주소를 눌러 신분증 촬영을 진행해주세요.\n\n"
        f"{url}\n"
    )


def build_auth_links_sms(auth_id: str) -> str:
    """하위 호환: 기존 통합 본문."""
    base = auth_public_base()
    face_url = f"{base}/auth/{auth_id}"
    ocr_url = f"{base}/ocr-auth/{auth_id}"
    return (
        "[시시콜콜] 본인인증\n"
        "아래 주소는 줄 단위로 눌러 열어주세요.\n"
        "\n"
        "(1) 얼굴 Liveness\n"
        f"{face_url}\n"
        "\n"
        "(2) 신분증 OCR\n"
        f"{ocr_url}\n"
        "\n"
        "순서: (1) Liveness 완료 → (2) OCR → 다시 (1)에서 정면 촬영"
    )

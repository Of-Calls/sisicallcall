import hashlib
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v1.vision as vision_mod
from app.services.sms import get_sms_service
from app.services.sms.mock import MockSMSService
from app.services.vision.session_store import (
    VISION_STATUS_PROCESSING,
    VISION_STATUS_WAITING_UPLOAD,
    VisionUploadSessionAlreadyUsedError,
    VisionUploadSessionNotFoundError,
    VisionUploadSessionStore,
)
from app.services.vision.token import hash_upload_token


def test_hash_upload_token_uses_sha256_hex():
    token = "fixed-token"
    assert hash_upload_token(token) == hashlib.sha256(token.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_create_upload_session_stores_hash_key_and_ttl():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()

    data = await store.create_upload_session(
        token_hash="abc123hash",
        call_id="CAxxxxxxxx",
        tenant_id="tenant-123",
        caller_number="+821012345678",
        expires_in_sec=600,
    )

    store._redis.hset.assert_awaited_once()
    hset_args = store._redis.hset.call_args
    assert hset_args.args[0] == "vision_upload:abc123hash"
    mapping = hset_args.kwargs["mapping"]
    assert "token" not in mapping
    assert mapping["status"] == VISION_STATUS_WAITING_UPLOAD
    assert mapping["used"] == "false"
    assert data["expires_at"]
    store._redis.expire.assert_awaited_once_with("vision_upload:abc123hash", 600)


@pytest.mark.asyncio
async def test_get_upload_session_returns_hash_data():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.hgetall = AsyncMock(
        return_value={
            "call_id": "CAxxxxxxxx",
            "tenant_id": "tenant-123",
            "caller_number": "+821012345678",
            "status": VISION_STATUS_WAITING_UPLOAD,
            "used": "false",
            "created_at": "2026-04-30T00:00:00+00:00",
            "expires_at": "2026-04-30T00:10:00+00:00",
        }
    )

    data = await store.get_upload_session("abc123hash")

    store._redis.hgetall.assert_awaited_once_with("vision_upload:abc123hash")
    assert data["status"] == VISION_STATUS_WAITING_UPLOAD
    assert data["used"] == "false"


@pytest.mark.asyncio
async def test_get_upload_session_returns_none_when_missing():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.hgetall = AsyncMock(return_value={})

    data = await store.get_upload_session("missinghash")

    assert data is None


@pytest.mark.asyncio
async def test_set_call_vision_status_saves_when_call_session_exists():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.exists = AsyncMock(return_value=1)

    await store.set_call_vision_status("CAxxxxxxxx", VISION_STATUS_WAITING_UPLOAD)

    store._redis.exists.assert_awaited_once_with("call:CAxxxxxxxx:session")
    store._redis.hset.assert_awaited_once_with(
        "call:CAxxxxxxxx:session",
        "vision_status",
        VISION_STATUS_WAITING_UPLOAD,
    )


@pytest.mark.asyncio
async def test_set_call_vision_status_skips_when_call_session_missing():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.exists = AsyncMock(return_value=0)

    await store.set_call_vision_status("CAxxxxxxxx", VISION_STATUS_WAITING_UPLOAD)

    store._redis.exists.assert_awaited_once_with("call:CAxxxxxxxx:session")
    store._redis.hset.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_call_vision_status_does_not_raise_on_redis_failure():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.exists = AsyncMock(side_effect=ConnectionError("redis down"))

    await store.set_call_vision_status("CAxxxxxxxx", VISION_STATUS_WAITING_UPLOAD)


def test_get_sms_service_returns_mock_for_mock_provider(monkeypatch):
    monkeypatch.setattr(vision_mod.settings, "sms_provider", "mock")

    service = get_sms_service()

    assert isinstance(service, MockSMSService)


def test_get_sms_service_raises_for_unknown_provider(monkeypatch):
    monkeypatch.setattr(vision_mod.settings, "sms_provider", "bogus")

    with pytest.raises(ValueError, match="Unsupported SMS_PROVIDER"):
        get_sms_service()


def test_parse_redis_bool():
    assert vision_mod._parse_redis_bool("true") is True
    assert vision_mod._parse_redis_bool("false") is False
    assert vision_mod._parse_redis_bool(True) is True
    assert vision_mod._parse_redis_bool(False) is False


def test_post_upload_sessions_returns_url_and_sends_sms(monkeypatch):
    token = "fixed-token"
    expected_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    class FakeStore:
        def __init__(self):
            self.create_args = None
            self.call_status_args = None

        async def create_upload_session(self, **kwargs):
            self.create_args = kwargs
            return {}

        async def set_call_vision_status(self, call_id: str, status: str):
            self.call_status_args = (call_id, status)

    class FakeSMS:
        def __init__(self):
            self.sent = None

        async def send_sms(self, to: str, body: str) -> bool:
            self.sent = (to, body)
            return True

    fake_store = FakeStore()
    fake_sms = FakeSMS()
    monkeypatch.setattr(vision_mod, "create_upload_token", lambda: token)
    monkeypatch.setattr(vision_mod, "_session_store", fake_store)
    monkeypatch.setattr(vision_mod, "_sms_svc", fake_sms)
    monkeypatch.setattr(vision_mod.settings, "vision_upload_base_url", "http://localhost:5173")
    monkeypatch.setattr(vision_mod.settings, "vision_upload_path_prefix", "/v")
    monkeypatch.setattr(vision_mod.settings, "vision_upload_ttl_sec", 600)

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.post(
        "/vision/upload-sessions",
        json={
            "call_id": "CAxxxxxxxx",
            "tenant_id": "tenant-123",
            "caller_number": "+821012345678",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "upload_url": "http://localhost:5173/v/fixed-token",
        "expires_in_sec": 600,
        "sms_sent": True,
        "status": VISION_STATUS_WAITING_UPLOAD,
    }
    assert fake_store.create_args["token_hash"] == expected_hash
    assert fake_store.create_args["expires_in_sec"] == 600
    assert fake_store.call_status_args == ("CAxxxxxxxx", VISION_STATUS_WAITING_UPLOAD)
    assert fake_sms.sent[0] == "+821012345678"
    assert "http://localhost:5173/v/fixed-token" in fake_sms.sent[1]


def test_get_upload_session_returns_status(monkeypatch):
    token = "fixed-token"
    expected_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    class FakeStore:
        def __init__(self):
            self.token_hash = None

        async def get_upload_session(self, token_hash: str):
            self.token_hash = token_hash
            return {
                "call_id": "CAxxxxxxxx",
                "tenant_id": "tenant-123",
                "caller_number": "+821012345678",
                "status": VISION_STATUS_WAITING_UPLOAD,
                "used": "false",
                "created_at": "2026-04-30T00:00:00+00:00",
                "expires_at": "2026-04-30T00:10:00+00:00",
            }

    fake_store = FakeStore()
    monkeypatch.setattr(vision_mod, "_session_store", fake_store)

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.get("/vision/upload-sessions/fixed-token")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": VISION_STATUS_WAITING_UPLOAD,
        "expires_at": "2026-04-30T00:10:00+00:00",
        "used": False,
    }
    assert fake_store.token_hash == expected_hash


def test_get_upload_session_returns_404_when_missing(monkeypatch):
    class FakeStore:
        async def get_upload_session(self, token_hash: str):
            return None

    monkeypatch.setattr(vision_mod, "_session_store", FakeStore())

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.get("/vision/upload-sessions/missing-token")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "vision upload session not found or expired"


def test_get_upload_session_returns_409_when_used(monkeypatch):
    class FakeStore:
        async def get_upload_session(self, token_hash: str):
            return {
                "status": VISION_STATUS_WAITING_UPLOAD,
                "used": "true",
                "expires_at": "2026-04-30T00:10:00+00:00",
            }

    monkeypatch.setattr(vision_mod, "_session_store", FakeStore())

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.get("/vision/upload-sessions/used-token")

    assert resp.status_code == 409
    assert resp.json()["detail"] == "vision upload session already used"


def test_get_upload_session_returns_500_on_lookup_failure(monkeypatch):
    class FakeStore:
        async def get_upload_session(self, token_hash: str):
            raise ConnectionError("redis down")

    monkeypatch.setattr(vision_mod, "_session_store", FakeStore())

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.get("/vision/upload-sessions/error-token")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "vision upload session lookup failed"


@pytest.mark.asyncio
async def test_validate_image_upload_accepts_jpeg(monkeypatch):
    from io import BytesIO

    from starlette.datastructures import UploadFile

    from app.services.vision import validation

    monkeypatch.setattr(validation, "detect_mime_type", lambda content: "image/jpeg")

    file = UploadFile(filename="x.txt", file=BytesIO(b"fake-jpeg"))
    result = await validation.validate_image_upload(file, max_bytes=1024)

    assert result.mime_type == "image/jpeg"
    assert result.extension == ".jpg"
    assert result.size_bytes == len(b"fake-jpeg")


@pytest.mark.asyncio
async def test_validate_image_upload_rejects_empty_file():
    from io import BytesIO

    from starlette.datastructures import UploadFile

    from app.services.vision.validation import EmptyImageFileError, validate_image_upload

    file = UploadFile(filename="empty.png", file=BytesIO(b""))

    with pytest.raises(EmptyImageFileError):
        await validate_image_upload(file, max_bytes=1024)


@pytest.mark.asyncio
async def test_validate_image_upload_rejects_too_large_file():
    from io import BytesIO

    from starlette.datastructures import UploadFile

    from app.services.vision.validation import ImageFileTooLargeError, validate_image_upload

    file = UploadFile(filename="big.png", file=BytesIO(b"12345"))

    with pytest.raises(ImageFileTooLargeError):
        await validate_image_upload(file, max_bytes=4)


@pytest.mark.asyncio
async def test_validate_image_upload_rejects_unsupported_mime(monkeypatch):
    from io import BytesIO

    from starlette.datastructures import UploadFile

    from app.services.vision import validation
    from app.services.vision.validation import UnsupportedImageMimeTypeError

    monkeypatch.setattr(validation, "detect_mime_type", lambda content: "text/plain")

    file = UploadFile(filename="x.jpg", file=BytesIO(b"not-image"))

    with pytest.raises(UnsupportedImageMimeTypeError):
        await validation.validate_image_upload(file, max_bytes=1024)


@pytest.mark.asyncio
async def test_reserve_upload_session_for_processing_returns_session_data():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.eval = AsyncMock(
        return_value=[
            "ok",
            "call_id",
            "CAxxxxxxxx",
            "tenant_id",
            "tenant-123",
            "used",
            "true",
            "status",
            VISION_STATUS_PROCESSING,
        ]
    )

    data = await store.reserve_upload_session_for_processing("abc123hash")

    assert data == {
        "call_id": "CAxxxxxxxx",
        "tenant_id": "tenant-123",
        "used": "true",
        "status": VISION_STATUS_PROCESSING,
    }
    store._redis.eval.assert_awaited_once()
    assert store._redis.eval.call_args.args[1] == 1
    assert store._redis.eval.call_args.args[2] == "vision_upload:abc123hash"


@pytest.mark.asyncio
async def test_reserve_upload_session_for_processing_raises_when_missing():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.eval = AsyncMock(return_value=["missing"])

    with pytest.raises(VisionUploadSessionNotFoundError):
        await store.reserve_upload_session_for_processing("missinghash")


@pytest.mark.asyncio
async def test_reserve_upload_session_for_processing_raises_when_used():
    store = VisionUploadSessionStore()
    store._redis = AsyncMock()
    store._redis.eval = AsyncMock(return_value=["used"])

    with pytest.raises(VisionUploadSessionAlreadyUsedError):
        await store.reserve_upload_session_for_processing("usedhash")


def test_post_vision_upload_success(monkeypatch):
    from app.services.vision.validation import ValidatedImage

    token = "fixed-token"
    expected_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    class FakeStore:
        def __init__(self):
            self.reserve_token_hash = None
            self.attach_args = None
            self.call_status_args = None

        async def reserve_upload_session_for_processing(self, token_hash: str):
            self.reserve_token_hash = token_hash
            return {"call_id": "CAxxxxxxxx", "tenant_id": "tenant-123"}

        async def attach_upload_file_info(
            self,
            token_hash: str,
            image_path: str,
            mime_type: str,
            size_bytes: int,
        ):
            self.attach_args = (token_hash, image_path, mime_type, size_bytes)

        async def set_call_vision_status(self, call_id: str, status: str):
            self.call_status_args = (call_id, status)

    fake_store = FakeStore()
    save_args = {}

    async def fake_validate_image_upload(file, max_bytes: int):
        assert max_bytes == 1024
        return ValidatedImage(
            content=b"fake-jpeg",
            mime_type="image/jpeg",
            extension=".jpg",
            size_bytes=len(b"fake-jpeg"),
        )

    async def fake_save_vision_image(content, extension, call_id, upload_dir):
        save_args.update(
            {
                "content": content,
                "extension": extension,
                "call_id": call_id,
                "upload_dir": upload_dir,
            }
        )
        return "uploads/vision/CAxxxxxxxx_fake.jpg"

    monkeypatch.setattr(vision_mod, "_session_store", fake_store)
    monkeypatch.setattr(vision_mod, "validate_image_upload", fake_validate_image_upload)
    monkeypatch.setattr(vision_mod, "save_vision_image", fake_save_vision_image)
    monkeypatch.setattr(vision_mod.settings, "vision_upload_max_bytes", 1024)
    monkeypatch.setattr(vision_mod.settings, "vision_upload_dir", "uploads/vision")

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.post(
        "/vision/upload?token=fixed-token",
        files={"file": ("photo.jpg", b"fake-jpeg", "image/jpeg")},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "status": VISION_STATUS_PROCESSING,
        "message": "사진을 받았습니다. 제품을 분석하고 있습니다.",
    }
    assert fake_store.reserve_token_hash == expected_hash
    assert save_args == {
        "content": b"fake-jpeg",
        "extension": ".jpg",
        "call_id": "CAxxxxxxxx",
        "upload_dir": "uploads/vision",
    }
    assert fake_store.attach_args == (
        expected_hash,
        "uploads/vision/CAxxxxxxxx_fake.jpg",
        "image/jpeg",
        len(b"fake-jpeg"),
    )
    assert fake_store.call_status_args == ("CAxxxxxxxx", VISION_STATUS_PROCESSING)


def test_post_vision_upload_returns_404_when_session_missing(monkeypatch):
    from app.services.vision.validation import ValidatedImage

    class FakeStore:
        async def reserve_upload_session_for_processing(self, token_hash: str):
            raise VisionUploadSessionNotFoundError()

    async def fake_validate_image_upload(file, max_bytes: int):
        return ValidatedImage(
            content=b"fake-jpeg",
            mime_type="image/jpeg",
            extension=".jpg",
            size_bytes=len(b"fake-jpeg"),
        )

    monkeypatch.setattr(vision_mod, "_session_store", FakeStore())
    monkeypatch.setattr(vision_mod, "validate_image_upload", fake_validate_image_upload)

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.post(
        "/vision/upload?token=missing-token",
        files={"file": ("photo.jpg", b"fake-jpeg", "image/jpeg")},
    )

    assert resp.status_code == 404
    assert resp.json()["detail"] == "vision upload session not found or expired"


def test_post_vision_upload_returns_409_when_session_used(monkeypatch):
    from app.services.vision.validation import ValidatedImage

    class FakeStore:
        async def reserve_upload_session_for_processing(self, token_hash: str):
            raise VisionUploadSessionAlreadyUsedError()

    async def fake_validate_image_upload(file, max_bytes: int):
        return ValidatedImage(
            content=b"fake-jpeg",
            mime_type="image/jpeg",
            extension=".jpg",
            size_bytes=len(b"fake-jpeg"),
        )

    monkeypatch.setattr(vision_mod, "_session_store", FakeStore())
    monkeypatch.setattr(vision_mod, "validate_image_upload", fake_validate_image_upload)

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.post(
        "/vision/upload?token=used-token",
        files={"file": ("photo.jpg", b"fake-jpeg", "image/jpeg")},
    )

    assert resp.status_code == 409
    assert resp.json()["detail"] == "vision upload session already used"


def test_post_vision_upload_returns_400_when_mime_unsupported(monkeypatch):
    from app.services.vision.validation import UnsupportedImageMimeTypeError

    class FakeStore:
        async def reserve_upload_session_for_processing(self, token_hash: str):
            raise AssertionError("reserve should not be called")

    async def fake_validate_image_upload(file, max_bytes: int):
        raise UnsupportedImageMimeTypeError()

    monkeypatch.setattr(vision_mod, "_session_store", FakeStore())
    monkeypatch.setattr(vision_mod, "validate_image_upload", fake_validate_image_upload)

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.post(
        "/vision/upload?token=fixed-token",
        files={"file": ("photo.txt", b"not-image", "text/plain")},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "unsupported image MIME type"


def test_post_vision_upload_returns_404_when_token_missing(monkeypatch):
    class FakeStore:
        async def reserve_upload_session_for_processing(self, token_hash: str):
            raise AssertionError("reserve should not be called")

    monkeypatch.setattr(vision_mod, "_session_store", FakeStore())

    app = FastAPI()
    app.include_router(vision_mod.router, prefix="/vision")
    client = TestClient(app)

    resp = client.post(
        "/vision/upload",
        files={"file": ("photo.jpg", b"fake-jpeg", "image/jpeg")},
    )

    assert resp.status_code == 404
    assert resp.json()["detail"] == "vision upload session not found or expired"

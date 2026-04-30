import hashlib
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v1.vision as vision_mod
from app.services.sms import get_sms_service
from app.services.sms.mock import MockSMSService
from app.services.vision.session_store import (
    VISION_STATUS_WAITING_UPLOAD,
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

"""
OAuth 서비스 계층 테스트.

검증 범위:
  token_crypto:
    1.  encrypt_token → decrypt_token 왕복 성공
    2.  잘못된 키로 복호화 시 ValueError
    3.  TOKEN_ENCRYPTION_KEY 미설정 시 RuntimeError
    4.  reset_fernet_cache 후 새 키 적용

  state:
    5.  create_oauth_state → verify_oauth_state 성공 (1회 소모)
    6.  동일 state 두 번 검증 시 두 번째는 None
    7.  존재하지 않는 state → None
    8.  create_oauth_state 반환값에 tenant_id, provider, return_url 포함

  base / provider 구조:
    9.  GoogleGmailOAuth.get_authorize_url에 state 포함
    10. GoogleCalendarOAuth.get_authorize_url에 state 포함
    11. SlackOAuth.get_authorize_url에 state 포함
    12. JiraOAuth.get_authorize_url에 state 포함
    13. 각 provider에 provider_name 속성 설정 확인
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_oauth_state():
    from app.services.oauth.state import clear_oauth_states
    from app.services.oauth.token_crypto import reset_fernet_cache
    clear_oauth_states()
    reset_fernet_cache()
    yield
    clear_oauth_states()
    reset_fernet_cache()


# ── 1. encrypt/decrypt 왕복 ───────────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip(monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())

    from app.services.oauth.token_crypto import encrypt_token, decrypt_token, reset_fernet_cache
    reset_fernet_cache()

    plaintext = "ya29.access_token_example_12345"
    ciphertext = encrypt_token(plaintext)
    assert ciphertext != plaintext
    assert decrypt_token(ciphertext) == plaintext


# ── 2. 잘못된 키로 복호화 → ValueError ───────────────────────────────────────

def test_decrypt_with_wrong_key_raises_value_error(monkeypatch):
    key1 = Fernet.generate_key()
    key2 = Fernet.generate_key()

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key1.decode())
    from app.services.oauth.token_crypto import encrypt_token, reset_fernet_cache
    reset_fernet_cache()
    ciphertext = encrypt_token("secret-token")

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key2.decode())
    reset_fernet_cache()

    from app.services.oauth.token_crypto import decrypt_token
    with pytest.raises(ValueError, match="복호화 실패"):
        decrypt_token(ciphertext)


# ── 3. TOKEN_ENCRYPTION_KEY 미설정 → RuntimeError ────────────────────────────

def test_encrypt_without_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)
    from app.services.oauth.token_crypto import encrypt_token, reset_fernet_cache
    reset_fernet_cache()

    with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
        encrypt_token("some-token")


# ── 4. reset_fernet_cache 후 새 키 적용 ──────────────────────────────────────

def test_reset_fernet_cache_applies_new_key(monkeypatch):
    key1 = Fernet.generate_key()
    key2 = Fernet.generate_key()

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key1.decode())
    from app.services.oauth.token_crypto import encrypt_token, decrypt_token, reset_fernet_cache
    reset_fernet_cache()
    enc1 = encrypt_token("token-a")

    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key2.decode())
    reset_fernet_cache()
    enc2 = encrypt_token("token-b")
    assert decrypt_token(enc2) == "token-b"

    # key2로는 enc1 복호화 불가
    with pytest.raises(ValueError):
        decrypt_token(enc1)


# ── 5. create/verify state 왕복 ───────────────────────────────────────────────

def test_create_and_verify_oauth_state():
    from app.services.oauth.state import create_oauth_state, verify_oauth_state

    state = create_oauth_state("tenant-1", "google_gmail", "https://example.com/return")
    entry = verify_oauth_state(state)

    assert entry is not None
    assert entry.tenant_id == "tenant-1"
    assert entry.provider == "google_gmail"
    assert entry.return_url == "https://example.com/return"


# ── 6. state 1회 소모 확인 ────────────────────────────────────────────────────

def test_verify_oauth_state_consumed_after_first_use():
    from app.services.oauth.state import create_oauth_state, verify_oauth_state

    state = create_oauth_state("tenant-2", "slack", "")
    assert verify_oauth_state(state) is not None
    assert verify_oauth_state(state) is None  # 2번째 → None


# ── 7. 없는 state → None ─────────────────────────────────────────────────────

def test_verify_nonexistent_state_returns_none():
    from app.services.oauth.state import verify_oauth_state

    assert verify_oauth_state("completely-fake-state-xyz") is None


# ── 8. state entry 필드 확인 ─────────────────────────────────────────────────

def test_oauth_state_entry_has_all_fields():
    from app.services.oauth.state import create_oauth_state, verify_oauth_state

    state = create_oauth_state("tenant-3", "jira", "https://app.example.com/done")
    entry = verify_oauth_state(state)
    assert entry is not None
    assert entry.tenant_id == "tenant-3"
    assert entry.provider == "jira"
    assert entry.return_url == "https://app.example.com/done"
    assert not entry.is_expired()


# ── 9. GoogleGmailOAuth.get_authorize_url ─────────────────────────────────────

def test_google_gmail_oauth_authorize_url_contains_state(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://app.example.com/callback")

    from app.services.oauth.google_oauth import GoogleGmailOAuth
    oauth = GoogleGmailOAuth()
    url = oauth.get_authorize_url("my-state-token")

    assert "my-state-token" in url
    assert "accounts.google.com" in url
    assert "gmail" in url


# ── 10. GoogleCalendarOAuth.get_authorize_url ─────────────────────────────────

def test_google_calendar_oauth_authorize_url_contains_state(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_REDIRECT_URI", "https://app.example.com/callback")

    from app.services.oauth.google_oauth import GoogleCalendarOAuth
    oauth = GoogleCalendarOAuth()
    url = oauth.get_authorize_url("calendar-state")

    assert "calendar-state" in url
    assert "calendar" in url


# ── 11. SlackOAuth.get_authorize_url ─────────────────────────────────────────

def test_slack_oauth_authorize_url_contains_state(monkeypatch):
    monkeypatch.setenv("SLACK_CLIENT_ID", "slack-client-id")
    monkeypatch.setenv("SLACK_REDIRECT_URI", "https://app.example.com/slack/callback")

    from app.services.oauth.slack_oauth import SlackOAuth
    oauth = SlackOAuth()
    url = oauth.get_authorize_url("slack-state")

    assert "slack-state" in url
    assert "slack.com" in url


# ── 12. JiraOAuth.get_authorize_url ──────────────────────────────────────────

def test_jira_oauth_authorize_url_contains_state(monkeypatch):
    monkeypatch.setenv("ATLASSIAN_CLIENT_ID", "atlassian-client-id")
    monkeypatch.setenv("ATLASSIAN_REDIRECT_URI", "https://app.example.com/jira/callback")

    from app.services.oauth.jira_oauth import JiraOAuth
    oauth = JiraOAuth()
    url = oauth.get_authorize_url("jira-state")

    assert "jira-state" in url
    assert "atlassian.com" in url


# ── 13. provider_name 속성 확인 ──────────────────────────────────────────────

def test_oauth_provider_names():
    from app.services.oauth.google_oauth import GoogleGmailOAuth, GoogleCalendarOAuth
    from app.services.oauth.slack_oauth import SlackOAuth
    from app.services.oauth.jira_oauth import JiraOAuth

    assert GoogleGmailOAuth.provider_name == "google_gmail"
    assert GoogleCalendarOAuth.provider_name == "google_calendar"
    assert SlackOAuth.provider_name == "slack"
    assert JiraOAuth.provider_name == "jira"

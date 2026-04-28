"""
OAuth CSRF state 관리 — in-memory TTL 10분.

state 값은 UUID4 + tenant_id + provider 조합으로 위조 불가.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

_TTL_SECONDS = 600  # 10분


@dataclass
class _StateEntry:
    tenant_id: str
    provider: str
    return_url: str
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > _TTL_SECONDS


_state_store: dict[str, _StateEntry] = {}


def create_oauth_state(tenant_id: str, provider: str, return_url: str = "") -> str:
    """새 CSRF state 토큰을 생성·저장하고 반환한다."""
    _purge_expired()
    state = secrets.token_urlsafe(32)
    _state_store[state] = _StateEntry(
        tenant_id=tenant_id,
        provider=provider,
        return_url=return_url,
    )
    return state


def verify_oauth_state(state: str) -> _StateEntry | None:
    """state 검증 후 entry 반환. 없거나 만료된 경우 None. 검증 성공 시 자동 삭제."""
    entry = _state_store.get(state)
    if entry is None:
        return None
    if entry.is_expired():
        _state_store.pop(state, None)
        return None
    _state_store.pop(state, None)
    return entry


def clear_oauth_states() -> None:
    """테스트 격리용 — 모든 state 초기화."""
    _state_store.clear()


def _purge_expired() -> None:
    expired = [k for k, v in _state_store.items() if v.is_expired()]
    for k in expired:
        del _state_store[k]

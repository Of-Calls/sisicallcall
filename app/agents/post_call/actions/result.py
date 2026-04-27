from __future__ import annotations

_VALID_STATUSES = frozenset({"success", "failed", "skipped"})


def _make(
    action: dict,
    *,
    status: str,
    external_id: str | None = None,
    error: str | None = None,
    result: dict | None = None,
) -> dict:
    assert status in _VALID_STATUSES, f"status must be one of {_VALID_STATUSES}, got {status!r}"
    return {
        **action,                               # params, priority 등 원본 필드 보존
        "action_type": action.get("action_type", ""),
        "tool": action.get("tool", ""),
        "status": status,
        "external_id": external_id,
        "error": error,
        "result": result if result is not None else {},
    }


def action_success(
    action: dict,
    *,
    external_id: str | None = None,
    result: dict | None = None,
) -> dict:
    return _make(action, status="success", external_id=external_id, result=result)


def action_failed(
    action: dict,
    *,
    error: str,
    result: dict | None = None,
) -> dict:
    return _make(action, status="failed", error=error, result=result)


def action_skipped(
    action: dict,
    *,
    reason: str,
    result: dict | None = None,
) -> dict:
    return _make(action, status="skipped", error=reason, result=result)

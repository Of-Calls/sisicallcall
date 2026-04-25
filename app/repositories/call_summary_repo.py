from __future__ import annotations
from app.utils.logger import get_logger

logger = get_logger(__name__)

_store: dict[str, dict] = {}

_SAMPLE_CONTEXT = {
    "metadata": {
        "call_id": "sample-001",
        "tenant_id": "default",
        "start_time": "2026-04-25T10:00:00Z",
        "end_time": "2026-04-25T10:03:00Z",
    },
    "transcripts": [
        {"role": "customer", "text": "요금제 변경하고 싶은데요."},
        {"role": "agent", "text": "네, 도와드리겠습니다. 어떤 요금제로 변경을 원하시나요?"},
        {"role": "customer", "text": "더 저렴한 걸로 바꾸고 싶어요."},
    ],
    "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
}


class CallSummaryRepository:
    async def get_call_context(self, call_id: str) -> dict:
        if call_id in _store:
            return _store[call_id]
        logger.debug("call_context not found call_id=%s — sample 반환", call_id)
        ctx = dict(_SAMPLE_CONTEXT)
        ctx["metadata"] = {**_SAMPLE_CONTEXT["metadata"], "call_id": call_id}
        return ctx

    async def save_summary(self, call_id: str, summary: dict) -> None:
        if call_id not in _store:
            _store[call_id] = {}
        _store[call_id]["summary"] = summary
        logger.debug("summary saved call_id=%s", call_id)

    async def seed(self, call_id: str, context: dict) -> None:
        """테스트·스크립트용 in-memory 시드."""
        _store[call_id] = context

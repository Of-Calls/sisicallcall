from __future__ import annotations

import copy
import json
import os

from app.services.llm.base import BaseLLMService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_RETRY_SUFFIX = (
    "\n\n[IMPORTANT] Your previous response could not be parsed as JSON. "
    "Respond with ONLY a valid JSON object. "
    "No markdown fences, no explanations, no extra text of any kind."
)


# ── 실제 LLM 래퍼 ─────────────────────────────────────────────────────────────

class PostCallLLMCaller:
    """BaseLLMService 래퍼 — JSON 응답 파싱 + 1회 재시도."""

    def __init__(self, provider: BaseLLMService) -> None:
        self._provider = provider

    async def call_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1024,
    ) -> dict:
        raw = await self._provider.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        result, ok = _try_parse(raw)
        if ok:
            return result

        logger.warning("JSON 파싱 실패 — 1회 재시도 raw_preview=%r", raw[:200])
        raw2 = await self._provider.generate(
            system_prompt=system_prompt + _RETRY_SUFFIX,
            user_message=user_message,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        result2, ok2 = _try_parse(raw2)
        if ok2:
            return result2

        raise ValueError(f"LLM JSON 파싱 2회 연속 실패. last_raw={raw2[:300]!r}")


# ── 로컬 Mock (POST_CALL_USE_REAL_LLM 미설정 시 기본) ────────────────────────

class MockLLMCaller:
    """OpenAI 없이도 동작하는 인-메모리 mock.

    POST_CALL_USE_REAL_LLM=true 가 아닐 때 팩토리 함수에서 반환된다.
    스키마는 schemas.py 의 SummaryResult / VOCResult / PriorityNodeResult 를 따른다.
    """

    _MOCK_SUMMARY = {
        "summary_short": "[MOCK] 상담 요약",
        "summary_detailed": "[MOCK] 고객이 서비스 문의를 했고 상담원이 안내 후 처리됨",
        "customer_intent": "서비스 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["문의", "안내"],
        "handoff_notes": None,
    }

    _MOCK_VOC = {
        "sentiment_result": {
            "sentiment": "neutral",
            "intensity": 0.2,
            "reason": "[MOCK] 특이사항 없음",
        },
        "intent_result": {
            "primary_category": "서비스 문의",
            "sub_categories": [],
            "is_repeat_topic": False,
            "faq_candidate": False,
        },
        "priority_result": {
            "priority": "low",
            "action_required": False,
            "suggested_action": None,
            "reason": "[MOCK] 일반 처리",
        },
    }

    _MOCK_PRIORITY = {
        "priority": "low",
        "tier": "low",
        "action_required": False,
        "suggested_action": None,
        "reason": "[MOCK] 일반 처리",
    }

    async def call_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1024,
    ) -> dict:
        if "summary_short" in system_prompt:
            return copy.deepcopy(self._MOCK_SUMMARY)
        if "sentiment_result" in system_prompt:
            return copy.deepcopy(self._MOCK_VOC)
        # priority prompt 에는 "tier" 와 "action_required" 가 모두 포함됨
        return copy.deepcopy(self._MOCK_PRIORITY)


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def _try_parse(text: str) -> tuple[dict, bool]:
    """마크다운 코드블록 제거 후 JSON 파싱 시도."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, True
        return {}, False
    except json.JSONDecodeError:
        return {}, False


def _use_real_llm() -> bool:
    """POST_CALL_USE_REAL_LLM=true 일 때만 실제 Provider 를 사용한다.

    함수 호출 시점에 env var 을 평가하므로 테스트에서 os.environ 을 변경해도 반영된다.
    """
    return os.environ.get("POST_CALL_USE_REAL_LLM", "").lower() == "true"


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
# 각 노드 모듈에서 lazy 초기화 시 호출.
# 테스트에서는 노드 모듈의 _caller 를 monkeypatch 로 직접 교체하므로
# 여기서 실제 Provider 를 import 하지 않아도 된다.

def make_summary_caller() -> PostCallLLMCaller | MockLLMCaller:
    if _use_real_llm():
        # pydantic_settings / openai 는 실제 사용 시점에만 import
        from app.services.llm.gpt4o import GPT4OService  # noqa: PLC0415
        logger.info("POST_CALL_USE_REAL_LLM=true — GPT4OService 사용 (summary)")
        return PostCallLLMCaller(GPT4OService())
    logger.debug("POST_CALL_USE_REAL_LLM 미설정 — MockLLMCaller 사용 (summary)")
    return MockLLMCaller()


def make_voc_caller() -> PostCallLLMCaller | MockLLMCaller:
    if _use_real_llm():
        from app.services.llm.gpt4o import GPT4OService  # noqa: PLC0415
        logger.info("POST_CALL_USE_REAL_LLM=true — GPT4OService 사용 (voc)")
        return PostCallLLMCaller(GPT4OService())
    logger.debug("POST_CALL_USE_REAL_LLM 미설정 — MockLLMCaller 사용 (voc)")
    return MockLLMCaller()


def make_priority_caller() -> PostCallLLMCaller | MockLLMCaller:
    if _use_real_llm():
        from app.services.llm.gpt4o_mini import GPT4OMiniService  # noqa: PLC0415
        logger.info("POST_CALL_USE_REAL_LLM=true — GPT4OMiniService 사용 (priority)")
        return PostCallLLMCaller(GPT4OMiniService())
    logger.debug("POST_CALL_USE_REAL_LLM 미설정 — MockLLMCaller 사용 (priority)")
    return MockLLMCaller()

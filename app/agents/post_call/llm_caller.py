from __future__ import annotations

import copy
import json
import os

from app.services.llm.base import BaseLLMService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_RETRY_SUFFIX = (
    "\n\n[IMPORTANT] Your previous response could not be parsed as JSON. "
    "Respond with ONLY a valid JSON object. "
    "No markdown fences, no explanations, no extra text of any kind."
)

_MODE_MOCK = "mock"
_MODE_REAL = "real"
_DEFAULT_MODEL = "gpt-4o-mini"


# ── 실제 LLM 래퍼 ─────────────────────────────────────────────────────────────

class PostCallLLMCaller:
    """BaseLLMService 래퍼 — JSON 응답 파싱 + 1회 재시도."""

    def __init__(
        self,
        provider: BaseLLMService,
        *,
        fallback: "MockLLMCaller | None" = None,
        purpose: str = "post_call",
    ) -> None:
        self._provider = provider
        self._fallback = fallback
        self._purpose = purpose

    async def call_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1024,
    ) -> dict:
        try:
            raw = await self._provider.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            result, ok = _try_parse(raw)
            if ok:
                return result

            logger.warning("post_call real LLM JSON parse failed; retrying raw_preview=%r", raw[:200])
            raw2 = await self._provider.generate(
                system_prompt=system_prompt + _RETRY_SUFFIX,
                user_message=user_message,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            result2, ok2 = _try_parse(raw2)
            if ok2:
                return result2

            raise ValueError(f"LLM JSON parse failed twice. last_raw={raw2[:300]!r}")
        except Exception as exc:
            if self._fallback is not None:
                logger.warning(
                    "post_call real LLM failed purpose=%s err=%s; falling back to mock",
                    self._purpose,
                    exc,
                )
                fallback_result = await self._fallback.call_json(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tokens,
                )
                if isinstance(fallback_result, dict):
                    fallback_result = copy.deepcopy(fallback_result)
                    fallback_result["_llm_fallback"] = True
                    fallback_result["_llm_fallback_reason"] = str(exc)
                return fallback_result
            raise


class PostCallOpenAIService(BaseLLMService):
    """Post-call scoped OpenAI chat provider with runtime model selection."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or get_post_call_llm_model()
        self._client = None

    def _api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY") or settings.openai_api_key

    def _get_client(self):
        if not self._api_key():
            raise RuntimeError("OPENAI_API_KEY is required for POST_CALL_LLM_MODE=real")
        if self._client is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            self._client = AsyncOpenAI(api_key=self._api_key())
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        response = await self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=min(temperature, 0.2),
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


# ── 로컬 Mock (POST_CALL_USE_REAL_LLM 미설정 시 기본) ────────────────────────

class MockLLMCaller:
    """OpenAI 없이도 동작하는 인-메모리 mock.

    POST_CALL_USE_REAL_LLM=true 가 아닐 때 팩토리 함수에서 반환된다.
    스키마는 schemas.py 의 SummaryResult / VOCResult / PriorityNodeResult 를 따른다.

    라우팅 우선순위 (system_prompt 마커 기준):
      1. ANALYSIS_COMBINED → 통합 분석 결과 (post_call_analysis_node)
      2. REVIEW_VERDICT    → 검토 결과 pass (review_node)
      3. summary_short     → 요약 결과 (legacy summary_node)
      4. sentiment_result  → VOC 결과 (legacy voc_analysis_node)
      5. else              → 우선순위 결과 (legacy priority_node)
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

    _MOCK_ANALYSIS = {
        "summary": {
            "summary_short": "[MOCK] 상담 요약",
            "summary_detailed": "[MOCK] 고객이 서비스 문의를 했고 상담원이 안내 후 처리됨",
            "customer_intent": "서비스 문의",
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "keywords": ["문의", "안내"],
            "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.2, "reason": "[MOCK] 특이사항 없음"},
            "intent_result": {
                "primary_category": "서비스 문의",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {"priority": "low", "action_required": False, "suggested_action": None, "reason": "[MOCK] 일반 처리"},
        },
        "priority_result": {"priority": "low", "tier": "low", "action_required": False, "suggested_action": None, "reason": "[MOCK] 일반 처리"},
    }

    _MOCK_REVIEW_PASS = {
        "verdict": "pass",
        "confidence": 0.95,
        "issues": [],
        "corrections": {"summary": {}, "voc_analysis": {}, "priority_result": {}},
        "blocked_actions": [],
        "reason": "[MOCK] Analysis validated.",
    }

    async def call_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1024,
    ) -> dict:
        # 우선순위 1: 통합 분석 노드
        if "ANALYSIS_COMBINED" in system_prompt:
            return copy.deepcopy(self._MOCK_ANALYSIS)
        # 우선순위 2: 검토 게이트 노드
        if "REVIEW_VERDICT" in system_prompt:
            return copy.deepcopy(self._MOCK_REVIEW_PASS)
        # 하위 호환 — legacy 노드
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


def get_post_call_llm_mode() -> str:
    """Resolve post-call LLM mode.

    POST_CALL_LLM_MODE is the preferred switch. The legacy
    POST_CALL_USE_REAL_LLM=true flag is still honored when the new mode is not
    set, so existing local scripts keep working.
    """
    raw = os.environ.get("POST_CALL_LLM_MODE")
    if raw is not None:
        mode = raw.strip().lower()
        if mode in {_MODE_MOCK, _MODE_REAL}:
            return mode
        logger.warning("unknown POST_CALL_LLM_MODE=%s; falling back to mock", raw)
        return _MODE_MOCK

    legacy = os.environ.get("POST_CALL_USE_REAL_LLM", "").strip().lower()
    if legacy in {"1", "true", "yes", "on"}:
        return _MODE_REAL
    return _MODE_MOCK


def _use_real_llm() -> bool:
    return get_post_call_llm_mode() == _MODE_REAL


def get_post_call_llm_model() -> str:
    return os.environ.get("POST_CALL_LLM_MODEL", "").strip() or _DEFAULT_MODEL


def describe_post_call_llm() -> str:
    if _use_real_llm():
        return f"OpenAI Real LLM ({get_post_call_llm_model()})"
    return "Demo Mock LLM"


def post_call_openai_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or settings.openai_api_key)


def _make_caller(purpose: str) -> PostCallLLMCaller | MockLLMCaller:
    if _use_real_llm():
        model = get_post_call_llm_model()
        logger.info("POST_CALL_LLM_MODE=real - OpenAI model=%s purpose=%s", model, purpose)
        return PostCallLLMCaller(
            PostCallOpenAIService(model=model),
            fallback=MockLLMCaller(),
            purpose=purpose,
        )

    logger.debug("POST_CALL_LLM_MODE=mock - MockLLMCaller purpose=%s", purpose)
    return MockLLMCaller()


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
# 각 노드 모듈에서 lazy 초기화 시 호출.
# 테스트에서는 노드 모듈의 _caller 를 monkeypatch 로 직접 교체하므로
# 여기서 실제 Provider 를 import 하지 않아도 된다.

def make_summary_caller() -> PostCallLLMCaller | MockLLMCaller:
    return _make_caller("summary")


def make_voc_caller() -> PostCallLLMCaller | MockLLMCaller:
    return _make_caller("voc")


def make_priority_caller() -> PostCallLLMCaller | MockLLMCaller:
    return _make_caller("priority")


def make_analysis_caller() -> PostCallLLMCaller | MockLLMCaller:
    return _make_caller("analysis")


def make_review_caller() -> PostCallLLMCaller | MockLLMCaller:
    return _make_caller("review")

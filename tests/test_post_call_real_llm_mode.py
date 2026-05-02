from __future__ import annotations

import pytest

import app.agents.post_call.llm_caller as llm_mod


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch):
    monkeypatch.delenv("POST_CALL_LLM_MODE", raising=False)
    monkeypatch.delenv("POST_CALL_USE_REAL_LLM", raising=False)
    monkeypatch.delenv("POST_CALL_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_mod.settings, "openai_api_key", "")


def test_default_post_call_llm_mode_is_mock():
    assert llm_mod.get_post_call_llm_mode() == "mock"
    assert llm_mod.describe_post_call_llm() == "Demo Mock LLM"


def test_post_call_llm_mode_real_selects_real_description(monkeypatch):
    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setenv("POST_CALL_LLM_MODEL", "gpt-4o-mini")

    assert llm_mod.get_post_call_llm_mode() == "real"
    assert llm_mod.describe_post_call_llm() == "OpenAI Real LLM (gpt-4o-mini)"


def test_legacy_post_call_use_real_llm_still_selects_real(monkeypatch):
    monkeypatch.setenv("POST_CALL_USE_REAL_LLM", "true")

    assert llm_mod.get_post_call_llm_mode() == "real"


def test_cli_llm_mode_overrides_environment(monkeypatch):
    import scripts.run_post_call_from_db as db_runner

    monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")

    assert db_runner._apply_llm_mode("real") == "real"
    assert llm_mod.get_post_call_llm_mode() == "real"
    assert db_runner._apply_llm_mode("mock") == "mock"
    assert llm_mod.get_post_call_llm_mode() == "mock"


@pytest.mark.asyncio
async def test_real_llm_json_parsing_success(monkeypatch):
    class FakeOpenAIService:
        def __init__(self, model=None):
            self.model = model

        async def generate(self, **kwargs):
            return """
            {
              "summary": {
                "summary_short": "Reservation change request",
                "summary_detailed": "Customer asked to change a reservation.",
                "customer_intent": "Change reservation",
                "customer_emotion": "neutral",
                "resolution_status": "resolved",
                "keywords": ["reservation", "change"],
                "handoff_notes": null
              },
              "voc_analysis": {
                "sentiment_result": {"sentiment": "neutral", "intensity": 0.3, "reason": "informational"},
                "intent_result": {"primary_category": "예약/일정", "sub_categories": ["예약 변경"], "is_repeat_topic": false, "faq_candidate": false},
                "priority_result": {"priority": "medium", "action_required": false, "suggested_action": null, "reason": "ordinary request"}
              },
              "priority_result": {"priority": "medium", "tier": "medium", "action_required": false, "suggested_action": null, "reason": "ordinary request"}
            }
            """

    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setattr(llm_mod, "PostCallOpenAIService", FakeOpenAIService)

    caller = llm_mod.make_analysis_caller()
    result = await caller.call_json("ANALYSIS_COMBINED", "transcript")

    assert result["summary"]["summary_short"] == "Reservation change request"
    assert result["voc_analysis"]["intent_result"]["primary_category"] == "예약/일정"


@pytest.mark.asyncio
async def test_real_llm_parse_failure_falls_back_to_mock(monkeypatch):
    class BadOpenAIService:
        def __init__(self, model=None):
            self.model = model

        async def generate(self, **kwargs):
            return "not-json"

    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setattr(llm_mod, "PostCallOpenAIService", BadOpenAIService)

    caller = llm_mod.make_analysis_caller()
    result = await caller.call_json("ANALYSIS_COMBINED", "transcript")

    assert "summary" in result
    assert "voc_analysis" in result
    assert result["priority_result"]["priority"] == "low"
    assert result["_llm_fallback"] is True
    assert "LLM JSON parse failed" in result["_llm_fallback_reason"]


@pytest.mark.asyncio
async def test_real_llm_missing_openai_api_key_is_clear():
    service = llm_mod.PostCallOpenAIService(model="gpt-4o-mini")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await service.generate(
            system_prompt="system",
            user_message="user",
            max_tokens=10,
        )

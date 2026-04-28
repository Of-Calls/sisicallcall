"""시연용 Post-call 실행 스크립트.

demo-call-critical 컨텍스트를 시드한 뒤 PostCallAgent를 실행해 MCP 액션 결과를 출력한다.
기본값은 demo LLM mock (OpenAI 불필요).
POST_CALL_USE_REAL_LLM=true 환경변수를 설정하면 실제 LLM을 사용한다.

사용 예:
    python scripts/run_post_call_demo.py
    python scripts/run_post_call_demo.py --real-actions
    python scripts/run_post_call_demo.py --tenant-id my-tenant --call-id my-call-001
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.demo_post_call_context import (  # noqa: E402
    DEMO_POST_CALL_CONTEXT,
    DEMO_LLM_SUMMARY,
    DEMO_LLM_VOC,
    DEMO_LLM_PRIORITY,
)
from app.agents.post_call.context_provider import seed_test_context  # noqa: E402
from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call  # noqa: E402

# ── ANSI 색상 ─────────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _status_color(status: str) -> str:
    mapping = {"success": _GREEN, "failed": _RED, "skipped": _YELLOW}
    color = mapping.get(status, "")
    return _c(color + _BOLD, status.upper())


# ── 출력 헬퍼 ─────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{_c(_BOLD, '─' * 60)}")
    print(_c(_CYAN + _BOLD, f"  {title}"))
    print(_c(_BOLD, "─" * 60))


def _print_result(result: dict) -> None:
    _print_section("Post-call 분석 결과")

    summary = result.get("summary") or {}
    print(f"  summary_short    : {summary.get('summary_short', '—')}")
    print(f"  customer_emotion : {_c(_BOLD, str(summary.get('customer_emotion', '—')))}")
    print(f"  resolution_status: {summary.get('resolution_status', '—')}")

    priority = result.get("priority_result") or {}
    print(f"  priority         : {_c(_BOLD, str(priority.get('priority', '—')))}")
    print(f"  action_required  : {priority.get('action_required', False)}")

    _print_section("Action Plan")
    plan = result.get("action_plan") or {}
    actions = plan.get("actions") or []
    if actions:
        for a in actions:
            print(f"    · {a.get('action_type'):35s} tool={a.get('tool')}")
    else:
        print("    (계획된 액션 없음)")

    _print_section("실행된 액션")
    executed = result.get("executed_actions") or []
    if executed:
        for a in executed:
            status_str = _status_color(a.get("status", "unknown"))
            ext_id = a.get("external_id") or "—"
            err    = a.get("error") or ""
            line = (
                f"    [{status_str}] "
                f"{a.get('action_type', '?'):35s} "
                f"tool={a.get('tool', '?'):20s} "
                f"external_id={ext_id}"
            )
            if err:
                line += f"  {_c(_RED, 'err=' + err)}"
            print(line)
    else:
        print("    (실행된 액션 없음)")

    errors = result.get("errors") or []
    partial = result.get("partial_success", False)

    if errors:
        _print_section("오류")
        for e in errors:
            print(f"    · {_c(_RED, str(e))}")

    _print_section("요약")
    failed_cnt  = sum(1 for a in executed if a.get("status") == "failed")
    skipped_cnt = sum(1 for a in executed if a.get("status") == "skipped")
    success_cnt = sum(1 for a in executed if a.get("status") == "success")
    print(f"  partial_success : {partial}")
    print(
        f"  액션 결과       : "
        f"{_c(_GREEN, str(success_cnt) + ' success')}  "
        f"{_c(_YELLOW, str(skipped_cnt) + ' skipped')}  "
        f"{_c(_RED, str(failed_cnt) + ' failed')}"
    )


# ── Demo LLM — DEMO_LLM_* 픽스처를 반환하는 stub ────────────────────────────

class _DemoLLM:
    async def call_json(self, system_prompt: str, user_message: str, max_tokens: int = 1024) -> dict:
        if "summary_short" in system_prompt:
            return DEMO_LLM_SUMMARY
        if "sentiment_result" in system_prompt:
            return DEMO_LLM_VOC
        return DEMO_LLM_PRIORITY


def _patch_llm_nodes() -> None:
    """LLM 노드의 _caller를 demo stub으로 교체한다."""
    import app.agents.post_call.nodes.summary_node as _summary
    import app.agents.post_call.nodes.voc_analysis_node as _voc
    import app.agents.post_call.nodes.priority_node as _priority

    _demo = _DemoLLM()
    _summary._caller  = _demo  # type: ignore[attr-defined]
    _voc._caller      = _demo  # type: ignore[attr-defined]
    _priority._caller = _demo  # type: ignore[attr-defined]


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

async def _run(tenant_id: str, call_id: str, real_actions: bool) -> None:
    ctx = DEMO_POST_CALL_CONTEXT

    print(_c(_BOLD, "\n시시콜콜 Post-call MCP 시연 스크립트"))
    print(f"  call_id   : {call_id}")
    print(f"  tenant_id : {tenant_id}")

    if real_actions:
        os.environ["SMS_MCP_REAL"]    = "true"
        os.environ["NOTION_MCP_REAL"] = "true"
        os.environ["SLACK_MCP_REAL"]  = "true"
        print(f"  mode      : {_c(_GREEN + _BOLD, 'REAL 모드')} (SMS / Notion / Slack 실제 호출)")
    else:
        os.environ.setdefault("SMS_MCP_REAL",    "false")
        os.environ.setdefault("NOTION_MCP_REAL", "false")
        os.environ.setdefault("SLACK_MCP_REAL",  "false")
        print(
            f"  mode      : {_c(_YELLOW + _BOLD, 'MOCK 모드')} "
            f"(외부 API 미호출 — --real-actions 플래그로 실제 실행)"
        )

    use_real_llm = os.environ.get("POST_CALL_USE_REAL_LLM", "").lower() == "true"
    if use_real_llm:
        print(f"  LLM       : {_c(_GREEN, '실제 LLM (POST_CALL_USE_REAL_LLM=true)')}")
    else:
        print(f"  LLM       : {_c(_YELLOW, 'Demo Mock LLM (angry/critical 시나리오 고정)')}")
        _patch_llm_nodes()

    # 시연 컨텍스트 seed
    await seed_test_context(
        call_id=call_id,
        tenant_id=tenant_id,
        transcripts=ctx.get("transcripts"),
        call_metadata={**ctx.get("metadata", {}), "call_id": call_id, "tenant_id": tenant_id},
        branch_stats=ctx.get("branch_stats"),
    )

    print("\n  컨텍스트 시드 완료 — PostCallAgent 실행 중...")

    outcome = await run_post_call_for_completed_call(
        call_id=call_id,
        tenant_id=tenant_id,
        trigger="call_ended",
    )

    if not outcome.get("ok"):
        print(_c(_RED + _BOLD, f"\n[오류] PostCallAgent 실행 실패: {outcome.get('error')}"))
        sys.exit(1)

    _print_result(outcome["result"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="시연용 Post-call MCP 실행 스크립트 (기본: Mock 모드)",
    )
    parser.add_argument(
        "--tenant-id",
        default="demo-tenant",
        help="테넌트 ID (기본값: demo-tenant)",
    )
    parser.add_argument(
        "--call-id",
        default="demo-call-critical",
        help="통화 ID (기본값: demo-call-critical)",
    )
    parser.add_argument(
        "--real-actions",
        action="store_true",
        help="실제 SMS / Notion / Slack 호출 활성화 (기본: mock)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.tenant_id, args.call_id, args.real_actions))


if __name__ == "__main__":
    main()

"""PostCallAgent 수동 실행 스크립트.

기본 동작은 Mock LLM 사용 (OpenAI 불필요).
POST_CALL_USE_REAL_LLM=true 환경변수 설정 시 실제 GPT 사용.

사용 예:
    python scripts/run_post_call_agent.py --trigger call_ended
    python scripts/run_post_call_agent.py --trigger escalation_immediate
    python scripts/run_post_call_agent.py --trigger manual --call-id my-call-001
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# 프로젝트 루트를 sys.path 에 추가 (scripts/ 안에서 실행 시)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.post_call.agent import PostCallAgent  # noqa: E402

_VALID_TRIGGERS = ("call_ended", "manual", "escalation_immediate")


async def _run(trigger: str, call_id: str) -> None:
    agent = PostCallAgent()

    use_real = os.environ.get("POST_CALL_USE_REAL_LLM", "").lower() == "true"
    llm_mode = "실제 LLM (POST_CALL_USE_REAL_LLM=true)" if use_real else "Mock LLM"
    print(f"[run_post_call_agent] trigger={trigger!r}  call_id={call_id!r}  LLM={llm_mode}")

    result = await agent.run(call_id=call_id, trigger=trigger, tenant_id="demo")

    partial = result.get("partial_success", False)
    errors = result.get("errors", [])
    print(f"[run_post_call_agent] 완료  partial_success={partial}  errors={len(errors)}")

    if result.get("summary"):
        s = result["summary"]
        print(f"  summary_short  : {s.get('summary_short')}")
        print(f"  emotion        : {s.get('customer_emotion')}  "
              f"resolution: {s.get('resolution_status')}")

    if result.get("priority_result"):
        p = result["priority_result"]
        print(f"  priority       : {p.get('priority')}  action_required={p.get('action_required')}")

    if result.get("action_plan"):
        plan = result["action_plan"]
        print(f"  action_plan    : {len(plan.get('actions', []))}개 액션  "
              f"action_required={plan.get('action_required')}")
        for a in plan.get("actions", []):
            print(f"    · {a.get('action_type')} ({a.get('tool')})")

    if result.get("executed_actions"):
        executed = result["executed_actions"]
        failed_cnt = sum(1 for a in executed if a.get("status") == "failed")
        print(f"  executed       : {len(executed)}개 실행  failed={failed_cnt}")

    if errors:
        print("  [ERRORS]")
        for e in errors:
            print(f"    · {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PostCallAgent 수동 실행 (기본: Mock LLM)")
    parser.add_argument(
        "--trigger",
        choices=list(_VALID_TRIGGERS),
        default="call_ended",
        help="실행 트리거 (기본값: call_ended)",
    )
    parser.add_argument(
        "--call-id",
        default="test-001",
        help="테스트용 call_id (기본값: test-001)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.trigger, args.call_id))


if __name__ == "__main__":
    main()

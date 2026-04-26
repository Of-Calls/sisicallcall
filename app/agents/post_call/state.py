from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


class PostCallAgentState(TypedDict):
    call_id: str
    tenant_id: str
    trigger: str                    # "call_ended" | "escalation_immediate" | "manual"
    call_metadata: dict
    transcripts: list[dict]
    branch_stats: dict
    summary: Optional[dict]
    voc_analysis: Optional[dict]
    priority_result: Optional[dict]
    action_plan: Optional[dict]
    executed_actions: list[dict]
    dashboard_payload: Optional[dict]
    errors: list[dict]
    partial_success: bool

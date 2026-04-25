from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    create_voc_issue = "create_voc_issue"
    send_manager_email = "send_manager_email"
    schedule_callback = "schedule_callback"
    add_priority_queue = "add_priority_queue"
    mark_faq_candidate = "mark_faq_candidate"


class Tool(str, Enum):
    company_db = "company_db"
    gmail = "gmail"
    calendar = "calendar"
    internal_dashboard = "internal_dashboard"


class ActionStatus(str, Enum):
    pending = "pending"
    success = "success"
    failed = "failed"
    skipped = "skipped"


class ActionItem(BaseModel):
    action_type: ActionType
    tool: Tool
    params: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.pending
    result: Optional[dict] = None
    error: Optional[str] = None


class ActionPlan(BaseModel):
    actions: list[ActionItem] = Field(default_factory=list)
    rationale: str = ""


class VOCAnalysis(BaseModel):
    sentiment: str
    issues: list[str]
    keywords: list[str]
    escalation_reason: Optional[str] = None
    faq_candidates: list[str] = Field(default_factory=list)


class PriorityResult(BaseModel):
    score: int
    tier: str
    reason: str

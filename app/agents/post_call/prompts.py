SUMMARY_PROMPT = """\
You are a call-center quality analyst.
Given the transcripts below, produce a concise call summary in JSON format:
{{
  "summary_text": "...",
  "call_duration_sec": <int>,
  "customer_intent": "...",
  "resolution_status": "resolved" | "unresolved" | "escalated",
  "key_topics": ["...", ...]
}}

Transcripts:
{transcripts}
"""

VOC_ANALYSIS_PROMPT = """\
Analyze the following call summary and transcripts for VOC (Voice of Customer):
{summary}

Transcripts:
{transcripts}

Return JSON:
{{
  "sentiment": "positive" | "neutral" | "negative",
  "issues": ["..."],
  "keywords": ["..."],
  "escalation_reason": "..." | null,
  "faq_candidates": ["..."]
}}
"""

PRIORITY_PROMPT = """\
Given the VOC analysis below, assign a priority score (1=lowest, 5=highest):
{voc_analysis}

Return JSON:
{{
  "score": <int 1-5>,
  "tier": "critical" | "high" | "medium" | "low",
  "reason": "..."
}}
"""

ACTION_PLAN_PROMPT = """\
Based on the VOC analysis and priority result below, decide which actions to take:

VOC: {voc_analysis}
Priority: {priority_result}

Available actions: create_voc_issue, send_manager_email, schedule_callback, add_priority_queue, mark_faq_candidate
Available tools: company_db, gmail, calendar, internal_dashboard

Return JSON:
{{
  "actions": [
    {{"action_type": "...", "tool": "...", "params": {{...}}}},
    ...
  ],
  "rationale": "..."
}}
"""

"""
Notion MCP Connector — Internal Integration Token 방식.

시연용 Internal Integration Token + Database ID 방식을 사용한다.
OAuth 방식은 이번 범위에서 제외한다.

지원 action_type:
  - create_notion_call_record   통화 요약/결과 저장
  - create_notion_voc_record    VOC/에스컬레이션 기록 저장

── real mode (NOTION_MCP_REAL=true) ─────────────────────────────────────────
  POST https://api.notion.com/v1/pages
  Authorization: Bearer {NOTION_API_TOKEN}
  Notion-Version: 2022-06-28
  parent.database_id = NOTION_DATABASE_ID

── mock mode (NOTION_MCP_REAL=false, 기본) ──────────────────────────────────
  status: success
  external_id: notion-mock-{call_id}
  result: {page_id, mock}

── 설정 누락 ─────────────────────────────────────────────────────────────────
  NOTION_API_TOKEN 또는 NOTION_DATABASE_ID 없으면 skipped("notion_not_configured")

── Notion DB 필드 ────────────────────────────────────────────────────────────
  Name, Call ID, Tenant ID, Customer Emotion, Priority,
  Resolution Status, Summary, VOC Category, Action Required, Created At

── token 원문은 로그/result/error에 절대 출력하지 않는다. ────────────────────
"""
from __future__ import annotations

import os
from datetime import datetime

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)

_NOTION_PAGES_URL = "https://api.notion.com/v1/pages"
_NOTION_VERSION = "2022-06-28"


class NotionConnector(BaseMCPConnector):
    connector_name = "notion"
    _real_mode_env = "NOTION_MCP_REAL"
    _required_config = ("NOTION_API_TOKEN", "NOTION_DATABASE_ID")
    _oauth_provider_name = ""

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "NotionConnector call_id=%s action_type=%s real_mode=%s",
            call_id, action_type, self.is_real_mode(),
        )

        if not self.is_real_mode():
            return self._mock(params, call_id)

        api_token = os.getenv("NOTION_API_TOKEN")
        db_id = os.getenv("NOTION_DATABASE_ID")
        if not api_token or not db_id:
            logger.warning("NotionConnector: 설정 누락 call_id=%s", call_id)
            return self._skipped("notion_not_configured")

        return await self._create_page(api_token, db_id, action_type, params, call_id=call_id)

    def _build_properties(self, action_type: str, params: dict, call_id: str) -> dict:
        call_id_val = params.get("call_id", call_id)
        tenant_id = params.get("tenant_id", "")
        emotion = params.get("customer_emotion") or params.get("emotion") or ""
        priority = params.get("priority") or ""
        resolution = params.get("resolution_status") or ""
        summary = (params.get("summary_short") or params.get("summary") or "")[:2000]
        voc_cat = params.get("primary_category") or params.get("voc_category") or ""
        action_req = bool(params.get("action_required", False))
        name = f"[{action_type.replace('_', '-')}] {call_id_val}"

        props: dict = {
            "Name": {"title": [{"text": {"content": name}}]},
            "Call ID": {"rich_text": [{"text": {"content": call_id_val}}]},
            "Tenant ID": {"rich_text": [{"text": {"content": tenant_id}}]},
            "Summary": {"rich_text": [{"text": {"content": summary}}]},
            "VOC Category": {"rich_text": [{"text": {"content": voc_cat}}]},
            "Action Required": {"checkbox": action_req},
            "Created At": {"date": {"start": datetime.utcnow().isoformat()}},
        }
        if emotion:
            props["Customer Emotion"] = {"select": {"name": emotion}}
        if priority:
            props["Priority"] = {"select": {"name": priority}}
        if resolution:
            props["Resolution Status"] = {"select": {"name": resolution}}
        return props

    async def _create_page(
        self,
        api_token: str,
        db_id: str,
        action_type: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        """Notion Pages API를 호출한다. api_token은 로그에 출력하지 않는다."""
        import httpx

        payload = {
            "parent": {"database_id": db_id},
            "properties": self._build_properties(action_type, params, call_id),
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    _NOTION_PAGES_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_token}",
                        "Notion-Version": _NOTION_VERSION,
                        "Content-Type": "application/json",
                    },
                    timeout=15.0,
                )

            if resp.status_code not in (200, 201):
                preview = resp.text[:200]
                logger.error(
                    "NotionConnector: API 오류 call_id=%s status=%d preview=%s",
                    call_id, resp.status_code, preview,
                )
                return self._failed(f"notion_api_error:{resp.status_code}")

            data = resp.json()
            page_id = data.get("id")
            logger.info(
                "NotionConnector: 페이지 생성 완료 call_id=%s page_id=%s",
                call_id, page_id,
            )
            return self._success(
                external_id=page_id,
                result={"page_id": page_id, "url": data.get("url")},
            )

        except Exception as exc:
            logger.error(
                "NotionConnector: 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"notion_exception:{type(exc).__name__}")

    def _mock(self, params: dict, call_id: str) -> dict:
        return self._success(
            external_id=f"notion-mock-{call_id}",
            result={"page_id": f"notion-mock-{call_id}", "url": None, "mock": True},
        )

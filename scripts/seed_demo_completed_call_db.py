"""Seed one completed demo call into Postgres for DB-backed post-call checks.

This script is intentionally idempotent:
- the tenant row is upserted from the provided tenant label
- the calls row is upserted by twilio_call_sid
- transcripts for that call are deleted and re-inserted

Run manually:
    python scripts/seed_demo_completed_call_db.py
    python scripts/seed_demo_completed_call_db.py --call-id demo-db-call-critical --tenant-id demo-tenant
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import settings  # noqa: E402
from tests.fixtures.demo_post_call_context import DEMO_POST_CALL_CONTEXT  # noqa: E402

DEFAULT_CALL_ID = "demo-db-call-critical"
DEFAULT_TENANT_ID = "demo-tenant"
DEFAULT_CALLER_NUMBER = "+821049460829"
DEFAULT_BRANCH_STATS = {"faq": 1, "task": 2, "escalation": 1}


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _tenant_uuid(tenant_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(tenant_id)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"sisicallcall-demo-tenant:{tenant_id}")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _demo_transcripts() -> list[dict]:
    return list(DEMO_POST_CALL_CONTEXT.get("transcripts") or [])


async def _upsert_tenant(
    conn: asyncpg.Connection,
    *,
    tenant_id: str,
    tenant_uuid: uuid.UUID,
) -> uuid.UUID:
    row = await conn.fetchrow(
        """
        INSERT INTO tenants (id, name, twilio_number, industry, plan, settings)
        VALUES ($1::uuid, $2, $3, 'finance', 'basic', $4::jsonb)
        ON CONFLICT (twilio_number) DO UPDATE
            SET name = EXCLUDED.name,
                industry = EXCLUDED.industry,
                plan = EXCLUDED.plan,
                settings = EXCLUDED.settings,
                updated_at = now()
        RETURNING id
        """,
        str(tenant_uuid),
        f"Demo tenant ({tenant_id})",
        DEFAULT_CALLER_NUMBER,
        json.dumps({"external_tenant_id": tenant_id, "seed": "demo_completed_call"}),
    )
    return row["id"]


async def _upsert_call(
    conn: asyncpg.Connection,
    *,
    db_tenant_id: uuid.UUID,
    call_id: str,
    caller_number: str,
) -> uuid.UUID:
    started_at = _parse_timestamp(DEMO_POST_CALL_CONTEXT["metadata"].get("start_time"))
    ended_at = _parse_timestamp(DEMO_POST_CALL_CONTEXT["metadata"].get("end_time"))

    row = await conn.fetchrow(
        """
        INSERT INTO calls (
            tenant_id, twilio_call_sid, caller_number, status,
            started_at, ended_at, duration_sec, branch_stats
        )
        VALUES ($1::uuid, $2, $3, 'completed', $4, $5, 840, $6::jsonb)
        ON CONFLICT (twilio_call_sid) DO UPDATE
            SET tenant_id = EXCLUDED.tenant_id,
                caller_number = EXCLUDED.caller_number,
                status = EXCLUDED.status,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                duration_sec = EXCLUDED.duration_sec,
                branch_stats = EXCLUDED.branch_stats
        RETURNING id
        """,
        str(db_tenant_id),
        call_id,
        caller_number,
        started_at,
        ended_at,
        json.dumps(DEFAULT_BRANCH_STATS),
    )
    return row["id"]


async def _replace_transcripts(
    conn: asyncpg.Connection,
    *,
    db_call_id: uuid.UUID,
    transcripts: list[dict],
) -> int:
    await conn.execute("DELETE FROM transcripts WHERE call_id = $1::uuid", str(db_call_id))

    for index, item in enumerate(transcripts):
        await conn.execute(
            """
            INSERT INTO transcripts (call_id, turn_index, speaker, text, spoken_at)
            VALUES ($1::uuid, $2, $3, $4, $5)
            """,
            str(db_call_id),
            index,
            item.get("role") or "customer",
            item.get("text") or "",
            _parse_timestamp(item.get("timestamp")),
        )

    return len(transcripts)


async def seed_demo_completed_call(
    *,
    call_id: str,
    tenant_id: str,
    caller_number: str,
) -> None:
    conn = await asyncpg.connect(_database_url())
    try:
        async with conn.transaction():
            db_tenant_id = await _upsert_tenant(
                conn,
                tenant_id=tenant_id,
                tenant_uuid=_tenant_uuid(tenant_id),
            )
            db_call_id = await _upsert_call(
                conn,
                db_tenant_id=db_tenant_id,
                call_id=call_id,
                caller_number=caller_number,
            )
            transcript_count = await _replace_transcripts(
                conn,
                db_call_id=db_call_id,
                transcripts=_demo_transcripts(),
            )
    finally:
        await conn.close()

    print("Seeded demo completed call")
    print(f"  call_id          : {call_id}")
    print(f"  tenant_id        : {tenant_id}")
    print(f"  db_tenant_uuid   : {db_tenant_id}")
    print(f"  db_call_uuid     : {db_call_id}")
    print(f"  transcript_count : {transcript_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed DB rows for the post-call completed-call runner demo.",
    )
    parser.add_argument("--call-id", default=DEFAULT_CALL_ID)
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    parser.add_argument("--caller-number", default=DEFAULT_CALLER_NUMBER)
    args = parser.parse_args()

    asyncio.run(
        seed_demo_completed_call(
            call_id=args.call_id,
            tenant_id=args.tenant_id,
            caller_number=args.caller_number,
        )
    )


if __name__ == "__main__":
    main()

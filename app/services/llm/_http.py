"""OpenAI API 용 공유 httpx AsyncClient.

기본 httpx keepalive_expiry 가 5초라 startup 워밍 직후 idle 되면 connection 죽음.
시연 시나리오 (서버 켜고 수분 후 첫 통화) 에서 워밍 효과 보존하려면 keepalive 연장 필요.

module-level singleton — gpt4o / gpt4o_mini / post_call llm_caller 가 같은 pool 공유.
"""
import httpx

_http_client: httpx.AsyncClient | None = None


def get_openai_http_client() -> httpx.AsyncClient:
    """OpenAI 전용 httpx AsyncClient (singleton).

    keepalive_expiry=300s (5분) — startup 워밍 후 5분 idle 까지 connection 살아있음.
    timeout = httpx 기본보다 조금 넉넉 (read 60s — LLM 응답 여유).
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=300.0,
            ),
        )
    return _http_client

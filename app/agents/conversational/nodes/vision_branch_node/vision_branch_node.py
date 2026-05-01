from app.agents.conversational.state import CallState

_UPLOAD_PROMPT = "어떤 제품인지 정확히 알려드리려면 사진이 필요합니다. 사진을 찍어 업로드해주세요."


async def vision_branch_node(state: CallState) -> dict:
    print(f"[vision_branch] 진입 user_text='{state['user_text']}'")
    return {"response_text": _UPLOAD_PROMPT}

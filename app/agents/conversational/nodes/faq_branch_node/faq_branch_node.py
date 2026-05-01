from app.agents.conversational.state import CallState


async def faq_branch_node(state: CallState) -> dict:
    print(f"[faq_branch] 진입 user_text='{state['user_text']}'")
    return {"response_text": "[FAQ 분기 도달]"}

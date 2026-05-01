from app.agents.conversational.state import CallState


async def auth_branch_node(state: CallState) -> dict:
    print(f"[auth_branch] 진입 user_text='{state['user_text']}'")
    return {"response_text": "[Auth 분기 도달]"}

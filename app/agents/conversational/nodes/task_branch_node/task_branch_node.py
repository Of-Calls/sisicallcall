from app.agents.conversational.state import CallState


async def task_branch_node(state: CallState) -> dict:
    print(f"[task_branch] 진입 user_text='{state['user_text']}'")
    return {"response_text": "[Task 분기 도달]"}

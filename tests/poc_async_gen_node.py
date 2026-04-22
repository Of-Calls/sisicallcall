"""PoC — LangGraph 0.2.28 async generator node support verification.

RFC 001 의 γ 채택 전제: StateGraph 가 async gen 노드의 여러 yield 를 state 에
순차 병합하는지 확인. 만약 지원되지 않으면 γ 재설계 필요.

실행: venv/Scripts/python tests/poc_async_gen_node.py
"""
import asyncio
from typing import TypedDict
from langgraph.graph import END, StateGraph


class SimpleState(TypedDict, total=False):
    counter: int
    messages: list[str]
    final: str


# ---------------------------------------------------------------------------
# Test 1: async gen node with multiple yields — does graph merge them all?
# ---------------------------------------------------------------------------

async def multi_yield_node(state: SimpleState):
    """Async gen node emitting 3 partial updates."""
    print(f"  [multi_yield_node] entered, state.counter={state.get('counter', 0)}")
    yield {"messages": state.get("messages", []) + ["first"]}
    print("  [multi_yield_node] yielded 1st")
    await asyncio.sleep(0.01)
    yield {"messages": state.get("messages", []) + ["second"]}
    print("  [multi_yield_node] yielded 2nd")
    await asyncio.sleep(0.01)
    yield {"final": "done", "counter": state.get("counter", 0) + 1}
    print("  [multi_yield_node] yielded 3rd (final)")


def single_return_node(state: SimpleState) -> dict:
    """Baseline: regular sync node returning dict."""
    return {"counter": state.get("counter", 0) + 100}


async def test_1_multi_yield_merging():
    print("\n=== Test 1: async gen node with multiple yields ===")
    graph = StateGraph(SimpleState)
    graph.add_node("multi", multi_yield_node)
    graph.set_entry_point("multi")
    graph.add_edge("multi", END)
    app = graph.compile()

    result = await app.ainvoke({"counter": 0, "messages": []})
    print(f"  FINAL STATE: {result}")

    # 기대: 3번 yield 모두 반영 → messages 에 ["first","second"] 중 하나 또는 둘 다,
    # final="done", counter=1
    print(f"  messages = {result.get('messages')}")
    print(f"  final = {result.get('final')}")
    print(f"  counter = {result.get('counter')}")

    # 판정
    if result.get("final") == "done" and result.get("counter") == 1:
        if result.get("messages") == ["first", "second"]:
            print("  [PASS] LangGraph received all yields and accumulated (reducer-like)")
        elif result.get("messages") == ["second"]:
            print("  [PARTIAL] Only LAST messages yield survived (last-write-wins per key)")
        elif result.get("messages") == []:
            print("  [HOLE] messages not preserved - LangGraph only applied LAST yield entirely")
        else:
            print(f"  [UNEXPECTED] messages={result.get('messages')}")
    else:
        print(f"  [FAIL] Final yield not applied - async gen nodes NOT fully supported")


# ---------------------------------------------------------------------------
# Test 2: astream_events — can we observe yields as stream events?
# ---------------------------------------------------------------------------

async def test_2_astream_events():
    print("\n=== Test 2: astream_events observes intermediate yields ===")
    graph = StateGraph(SimpleState)
    graph.add_node("multi", multi_yield_node)
    graph.set_entry_point("multi")
    graph.add_edge("multi", END)
    app = graph.compile()

    events = []
    async for event in app.astream({"counter": 0, "messages": []}, stream_mode="updates"):
        events.append(event)
        print(f"  stream event: {event}")

    print(f"  TOTAL EVENTS: {len(events)}")
    if len(events) >= 3:
        print("  [PASS] astream yielded multiple updates - streaming works")
    else:
        print(f"  [NOTE] Only {len(events)} events received - yield granularity lower than expected")


# ---------------------------------------------------------------------------
# Test 3: Typed reducer — does total=False with list concatenate?
# ---------------------------------------------------------------------------

async def test_3_reducer_behavior():
    """TypedDict 에 reducer 를 쓰지 않으면 last-write-wins.
    stall + final 방출 시 'stall_text' 필드는 last yield 로 덮어씌워지므로
    graph runner 가 yield 각각을 따로 소비해야 함."""
    print("\n=== Test 3: TypedDict last-write-wins behavior ===")
    graph = StateGraph(SimpleState)
    graph.add_node("multi", multi_yield_node)
    graph.set_entry_point("multi")
    graph.add_edge("multi", END)
    app = graph.compile()

    # ainvoke 는 최종 state 만. messages 가 덮어쓰기 vs 누적 어느 동작?
    result = await app.ainvoke({"counter": 0, "messages": []})
    print(f"  ainvoke final state: {result}")
    if result.get("messages") == ["first"]:
        print("  → First yield 만 반영 (downstream yields overwrote themselves, last-write-wins 문제)")
    elif result.get("messages") == ["second"]:
        print("  → Last yield 만 반영 (first yield 덮어씀 — 'stall' 을 final 이 덮어쓸 것)")
    elif result.get("messages") == ["first", "second"]:
        print("  → 두 yield 모두 누적 (reducer 자동 동작?)")
    else:
        print(f"  → 예상 밖: {result.get('messages')}")


async def main():
    print("LangGraph version:")
    import langgraph
    print(f"  {getattr(langgraph, '__version__', '0.2.28 (no __version__ attr)')}")

    try:
        await test_1_multi_yield_merging()
    except Exception as e:
        print(f"  [EXCEPTION] Test 1 failed: {type(e).__name__}: {e}")

    try:
        await test_2_astream_events()
    except Exception as e:
        print(f"  [EXCEPTION] Test 2 failed: {type(e).__name__}: {e}")

    try:
        await test_3_reducer_behavior()
    except Exception as e:
        print(f"  [EXCEPTION] Test 3 failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

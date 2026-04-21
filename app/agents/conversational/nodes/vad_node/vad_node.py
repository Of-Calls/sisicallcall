from app.agents.conversational.state import CallState
from app.services.vad.benchmark import compare_vad_models


async def vad_node(state: CallState) -> dict:
    results = await compare_vad_models(state["audio_chunk"])
    speech_votes = sum(1 for result in results if result.is_speech)
    is_speech = speech_votes >= 2
    latency_by_model = dict(state.get("vad_latency_ms_by_model", {}))
    is_speech_by_model = dict(state.get("vad_is_speech_by_model", {}))
    true_count_by_model = dict(state.get("vad_true_count_by_model", {}))
    for result in results:
        latency_by_model.setdefault(result.model, []).append(result.latency_ms)
        is_speech_by_model[result.model] = result.is_speech
        if result.is_speech:
            true_count_by_model[result.model] = true_count_by_model.get(result.model, 0) + 1
    return {
        "is_speech": is_speech,
        "vad_latency_ms_by_model": latency_by_model,
        "vad_is_speech_by_model": is_speech_by_model,
        "vad_true_count_by_model": true_count_by_model,
    }
"""query_refine_node 헬퍼 단위 테스트 — 2026-04-30 ambiguity gate 보수 검증."""

from app.agents.conversational.nodes.query_refine_node.query_refine_node import (
    _is_ambiguous,
    _normalize_text,
)

# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


def test_normalize_repeated_fillers():
    """반복 필러 "어어" 제거 — tenant_keyterms 유무 무관하게 "대중교통" 반환."""
    # keyterms 없음: 필러만 제거
    result_no_kw = _normalize_text("어어대중교통", tenant_keyterms=[])
    assert result_no_kw == "대중교통"

    # keyterms 있음: 필러 제거 후 키텀이 텍스트 시작에 위치(pos==0) → 공백 삽입 조건 미충족
    result_with_kw = _normalize_text("어어대중교통", tenant_keyterms=["대중교통"])
    assert result_with_kw == "대중교통"


def test_normalize_preserves_eotteoke():
    """정상 단어 "어떻게가죠" — '어' 1회만 등장, \\1+ 미충족 → 보호."""
    result = _normalize_text("어떻게가죠", tenant_keyterms=[])
    assert result == "어떻게가죠"


def test_normalize_single_filler_with_space():
    """단발 필러 + 공백: "어 안녕하세요" → "안녕하세요"."""
    result = _normalize_text("어 안녕하세요", tenant_keyterms=[])
    assert result == "안녕하세요"


# ---------------------------------------------------------------------------
# _is_ambiguous — 모호(True) 케이스
# ---------------------------------------------------------------------------


def test_ambiguity_short_text_no_signal():
    """중간 길이(5자) + 신호 없음 → 모호.

    "대중 교통" len=5 → 4 < 5 ≤ 7 구간.
    matched_keywords 없음 + distance 1.253 > 0.85 → 신호 약함 → 다음 조건 평가.
    is_auth=False, matched_keywords 없음, distance > 0.95 → 모두 탈락 → short_no_signal.
    """
    is_amb, reason = _is_ambiguous(
        "대중 교통",
        rag_probe={"top_distance": 1.253, "matched_keywords": [], "is_auth": False},
        auth_pending=False,
    )
    assert is_amb is True
    assert reason == "short_no_signal"


def test_is_auth_weak_distance_ignored():
    """is_auth=True 이지만 distance 1.253 > 0.85 → rag_auth_signal 조건 미충족 → 모호.

    "뭐" len=1 ≤ SHORT_LEN(4) → 구간 2·3 스킵.
    is_auth=True + distance 1.253 > 0.85 → 조건 4 탈락.
    matched_keywords 없음, distance > 0.95 → 모두 탈락 → short_no_signal.
    """
    is_amb, reason = _is_ambiguous(
        "뭐",
        rag_probe={"top_distance": 1.253, "matched_keywords": [], "is_auth": True},
        auth_pending=False,
    )
    assert is_amb is True
    assert reason == "short_no_signal"


# ---------------------------------------------------------------------------
# _is_ambiguous — 명확(False) 케이스
# ---------------------------------------------------------------------------


def test_ambiguity_mid_text_with_keyword():
    """중간 길이(5자) + matched_keywords 동반 → sufficient_with_signal."""
    is_amb, reason = _is_ambiguous(
        "대중 교통",
        rag_probe={"top_distance": 1.253, "matched_keywords": ["대중교통"], "is_auth": False},
        auth_pending=False,
    )
    assert is_amb is False
    assert reason == "sufficient_with_signal"


def test_ambiguity_long_text_no_signal():
    """len > 7 + 필러 단독 아님 → sufficient_length (distance/keywords 무관).

    "어떻게가죠 대중 교통타고가능방법을알려주세요" len=21.
    distance=0.924 < 0.95 지만 길이 조건이 먼저 매칭되어 sufficient_length 반환.
    """
    is_amb, reason = _is_ambiguous(
        "어떻게가죠 대중 교통타고가능방법을알려주세요",
        rag_probe={"top_distance": 0.924, "matched_keywords": [], "is_auth": False},
        auth_pending=False,
    )
    assert is_amb is False
    assert reason == "sufficient_length"


def test_is_auth_strong_distance_passes():
    """is_auth=True + distance 0.7 ≤ 0.85 → rag_auth_signal."""
    is_amb, reason = _is_ambiguous(
        "뭐",
        rag_probe={"top_distance": 0.7, "matched_keywords": [], "is_auth": True},
        auth_pending=False,
    )
    assert is_amb is False
    assert reason == "rag_auth_signal"


def test_auth_pending_short_circuit():
    """auth_pending=True → 조기 반환 auth_pending (rag_probe=None 이어도 안전)."""
    is_amb, reason = _is_ambiguous(
        "어어",
        rag_probe=None,
        auth_pending=True,
    )
    assert is_amb is False
    assert reason == "auth_pending"

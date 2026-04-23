"""Extract question-ready metadata from semantic chunks.

This module handles the first intent bootstrap stage only:
- call the LLM for one chunk or a batch of chunks
- parse category, product_name, raw_keywords, and askable_units
- clean and normalize extracted askable units
- return a safe fallback on failures

It does not perform storage, embeddings, slug generation, or leaf intent creation.
"""

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.llm.base import BaseLLMService
from app.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_SYSTEM_PROMPT = """
너는 semantic chunk에서 질문 가능한 구조화 정보를 추출하는 시스템이다.
반드시 JSON 객체만 출력하고 설명, 코드블록, 마크다운을 추가하지 마라.

출력 형식:
{
  "category": "문서 카테고리",
  "product_name": "청크 핵심 주제",
  "raw_keywords": ["핵심 키워드1", "핵심 키워드2"],
  "askable_units": [
    {
      "attribute": "사용자가 물어볼 속성명",
      "entity": "질문의 대상",
      "value_hint": "이 청크가 답하는 값의 요약",
      "evidence": "원문 근거 문장",
      "salience": 0.95,
      "answerability": 0.98
    }
  ]
}

규칙:
- JSON만 출력
- raw_keywords는 3~8개
- askable_units는 1~5개
- attribute는 '무엇을 묻는가'가 드러나야 한다
- entity는 '무엇에 대한 질문인가'가 드러나야 한다
- evidence는 반드시 입력 chunk 원문 일부여야 한다
- salience, answerability는 0~1 float
- category가 불명확하면 "기타"
- product_name이 불명확하면 ""
""".strip()

GENERIC_ATTRIBUTES = frozenset(
    {
        "\uc815\ubcf4",
        "\uc548\ub0b4",
        "\ub0b4\uc6a9",
        "\uae30\ud0c0",
        "\ubb38\uc758",
        "\ubb38\uc758\uc0ac\ud56d",
        "\uc0ac\ud56d",
        "\uc124\uba85",
        "\uad00\ub828 \ub0b4\uc6a9",
        "\uc11c\ube44\uc2a4 \uc548\ub0b4",
        "\uc774\uc6a9 \uc548\ub0b4",
        "\uae30\ubcf8 \uc815\ubcf4",
    }
)
GENERIC_ATTRIBUTE_KEYS = frozenset(value.casefold() for value in GENERIC_ATTRIBUTES)
MAX_RAW_KEYWORDS = 8
MAX_ASKABLE_UNITS = 5
FALLBACK_CATEGORY = "\uae30\ud0c0"


@dataclass
class AskableUnit:
    attribute: str
    entity: str
    value_hint: str
    evidence: str
    salience: float = 0.0
    answerability: float = 0.0


@dataclass
class ChunkIntentExtraction:
    chunk_id: str
    chunk_text: str
    category: str
    product_name: str
    raw_keywords: list[str] = field(default_factory=list)
    askable_units: list[AskableUnit] = field(default_factory=list)


def extraction_to_dict(extraction: ChunkIntentExtraction) -> dict[str, Any]:
    return asdict(extraction)


class ChunkExtractor:
    def __init__(self, llm: BaseLLMService):
        self._llm = llm

    async def extract_one(self, chunk_id: str, chunk_text: str) -> ChunkIntentExtraction:
        if not chunk_text.strip():
            return self._fallback_result(chunk_id, chunk_text)

        try:
            raw = await self._llm.generate(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_message=chunk_text,
                temperature=0.1,
                max_tokens=400,
            )
            cleaned = self._clean_llm_json(raw)
            return self._parse_extraction(chunk_id, chunk_text, cleaned)
        except Exception:
            logger.exception("chunk extraction failed chunk_id=%s", chunk_id)
            return self._fallback_result(chunk_id, chunk_text)

    async def extract_batch(self, items: list[tuple[str, str]]) -> list[ChunkIntentExtraction]:
        if not items:
            return []

        tasks = [self.extract_one(chunk_id, chunk_text) for chunk_id, chunk_text in items]
        return list(await asyncio.gather(*tasks))

    def _clean_llm_json(self, raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = _extract_json_object(cleaned)
        return cleaned.strip()

    def _parse_extraction(
        self,
        chunk_id: str,
        chunk_text: str,
        cleaned: str,
    ) -> ChunkIntentExtraction:
        if not cleaned:
            logger.warning("chunk extraction empty response chunk_id=%s", chunk_id)
            return self._fallback_result(chunk_id, chunk_text)

        try:
            payload = json.loads(cleaned)
            if not isinstance(payload, dict):
                logger.warning("chunk extraction response is not dict chunk_id=%s", chunk_id)
                return self._fallback_result(chunk_id, chunk_text)

            category = _coerce_text(payload.get("category")) or FALLBACK_CATEGORY
            product_name = _coerce_text(payload.get("product_name"))
            raw_keywords = _clean_raw_keywords(payload.get("raw_keywords"))
            askable_units = _parse_askable_units(payload.get("askable_units"))
            askable_units = self._clean_askable_units(chunk_id, chunk_text, askable_units)

            return ChunkIntentExtraction(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                category=category,
                product_name=product_name,
                raw_keywords=raw_keywords,
                askable_units=askable_units,
            )
        except json.JSONDecodeError:
            logger.warning("chunk extraction json parse failed chunk_id=%s", chunk_id)
            return self._fallback_result(chunk_id, chunk_text)
        except Exception:
            logger.exception("chunk extraction normalization failed chunk_id=%s", chunk_id)
            return self._fallback_result(chunk_id, chunk_text)

    def _clean_askable_units(
        self,
        chunk_id: str,
        chunk_text: str,
        units: list[AskableUnit],
    ) -> list[AskableUnit]:
        cleaned_units: list[AskableUnit] = []
        seen: set[tuple[str, str]] = set()
        normalized_chunk_text = _normalize_ws(chunk_text)

        for unit in units:
            attribute = unit.attribute.strip()
            entity = unit.entity.strip()
            value_hint = unit.value_hint.strip()
            evidence = unit.evidence.strip()

            if not attribute:
                self._log_dropped_unit(chunk_id, "empty_attribute", attribute, entity)
                continue

            if attribute.casefold() in GENERIC_ATTRIBUTE_KEYS:
                self._log_dropped_unit(chunk_id, "generic_attribute", attribute, entity)
                continue

            if not evidence:
                self._log_dropped_unit(chunk_id, "missing_evidence", attribute, entity)
                continue

            if evidence not in chunk_text:
                normalized_evidence = _normalize_ws(evidence)
                if not normalized_evidence or normalized_evidence not in normalized_chunk_text:
                    self._log_dropped_unit(chunk_id, "missing_evidence", attribute, entity)
                    continue

            dedupe_key = (_normalize_key(attribute), _normalize_key(entity))
            if dedupe_key in seen:
                self._log_dropped_unit(chunk_id, "duplicate_unit", attribute, entity)
                continue

            seen.add(dedupe_key)
            cleaned_units.append(
                AskableUnit(
                    attribute=attribute,
                    entity=entity,
                    value_hint=value_hint,
                    evidence=evidence,
                    salience=_clamp_score(unit.salience),
                    answerability=_clamp_score(unit.answerability),
                )
            )

            if len(cleaned_units) >= MAX_ASKABLE_UNITS:
                break

        return cleaned_units

    def _fallback_result(self, chunk_id: str, chunk_text: str) -> ChunkIntentExtraction:
        return ChunkIntentExtraction(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            category=FALLBACK_CATEGORY,
            product_name="",
            raw_keywords=[],
            askable_units=[],
        )

    def _log_dropped_unit(
        self,
        chunk_id: str,
        reason: str,
        attribute: str,
        entity: str,
    ) -> None:
        logger.debug(
            "drop askable_unit chunk_id=%s reason=%s attribute=%s entity=%s",
            chunk_id,
            reason,
            attribute,
            entity,
        )


def _clean_raw_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned_keywords: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            continue

        keyword = item.strip()
        if not keyword:
            continue

        dedupe_key = keyword.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        cleaned_keywords.append(keyword)

        if len(cleaned_keywords) >= MAX_RAW_KEYWORDS:
            break

    return cleaned_keywords


def _parse_askable_units(value: Any) -> list[AskableUnit]:
    if not isinstance(value, list):
        return []

    parsed_units: list[AskableUnit] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        parsed_units.append(
            AskableUnit(
                attribute=_coerce_text(item.get("attribute")),
                entity=_coerce_text(item.get("entity")),
                value_hint=_coerce_text(item.get("value_hint")),
                evidence=_coerce_text(item.get("evidence")),
                salience=_coerce_float(item.get("salience")),
                answerability=_coerce_float(item.get("answerability")),
            )
        )

    return parsed_units


def _coerce_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def _normalize_key(text: str) -> str:
    text = " ".join(text.split()).casefold()
    text = re.sub("[^\\w\\s\uAC00-\uD7A3]", "", text)
    return text


# Usage example:
# extractor = ChunkExtractor(llm)
# result = await extractor.extract_one("chunk-1", "Business hours are 9 AM to 6 PM on weekdays.")
# payload = extraction_to_dict(result)

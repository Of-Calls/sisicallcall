"""Generate example question candidates for normalized leaf intents.

This module is the third bootstrap stage for intent routing. It consumes
normalized leaf intent candidates and generates user-question examples that can
later be used as KNN training candidates.

It does not perform persistence, embedding deduplication, index construction,
or graph linkage.
"""

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.knn_router.intent_bootstrap.normalizer import (
    ChunkIntentCandidates,
    NormalizedAskableUnit,
)
from app.services.llm.base import BaseLLMService
from app.utils.logger import get_logger

logger = get_logger(__name__)

SEED_EXAMPLE_SYSTEM_PROMPT = """
너는 실제 고객이 전화로 할 법한 짧은 질문문 생성기다.
입력으로 주어지는 attribute, entity, value_hint, evidence, product_name을 참고해서
자연스러운 한국어 문의 문장만 만든다.

규칙:
- 반드시 질문문만 출력
- 3개에서 5개 사이로 출력
- 한 줄에 한 문장씩 출력
- 답변 금지
- 설명 금지
- 자연스러운 전화 문의 구어체 우선
- evidence를 그대로 복붙하지 않아도 되지만 의미는 유지

질문 스타일 예시:
- 운영 시간 -> 몇 시까지 해요?
- 예약 필요 여부 -> 이거 예약해야 하나요?
- 준비사항 -> 가기 전에 준비할 게 있나요?
""".strip()

PARAPHRASE_SYSTEM_PROMPT = """
너는 대표 질문을 다양한 실제 사용자 발화로 바꾸는 패러프레이즈 생성기다.
입력 질문들의 의미는 유지하되 표현을 다양하게 바꿔라.

규칙:
- 반드시 질문문만 출력
- 8개에서 12개 사이로 출력
- 한 줄에 한 문장씩 출력
- 답변 금지
- 설명 금지
- 지나치게 비슷한 표현 반복 금지
- 문어체보다 전화 문의 구어체 우선
""".strip()

MIN_QUESTION_LENGTH = 5
QUESTION_ENDINGS = (
    "?",
    "요?",
    "나요?",
    "한가요?",
    "될까요?",
    "알려주세요",
)
QUESTION_HINT_KEYWORDS = (
    "어떻게",
    "언제",
    "어디",
    "몇 시",
    "몇시",
    "얼마",
    "가능",
    "되나요",
    "되는지",
    "있나요",
    "있을까요",
    "필요",
    "예약",
    "준비",
    "문의",
    "알려주세요",
    "궁금",
)


@dataclass
class ExampleSentence:
    text: str
    source: str
    leaf_intent: str
    topic_slug: str
    detail_slug: str
    chunk_id: str = ""
    score_hint: float = 0.0


@dataclass
class LeafIntentExampleSet:
    leaf_intent: str
    topic_slug: str
    detail_slug: str
    seed_examples: list[ExampleSentence] = field(default_factory=list)
    paraphrase_examples: list[ExampleSentence] = field(default_factory=list)


@dataclass
class ExampleGenerationResult:
    leaf_intent_sets: list[LeafIntentExampleSet] = field(default_factory=list)


def example_result_to_dict(result: ExampleGenerationResult) -> dict[str, Any]:
    return asdict(result)


class ExampleGenerator:
    def __init__(self, llm: BaseLLMService):
        self._llm = llm

    async def generate_from_candidates(
        self,
        candidates: list[ChunkIntentCandidates],
    ) -> ExampleGenerationResult:
        grouped = self._group_units_by_leaf_intent(candidates)
        if not grouped:
            return ExampleGenerationResult()

        tasks = []
        for leaf_intent, grouped_units in grouped.items():
            if not grouped_units:
                continue
            chunk_id = grouped_units[0][1]
            product_name = grouped_units[0][2]
            tasks.append(
                self._generate_for_leaf_intent_from_sources(
                    leaf_intent=leaf_intent,
                    source_units=grouped_units,
                    chunk_id=chunk_id,
                    product_name=product_name,
                )
            )

        if not tasks:
            return ExampleGenerationResult()

        leaf_intent_sets = list(await asyncio.gather(*tasks))
        return ExampleGenerationResult(leaf_intent_sets=leaf_intent_sets)

    async def generate_for_leaf_intent(
        self,
        leaf_intent: str,
        units: list[NormalizedAskableUnit],
        chunk_id: str,
        product_name: str,
    ) -> LeafIntentExampleSet:
        source_units = [(unit, chunk_id, product_name) for unit in units]
        return await self._generate_for_leaf_intent_from_sources(
            leaf_intent=leaf_intent,
            source_units=source_units,
            chunk_id=chunk_id,
            product_name=product_name,
        )

    def _group_units_by_leaf_intent(
        self,
        candidates: list[ChunkIntentCandidates],
    ) -> dict[str, list[tuple[NormalizedAskableUnit, str, str]]]:
        grouped: dict[str, list[tuple[NormalizedAskableUnit, str, str]]] = defaultdict(list)

        for candidate in candidates:
            for unit in candidate.normalized_units:
                leaf_intent = unit.leaf_intent.strip()
                if not leaf_intent:
                    continue
                grouped[leaf_intent].append((unit, candidate.chunk_id, candidate.product_name))

        return dict(grouped)

    def _pick_seed_source_units(
        self,
        units: list[tuple[NormalizedAskableUnit, str, str]],
        max_units: int = 3,
    ) -> list[tuple[NormalizedAskableUnit, str, str]]:
        if not units or max_units <= 0:
            return []

        ranked = sorted(
            units,
            key=lambda item: self._unit_score(item[0]),
            reverse=True,
        )
        return ranked[:max_units]

    async def _generate_seed_examples(
        self,
        leaf_intent: str,
        topic_slug: str,
        detail_slug: str,
        source_units: list[tuple[NormalizedAskableUnit, str, str]],
    ) -> list[str]:
        if not source_units:
            return []

        source_blocks: list[str] = []
        for index, (unit, chunk_id, product_name) in enumerate(source_units, start=1):
            source_blocks.append(
                "\n".join(
                    [
                        f"[source {index}]",
                        f"chunk_id: {chunk_id}",
                        f"product_name: {product_name}",
                        f"attribute: {unit.attribute}",
                        f"entity: {unit.entity}",
                        f"value_hint: {unit.value_hint}",
                        f"evidence: {unit.evidence}",
                    ]
                )
            )

        user_message = (
            f"leaf_intent: {leaf_intent}\n"
            f"topic_slug: {topic_slug}\n"
            f"detail_slug: {detail_slug}\n\n"
            f"source units:\n{chr(10).join(source_blocks)}"
        )

        try:
            raw = await self._llm.generate(
                system_prompt=SEED_EXAMPLE_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.3,
                max_tokens=320,
            )
            parsed = self._parse_lines(raw)
            if len(parsed) >= 3:
                return parsed[:5]
            if parsed:
                fallback = self._build_fallback_seed_texts(source_units)
                return (parsed + fallback)[:5]
        except Exception:
            logger.exception("failed to generate seed examples leaf_intent=%s", leaf_intent)

        return self._build_fallback_seed_texts(source_units)[:5]

    async def _generate_paraphrases(
        self,
        leaf_intent: str,
        topic_slug: str,
        detail_slug: str,
        seed_examples: list[str],
        target_count: int = 8,
    ) -> list[str]:
        if not seed_examples:
            return []

        user_message = (
            f"leaf_intent: {leaf_intent}\n"
            f"topic_slug: {topic_slug}\n"
            f"detail_slug: {detail_slug}\n"
            f"target_count: {target_count}\n\n"
            "seed questions:\n- "
            + "\n- ".join(seed_examples)
        )

        try:
            raw = await self._llm.generate(
                system_prompt=PARAPHRASE_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.5,
                max_tokens=640,
            )
            parsed = self._parse_lines(raw)
            if not parsed:
                return []
            return parsed[: max(target_count, 12)]
        except Exception:
            logger.exception("failed to generate paraphrases leaf_intent=%s", leaf_intent)
            return []

    def _parse_lines(self, raw: str) -> list[str]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        items: list[str] = []
        if cleaned.startswith("["):
            try:
                payload = json.loads(cleaned)
                if isinstance(payload, list):
                    items = [item for item in payload if isinstance(item, str)]
            except json.JSONDecodeError:
                items = []

        if not items and cleaned.startswith("{"):
            try:
                payload = json.loads(cleaned)
                if isinstance(payload, dict):
                    for key in ("questions", "examples", "items"):
                        value = payload.get(key)
                        if isinstance(value, list):
                            items = [item for item in value if isinstance(item, str)]
                            break
            except json.JSONDecodeError:
                items = []

        if not items:
            items = cleaned.splitlines()

        parsed: list[str] = []
        seen: set[str] = set()

        for item in items:
            line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", item).strip()
            line = self._clean_question_text(line)
            if not line:
                continue
            if not self._looks_like_question(line):
                continue
            if line in seen:
                continue
            seen.add(line)
            parsed.append(line)

        return parsed

    def _clean_question_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.strip("'\"`“”‘’")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) < MIN_QUESTION_LENGTH:
            return ""
        return cleaned

    def _make_example_sentence(
        self,
        text: str,
        source: str,
        leaf_intent: str,
        topic_slug: str,
        detail_slug: str,
        chunk_id: str = "",
        score_hint: float = 0.0,
    ) -> ExampleSentence:
        return ExampleSentence(
            text=text,
            source=source,
            leaf_intent=leaf_intent,
            topic_slug=topic_slug,
            detail_slug=detail_slug,
            chunk_id=chunk_id,
            score_hint=score_hint,
        )

    def _fallback_example_set(
        self,
        leaf_intent: str,
        topic_slug: str,
        detail_slug: str,
    ) -> LeafIntentExampleSet:
        return LeafIntentExampleSet(
            leaf_intent=leaf_intent,
            topic_slug=topic_slug,
            detail_slug=detail_slug,
            seed_examples=[],
            paraphrase_examples=[],
        )

    async def _generate_for_leaf_intent_from_sources(
        self,
        leaf_intent: str,
        source_units: list[tuple[NormalizedAskableUnit, str, str]],
        chunk_id: str,
        product_name: str,
    ) -> LeafIntentExampleSet:
        if not source_units:
            return self._fallback_example_set(
                leaf_intent=leaf_intent,
                topic_slug="",
                detail_slug="",
            )

        picked_units = self._pick_seed_source_units(source_units)
        if not picked_units:
            return self._fallback_example_set(
                leaf_intent=leaf_intent,
                topic_slug="",
                detail_slug="",
            )

        anchor_unit = picked_units[0][0]
        topic_slug = anchor_unit.topic_slug
        detail_slug = anchor_unit.detail_slug
        anchor_chunk_id = picked_units[0][1] or chunk_id
        anchor_score = self._unit_score(anchor_unit)

        seed_texts = await self._generate_seed_examples(
            leaf_intent=leaf_intent,
            topic_slug=topic_slug,
            detail_slug=detail_slug,
            source_units=picked_units,
        )

        paraphrase_texts: list[str] = []
        if seed_texts:
            paraphrase_texts = await self._generate_paraphrases(
                leaf_intent=leaf_intent,
                topic_slug=topic_slug,
                detail_slug=detail_slug,
                seed_examples=seed_texts,
            )

        if not seed_texts and not paraphrase_texts:
            return self._fallback_example_set(
                leaf_intent=leaf_intent,
                topic_slug=topic_slug,
                detail_slug=detail_slug,
            )

        seed_examples = [
            self._make_example_sentence(
                text=text,
                source="seed",
                leaf_intent=leaf_intent,
                topic_slug=topic_slug,
                detail_slug=detail_slug,
                chunk_id=anchor_chunk_id,
                score_hint=anchor_score,
            )
            for text in seed_texts
        ]

        paraphrase_examples = [
            self._make_example_sentence(
                text=text,
                source="paraphrase",
                leaf_intent=leaf_intent,
                topic_slug=topic_slug,
                detail_slug=detail_slug,
                chunk_id=anchor_chunk_id,
                score_hint=anchor_score,
            )
            for text in paraphrase_texts
        ]

        return LeafIntentExampleSet(
            leaf_intent=leaf_intent,
            topic_slug=topic_slug,
            detail_slug=detail_slug,
            seed_examples=seed_examples,
            paraphrase_examples=paraphrase_examples,
        )

    def _build_fallback_seed_texts(
        self,
        source_units: list[tuple[NormalizedAskableUnit, str, str]],
    ) -> list[str]:
        fallback_texts: list[str] = []
        seen: set[str] = set()

        for unit, _chunk_id, product_name in source_units:
            target = unit.entity.strip() or product_name.strip()
            attribute = unit.attribute.strip()
            value_hint = unit.value_hint.strip()

            templates = self._make_fallback_templates(
                attribute=attribute,
                target=target,
                value_hint=value_hint,
            )

            for text in templates:
                cleaned = self._clean_question_text(text)
                if not cleaned or not self._looks_like_question(cleaned):
                    continue
                if cleaned in seen:
                    continue
                seen.add(cleaned)
                fallback_texts.append(cleaned)

        return fallback_texts

    def _make_fallback_templates(
        self,
        attribute: str,
        target: str,
        value_hint: str,
    ) -> list[str]:
        target_prefix = f"{target} " if target else ""
        templates = [
            f"{target_prefix}{attribute} 어떻게 되나요?",
            f"{target_prefix}{attribute} 알려주세요",
        ]

        if "시간" in attribute or "운영" in attribute or "영업" in attribute:
            templates.extend(
                [
                    f"{target_prefix}몇 시까지 하나요?",
                    f"{target_prefix}운영 시간이 어떻게 되나요?",
                ]
            )
        elif "예약" in attribute:
            templates.extend(
                [
                    f"{target_prefix}예약해야 하나요?",
                    f"{target_prefix}사전 예약이 필요한가요?",
                ]
            )
        elif "준비" in attribute or "유의" in attribute:
            templates.extend(
                [
                    f"{target_prefix}가기 전에 준비할 게 있나요?",
                    f"{target_prefix}미리 챙겨야 할 게 있나요?",
                ]
            )
        elif "위치" in attribute or "장소" in attribute:
            templates.extend(
                [
                    f"{target_prefix}어디로 가면 되나요?",
                    f"{target_prefix}위치가 어디인가요?",
                ]
            )
        elif "비용" in attribute or "가격" in attribute or "금액" in attribute:
            templates.extend(
                [
                    f"{target_prefix}비용이 얼마인가요?",
                    f"{target_prefix}가격 좀 알려주세요",
                ]
            )
        else:
            templates.append(f"{target_prefix}문의드리려고 하는데 {attribute} 관련해서 알려주세요")

        if value_hint and ("시간" in attribute or "운영" in attribute):
            templates.append(f"{target_prefix}{value_hint} 기준으로 보면 몇 시까지 가능한가요?")

        return templates

    def _looks_like_question(self, text: str) -> bool:
        if len(text) < MIN_QUESTION_LENGTH:
            return False

        normalized = text.strip()
        normalized = normalized.rstrip(".! ")
        if not normalized:
            return False

        if normalized.endswith(QUESTION_ENDINGS):
            return True

        return any(keyword in normalized for keyword in QUESTION_HINT_KEYWORDS)

    def _unit_score(self, unit: NormalizedAskableUnit) -> float:
        return (unit.salience * 0.6) + (unit.answerability * 0.4)


# Usage example:
# generator = ExampleGenerator(llm)
# result = await generator.generate_from_candidates(candidates)

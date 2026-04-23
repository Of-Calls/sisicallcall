"""Normalize extracted askable units into leaf intent candidates.

This module is the second bootstrap stage for intent routing. It consumes
ChunkIntentExtraction results, clusters semantically similar attributes,
generates canonical topic/detail slugs, and produces leaf intent candidates.

It does not perform database writes, KNN indexing, example generation, or
graph construction.
"""

import asyncio
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.embedding.base import BaseEmbeddingService
from app.services.knn_router.intent_bootstrap.chunk_extractor import ChunkIntentExtraction
from app.services.llm.base import BaseLLMService
from app.utils.logger import get_logger

logger = get_logger(__name__)

TOPIC_SLUG_SYSTEM_PROMPT = """
너는 여러 질문 속성(attribute) 표현을 대표하는 영어 snake_case topic slug 생성기다.
입력은 주로 한국어지만 출력은 반드시 영어 slug 하나만 해야 한다.

규칙:
- 영어 snake_case slug 하나만 출력
- JSON 출력 금지
- 따옴표 금지
- 설명 금지
- 소문자 영어, 숫자, 밑줄만 사용
- 클러스터 전체를 대표할 수 있도록 짧고 일관된 topic slug를 만든다

예시:
- 운영 시간, 영업 시간, 진료 시간 -> business_hours
- 예약 필요 여부, 사전 예약 필요 -> reservation_required
- 준비사항, 검사 전 유의사항 -> preparation
""".strip()

DETAIL_SLUG_SYSTEM_PROMPT = """
너는 질문 대상(entity 또는 product_name)을 대표하는 영어 snake_case detail slug 생성기다.
입력은 주로 한국어지만 출력은 반드시 영어 slug 하나만 해야 한다.

규칙:
- 영어 snake_case slug 하나만 출력
- JSON 출력 금지
- 따옴표 금지
- 설명 금지
- 소문자 영어, 숫자, 밑줄만 사용
- entity가 있으면 entity를 우선 반영하고, 없으면 product_name을 반영한다

예시:
- 외래 진료 -> outpatient
- MRI 검사 -> mri
- 주차장 -> parking_lot
- 응급실 -> emergency_room
""".strip()

DEFAULT_BRANCH = "faq"
DEFAULT_DETAIL_SLUG = "general"
DEFAULT_TOPIC_SLUG = "general"


@dataclass
class NormalizedAskableUnit:
    attribute: str
    entity: str
    value_hint: str
    evidence: str
    salience: float
    answerability: float
    topic_slug: str
    detail_slug: str
    leaf_intent: str


@dataclass
class ChunkIntentCandidates:
    chunk_id: str
    chunk_text: str
    category: str
    product_name: str
    raw_keywords: list[str] = field(default_factory=list)
    primary_leaf_intent: str = ""
    candidate_leaf_intents: list[str] = field(default_factory=list)
    normalized_units: list[NormalizedAskableUnit] = field(default_factory=list)


@dataclass
class TopicCluster:
    cluster_id: int
    members: list[str] = field(default_factory=list)
    representative_attribute: str = ""
    topic_slug: str = ""


def candidates_to_dict(candidate: ChunkIntentCandidates) -> dict[str, Any]:
    return asdict(candidate)


class IntentNormalizer:
    def __init__(
        self,
        embedder: BaseEmbeddingService,
        llm: BaseLLMService,
    ):
        self._embedder = embedder
        self._llm = llm

    async def normalize_chunks(
        self,
        extractions: list[ChunkIntentExtraction],
    ) -> list[ChunkIntentCandidates]:
        if not extractions:
            return []

        try:
            topic_mapping = await self._build_topic_mapping(extractions)
        except Exception:
            logger.exception("failed to build topic mapping")
            topic_mapping = {}

        tasks = [self.normalize_one(extraction, topic_mapping) for extraction in extractions]
        return list(await asyncio.gather(*tasks))

    async def normalize_one(
        self,
        extraction: ChunkIntentExtraction,
        topic_mapping: dict[str, str],
    ) -> ChunkIntentCandidates:
        try:
            if not extraction.askable_units:
                return self._fallback_candidate(extraction)

            detail_cache: dict[tuple[str, str], str] = {}
            normalized_units: list[NormalizedAskableUnit] = []

            for unit in extraction.askable_units:
                topic_slug = topic_mapping.get(unit.attribute) or self._slugify_fallback(
                    unit.attribute
                )
                detail_key = (unit.entity.strip(), extraction.product_name.strip())
                detail_slug = detail_cache.get(detail_key)
                if detail_slug is None:
                    detail_slug = await self._make_detail_slug(
                        unit.entity,
                        extraction.product_name,
                    )
                    detail_cache[detail_key] = detail_slug

                leaf_intent = self._build_leaf_intent(topic_slug, detail_slug)
                normalized_units.append(
                    NormalizedAskableUnit(
                        attribute=unit.attribute,
                        entity=unit.entity,
                        value_hint=unit.value_hint,
                        evidence=unit.evidence,
                        salience=unit.salience,
                        answerability=unit.answerability,
                        topic_slug=topic_slug,
                        detail_slug=detail_slug,
                        leaf_intent=leaf_intent,
                    )
                )

            merged_units = self._merge_duplicate_units(normalized_units)
            candidate_leaf_intents = self._dedupe_leaf_intents(merged_units)
            primary_leaf_intent = self._choose_primary_leaf_intent(merged_units)

            return ChunkIntentCandidates(
                chunk_id=extraction.chunk_id,
                chunk_text=extraction.chunk_text,
                category=extraction.category,
                product_name=extraction.product_name,
                raw_keywords=list(extraction.raw_keywords),
                primary_leaf_intent=primary_leaf_intent,
                candidate_leaf_intents=candidate_leaf_intents,
                normalized_units=merged_units,
            )
        except Exception:
            logger.exception("failed to normalize chunk chunk_id=%s", extraction.chunk_id)
            return self._fallback_candidate(extraction)

    async def _build_topic_mapping(
        self,
        extractions: list[ChunkIntentExtraction],
    ) -> dict[str, str]:
        texts = self._collect_attribute_texts(extractions)
        if not texts:
            return {}

        raw_attributes = self._collect_raw_attributes(extractions)
        embeddings = await self._embedder.embed_batch(texts)
        if not embeddings:
            return {}

        clusters = self._greedy_cluster_embeddings(texts, embeddings)
        if not clusters:
            return {}

        topic_clusters: list[TopicCluster] = []
        attribute_slug_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for cluster_id, cluster_indices in enumerate(clusters):
            cluster_members = [
                texts[index]
                for index in cluster_indices
                if 0 <= index < len(texts)
            ]
            if not cluster_members:
                continue

            representative_attribute = self._choose_cluster_representative(cluster_members)
            topic_slug = await self._make_topic_slug(
                representative_attribute,
                cluster_members,
            )

            topic_clusters.append(
                TopicCluster(
                    cluster_id=cluster_id,
                    members=cluster_members,
                    representative_attribute=representative_attribute,
                    topic_slug=topic_slug,
                )
            )

            for index in cluster_indices:
                if index >= len(raw_attributes):
                    continue
                raw_attribute = raw_attributes[index]
                if raw_attribute:
                    attribute_slug_counts[raw_attribute][topic_slug] += 1

        mapping: dict[str, str] = {}
        for raw_attribute, slug_counts in attribute_slug_counts.items():
            if len(slug_counts) >= 2:
                logger.debug(
                    "topic mapping collision attribute=%s slug_counts=%s",
                    raw_attribute,
                    dict(slug_counts),
                )
            mapping[raw_attribute] = slug_counts.most_common(1)[0][0]

        logger.debug(
            "topic mapping built clusters=%d attributes=%d",
            len(topic_clusters),
            len(mapping),
        )
        return mapping

    def _collect_attribute_texts(
        self,
        extractions: list[ChunkIntentExtraction],
    ) -> list[str]:
        texts: list[str] = []
        for extraction in extractions:
            for unit in extraction.askable_units:
                attribute = unit.attribute.strip()
                if not attribute:
                    continue
                entity = unit.entity.strip()
                texts.append(f"{attribute} | {entity}")
        return texts

    def _greedy_cluster_embeddings(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        threshold: float = 0.86,
    ) -> list[list[int]]:
        if not texts or not embeddings:
            return []

        limit = min(len(texts), len(embeddings))
        if limit == 0:
            return []

        clusters: list[list[int]] = []
        centroids: list[list[float]] = []
        cluster_sums: list[list[float]] = []

        for index in range(limit):
            embedding = self._normalize_embedding(embeddings[index])
            best_cluster_index = -1
            best_score = -1.0

            for cluster_index, centroid in enumerate(centroids):
                score = self._cosine_similarity(embedding, centroid)
                if score > best_score:
                    best_score = score
                    best_cluster_index = cluster_index

            if best_cluster_index == -1 or best_score < threshold:
                clusters.append([index])
                cluster_sums.append(list(embedding))
                centroids.append(self._normalize_embedding(embedding))
                continue

            clusters[best_cluster_index].append(index)
            cluster_sums[best_cluster_index] = [
                current + new_value
                for current, new_value in zip(cluster_sums[best_cluster_index], embedding)
            ]
            cluster_size = len(clusters[best_cluster_index])
            mean_centroid = [
                value / cluster_size for value in cluster_sums[best_cluster_index]
            ]
            centroids[best_cluster_index] = self._normalize_embedding(mean_centroid)

        return clusters

    def _choose_cluster_representative(
        self,
        cluster_texts: list[str],
    ) -> str:
        if not cluster_texts:
            return ""

        attributes = [self._split_attribute_entity(text)[0] for text in cluster_texts]
        filtered_attributes = [attribute for attribute in attributes if attribute]
        if not filtered_attributes:
            return self._split_attribute_entity(cluster_texts[0])[0]

        counts = Counter(filtered_attributes)
        best_count = counts.most_common(1)[0][1]
        for attribute in filtered_attributes:
            if counts[attribute] == best_count:
                return attribute

        return filtered_attributes[0]

    async def _make_topic_slug(
        self,
        representative_attribute: str,
        cluster_members: list[str],
    ) -> str:
        source_text = representative_attribute.strip()
        if not source_text:
            return DEFAULT_TOPIC_SLUG

        user_message = (
            f"대표 attribute: {source_text}\n"
            f"cluster members:\n- " + "\n- ".join(cluster_members)
        )

        try:
            raw = await self._llm.generate(
                system_prompt=TOPIC_SLUG_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=32,
            )
            cleaned = self._extract_slug(raw)
            if cleaned:
                return cleaned
        except Exception:
            logger.exception("failed to generate topic slug attribute=%s", source_text)

        return self._slugify_fallback(source_text)

    async def _make_detail_slug(
        self,
        entity: str,
        product_name: str,
    ) -> str:
        source_text = entity.strip() or product_name.strip()
        if not source_text:
            return DEFAULT_DETAIL_SLUG

        user_message = (
            f"entity: {entity.strip()}\n"
            f"product_name: {product_name.strip()}\n"
            f"대표 대상을 영어 snake_case detail slug 하나로 만들어라."
        )

        try:
            raw = await self._llm.generate(
                system_prompt=DETAIL_SLUG_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=32,
            )
            cleaned = self._extract_slug(raw)
            if cleaned:
                return cleaned
        except Exception:
            logger.exception(
                "failed to generate detail slug entity=%s product_name=%s",
                entity,
                product_name,
            )

        return self._slugify_fallback(source_text)

    def _build_leaf_intent(
        self,
        topic_slug: str,
        detail_slug: str,
        branch: str = DEFAULT_BRANCH,
    ) -> str:
        safe_branch = self._slugify_fallback(branch)
        safe_topic = self._slugify_fallback(topic_slug)
        safe_detail = self._slugify_fallback(detail_slug)
        return f"intent_{safe_branch}_{safe_topic}_{safe_detail}"

    def _choose_primary_leaf_intent(
        self,
        units: list[NormalizedAskableUnit],
    ) -> str:
        if not units:
            return ""

        best_unit = max(units, key=self._unit_score)
        return best_unit.leaf_intent

    def _dedupe_leaf_intents(
        self,
        units: list[NormalizedAskableUnit],
    ) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()

        for unit in units:
            leaf_intent = unit.leaf_intent.strip()
            if not leaf_intent or leaf_intent in seen:
                continue
            seen.add(leaf_intent)
            deduped.append(leaf_intent)

        return deduped

    def _slugify_fallback(self, text: str) -> str:
        raw = text.strip().lower()
        if not raw:
            return DEFAULT_DETAIL_SLUG

        ascii_text = raw.encode("ascii", "ignore").decode("ascii")
        ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
        ascii_text = re.sub(r"_+", "_", ascii_text).strip("_")
        if ascii_text:
            return ascii_text

        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
        return f"{DEFAULT_DETAIL_SLUG}_{digest}"

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _normalize_embedding(self, vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return list(vec)
        return [x / norm for x in vec]

    def _collect_raw_attributes(
        self,
        extractions: list[ChunkIntentExtraction],
    ) -> list[str]:
        attributes: list[str] = []
        for extraction in extractions:
            for unit in extraction.askable_units:
                attribute = unit.attribute.strip()
                if attribute:
                    attributes.append(attribute)
        return attributes

    def _split_attribute_entity(self, text: str) -> tuple[str, str]:
        if "|" not in text:
            return text.strip(), ""
        attribute, entity = text.split("|", 1)
        return attribute.strip(), entity.strip()

    def _extract_slug(self, raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:text|json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip().strip("'\"`")
        if not cleaned:
            return ""

        slug_tokens = re.findall(r"\b[a-z0-9]+(?:_[a-z0-9]+)*\b", cleaned.lower())
        if slug_tokens:
            return slug_tokens[-1]

        return self._slugify_fallback(cleaned)

    def _merge_duplicate_units(
        self,
        units: list[NormalizedAskableUnit],
    ) -> list[NormalizedAskableUnit]:
        merged_by_intent: dict[str, NormalizedAskableUnit] = {}
        order: list[str] = []

        for unit in units:
            leaf_intent = unit.leaf_intent
            if not leaf_intent:
                continue

            existing = merged_by_intent.get(leaf_intent)
            if existing is None:
                merged_by_intent[leaf_intent] = unit
                order.append(leaf_intent)
                continue

            if self._unit_score(unit) > self._unit_score(existing):
                merged_by_intent[leaf_intent] = unit

        return [merged_by_intent[leaf_intent] for leaf_intent in order]

    def _unit_score(self, unit: NormalizedAskableUnit) -> float:
        return (unit.salience * 0.6) + (unit.answerability * 0.4)

    def _fallback_candidate(
        self,
        extraction: ChunkIntentExtraction,
    ) -> ChunkIntentCandidates:
        return ChunkIntentCandidates(
            chunk_id=extraction.chunk_id,
            chunk_text=extraction.chunk_text,
            category=extraction.category,
            product_name=extraction.product_name,
            raw_keywords=list(extraction.raw_keywords),
            primary_leaf_intent="",
            candidate_leaf_intents=[],
            normalized_units=[],
        )


# Usage example:
# normalizer = IntentNormalizer(embedder, llm)
# candidates = await normalizer.normalize_chunks(extractions)

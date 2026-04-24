"""Build a canonical intent inventory from normalized leaf intent candidates.

This module consolidates auto-generated leaf intents into a canonical registry.
It collects support counts, merges conservative alias variants, exposes
alias-to-canonical mappings, derives coarse branch intents, and can apply the
registry back onto ChunkIntentCandidates.

It does not persist data or modify downstream runtime routing logic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.knn_router.intent_bootstrap.normalizer import (
    ChunkIntentCandidates,
    NormalizedAskableUnit,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

KNOWN_BRANCH_INTENTS = frozenset(
    {
        "intent_faq",
        "intent_task",
        "intent_auth",
        "intent_escalation",
    }
)
GENERIC_SLUG_RE = re.compile(r"^general(?:_[a-f0-9]{8})?$")

TOPIC_ALIAS_MAP = {
    "business_hours": {
        "hours",
        "business_hours",
        "opening_time",
        "opening_hours",
        "operating_hours",
        "clinic_hours",
        "service_hours",
        "business_time",
    },
    "reservation_required": {
        "reservation_required",
        "booking_required",
        "appointment_required",
        "pre_reservation_required",
        "advance_booking_required",
    },
    "preparation": {
        "preparation",
        "prep",
        "pre_exam_preparation",
        "pre_exam_instructions",
        "before_exam_preparation",
    },
    "location": {
        "location",
        "place",
        "directions",
        "where",
        "wayfinding",
    },
    "price": {
        "price",
        "cost",
        "fee",
        "charge",
        "pricing",
    },
}

DETAIL_ALIAS_MAP = {
    "outpatient": {
        "outpatient",
        "outpatient_clinic",
        "ambulatory",
    },
    "mri": {
        "mri",
        "mri_exam",
        "mri_scan",
    },
    "parking_lot": {
        "parking_lot",
        "parking",
        "parking_area",
    },
    "emergency_room": {
        "emergency_room",
        "er",
        "emergency",
        "emergency_department",
    },
}


@dataclass
class IntentAlias:
    alias_leaf_intent: str
    canonical_leaf_intent: str
    reason: str = ""
    similarity: float = 0.0


@dataclass
class IntentInventoryItem:
    canonical_leaf_intent: str
    branch_intent: str
    topic_slug: str
    detail_slug: str
    aliases: list[str] = field(default_factory=list)
    support_count: int = 0


@dataclass
class IntentRegistry:
    aliases: list[IntentAlias] = field(default_factory=list)
    inventory: list[IntentInventoryItem] = field(default_factory=list)
    alias_to_canonical: dict[str, str] = field(default_factory=dict)
    branch_by_leaf_intent: dict[str, str] = field(default_factory=dict)


@dataclass
class IntentRegistryBuildResult:
    registry: IntentRegistry
    candidates: list[ChunkIntentCandidates] = field(default_factory=list)


def registry_result_to_dict(result: IntentRegistryBuildResult) -> dict[str, Any]:
    return asdict(result)


@dataclass
class _LeafProfile:
    leaf_intent: str
    branch_intent: str
    topic_slug: str
    detail_slug: str
    support_count: int = 0


@dataclass
class _RegistryGroup:
    branch_intent: str
    normalized_topic: str
    normalized_detail: str
    members: list[_LeafProfile] = field(default_factory=list)


class IntentRegistryBuilder:
    def __init__(self) -> None:
        pass

    async def build_registry(
        self,
        candidates: list[ChunkIntentCandidates],
    ) -> IntentRegistryBuildResult:
        profiles = self._collect_leaf_profiles(candidates)
        groups = self._build_groups(profiles)
        registry = self._build_registry_from_groups(groups)
        canonical_candidates = self.apply_registry_to_candidates(candidates, registry)
        return IntentRegistryBuildResult(
            registry=registry,
            candidates=canonical_candidates,
        )

    def apply_registry_to_candidates(
        self,
        candidates: list[ChunkIntentCandidates],
        registry: IntentRegistry,
    ) -> list[ChunkIntentCandidates]:
        inventory_map = {
            item.canonical_leaf_intent: item for item in registry.inventory
        }

        updated_candidates: list[ChunkIntentCandidates] = []
        for candidate in candidates:
            canonical_primary = self.canonicalize_leaf_intent(
                candidate.primary_leaf_intent,
                registry,
            )

            canonical_units: list[NormalizedAskableUnit] = []
            for unit in candidate.normalized_units:
                canonical_leaf = self.canonicalize_leaf_intent(unit.leaf_intent, registry)
                inventory_item = inventory_map.get(canonical_leaf)
                canonical_units.append(
                    NormalizedAskableUnit(
                        attribute=unit.attribute,
                        entity=unit.entity,
                        value_hint=unit.value_hint,
                        evidence=unit.evidence,
                        salience=unit.salience,
                        answerability=unit.answerability,
                        topic_slug=inventory_item.topic_slug if inventory_item else unit.topic_slug,
                        detail_slug=inventory_item.detail_slug if inventory_item else unit.detail_slug,
                        leaf_intent=canonical_leaf,
                    )
                )

            canonical_candidate_leaf_intents = self._rebuild_candidate_leaf_intents(
                primary_leaf_intent=canonical_primary,
                candidate_leaf_intents=candidate.candidate_leaf_intents,
                units=canonical_units,
                registry=registry,
            )

            updated_candidates.append(
                ChunkIntentCandidates(
                    chunk_id=candidate.chunk_id,
                    chunk_text=candidate.chunk_text,
                    category=candidate.category,
                    product_name=candidate.product_name,
                    raw_keywords=list(candidate.raw_keywords),
                    primary_leaf_intent=canonical_primary,
                    candidate_leaf_intents=canonical_candidate_leaf_intents,
                    normalized_units=canonical_units,
                )
            )

        return updated_candidates

    def canonicalize_leaf_intent(
        self,
        leaf_intent: str,
        registry: IntentRegistry,
    ) -> str:
        if not leaf_intent.strip():
            return ""
        return registry.alias_to_canonical.get(leaf_intent, leaf_intent)

    def get_branch_intent_from_leaf(self, leaf_intent: str) -> str:
        parsed = self._parse_leaf_intent(leaf_intent)
        if parsed:
            branch_intent = f"intent_{parsed[0]}"
            if branch_intent in KNOWN_BRANCH_INTENTS:
                return branch_intent

        logger.debug(
            "unknown branch fallback leaf_intent=%s fallback=intent_faq",
            leaf_intent,
        )
        return "intent_faq"

    def _collect_leaf_profiles(
        self,
        candidates: list[ChunkIntentCandidates],
    ) -> list[_LeafProfile]:
        profiles_by_leaf: dict[str, _LeafProfile] = {}

        for candidate in candidates:
            for unit in candidate.normalized_units:
                leaf_intent = unit.leaf_intent.strip()
                if not leaf_intent:
                    continue
                profile = profiles_by_leaf.get(leaf_intent)
                if profile is None:
                    profile = _LeafProfile(
                        leaf_intent=leaf_intent,
                        branch_intent=self.get_branch_intent_from_leaf(leaf_intent),
                        topic_slug=unit.topic_slug,
                        detail_slug=unit.detail_slug,
                        support_count=0,
                    )
                    profiles_by_leaf[leaf_intent] = profile
                profile.support_count += 1

            if candidate.normalized_units:
                continue

            fallback_leafs = [
                leaf
                for leaf in [candidate.primary_leaf_intent, *candidate.candidate_leaf_intents]
                if leaf.strip()
            ]
            for leaf_intent in fallback_leafs:
                if leaf_intent not in profiles_by_leaf:
                    profiles_by_leaf[leaf_intent] = _LeafProfile(
                        leaf_intent=leaf_intent,
                        branch_intent=self.get_branch_intent_from_leaf(leaf_intent),
                        topic_slug=self._extract_topic_slug_from_leaf(leaf_intent),
                        detail_slug=self._extract_detail_slug_from_leaf(leaf_intent),
                        support_count=0,
                    )
                profiles_by_leaf[leaf_intent].support_count += 1

        return sorted(
            profiles_by_leaf.values(),
            key=lambda profile: (profile.branch_intent, profile.leaf_intent),
        )

    def _build_groups(self, profiles: list[_LeafProfile]) -> list[_RegistryGroup]:
        groups: list[_RegistryGroup] = []

        for profile in profiles:
            normalized_topic = self._normalize_topic_slug(profile.topic_slug)
            normalized_detail = self._normalize_detail_slug(profile.detail_slug)
            matched_group: _RegistryGroup | None = None

            for group in groups:
                if profile.branch_intent != group.branch_intent:
                    continue
                if not self._should_merge_profile(profile, group):
                    continue
                matched_group = group
                break

            if matched_group is None:
                groups.append(
                    _RegistryGroup(
                        branch_intent=profile.branch_intent,
                        normalized_topic=normalized_topic,
                        normalized_detail=normalized_detail,
                        members=[profile],
                    )
                )
                continue

            matched_group.members.append(profile)
            matched_group.normalized_topic = self._prefer_signature_value(
                matched_group.normalized_topic,
                normalized_topic,
            )
            matched_group.normalized_detail = self._prefer_signature_value(
                matched_group.normalized_detail,
                normalized_detail,
            )

        return groups

    def _build_registry_from_groups(
        self,
        groups: list[_RegistryGroup],
    ) -> IntentRegistry:
        alias_to_canonical: dict[str, str] = {}
        branch_by_leaf_intent: dict[str, str] = {}
        aliases: list[IntentAlias] = []
        inventory: list[IntentInventoryItem] = []

        for group in groups:
            canonical_profile = self._choose_canonical_profile(group)
            canonical_leaf_intent = canonical_profile.leaf_intent
            aliases_for_item: list[str] = []
            support_count = sum(member.support_count for member in group.members)

            for member in sorted(group.members, key=lambda item: item.leaf_intent):
                similarity = self._profile_similarity(member, canonical_profile)
                reason = self._alias_reason(member, canonical_profile, group)
                alias_to_canonical[member.leaf_intent] = canonical_leaf_intent
                branch_by_leaf_intent[member.leaf_intent] = member.branch_intent
                aliases_for_item.append(member.leaf_intent)
                aliases.append(
                    IntentAlias(
                        alias_leaf_intent=member.leaf_intent,
                        canonical_leaf_intent=canonical_leaf_intent,
                        reason=reason,
                        similarity=similarity,
                    )
                )

            branch_by_leaf_intent[canonical_leaf_intent] = canonical_profile.branch_intent
            inventory.append(
                IntentInventoryItem(
                    canonical_leaf_intent=canonical_leaf_intent,
                    branch_intent=canonical_profile.branch_intent,
                    topic_slug=canonical_profile.topic_slug,
                    detail_slug=canonical_profile.detail_slug,
                    aliases=self._dedupe_preserve_order(aliases_for_item),
                    support_count=support_count,
                )
            )

        inventory.sort(key=lambda item: (item.branch_intent, item.canonical_leaf_intent))
        aliases.sort(key=lambda item: (item.canonical_leaf_intent, item.alias_leaf_intent))

        return IntentRegistry(
            aliases=aliases,
            inventory=inventory,
            alias_to_canonical=alias_to_canonical,
            branch_by_leaf_intent=branch_by_leaf_intent,
        )

    def _choose_canonical_profile(self, group: _RegistryGroup) -> _LeafProfile:
        return max(
            group.members,
            key=lambda profile: self._canonical_profile_score(profile, group),
        )

    def _canonical_profile_score(
        self,
        profile: _LeafProfile,
        group: _RegistryGroup,
    ) -> tuple[float, float, int, str]:
        support_score = float(profile.support_count)
        specificity_score = float(self._specificity(profile.topic_slug, profile.detail_slug))
        generic_penalty = 0.0
        if self._is_generic_slug(profile.topic_slug):
            generic_penalty += 2.0
        if self._is_generic_slug(profile.detail_slug):
            generic_penalty += 2.0

        normalized_bonus = 0.0
        if profile.topic_slug == group.normalized_topic:
            normalized_bonus += 1.5
        if profile.detail_slug == group.normalized_detail:
            normalized_bonus += 1.5

        total_score = (support_score * 10.0) + specificity_score + normalized_bonus - generic_penalty
        return (
            total_score,
            support_score,
            len(profile.leaf_intent),
            profile.leaf_intent,
        )

    def _should_merge_profile(
        self,
        profile: _LeafProfile,
        group: _RegistryGroup,
    ) -> bool:
        group_profiles = group.members
        if not group_profiles:
            return False

        if (
            self._normalize_topic_slug(profile.topic_slug) == group.normalized_topic
            and self._normalize_detail_slug(profile.detail_slug) == group.normalized_detail
        ):
            return True

        best_match = max(
            (self._profile_similarity(profile, member) for member in group_profiles),
            default=0.0,
        )

        same_detail = any(
            self._normalize_detail_slug(profile.detail_slug)
            == self._normalize_detail_slug(member.detail_slug)
            for member in group_profiles
        )
        same_topic = any(
            self._normalize_topic_slug(profile.topic_slug)
            == self._normalize_topic_slug(member.topic_slug)
            for member in group_profiles
        )

        if same_detail and best_match >= 0.72:
            return True
        if same_topic and best_match >= 0.72:
            return True

        if same_detail and self._is_generic_slug(profile.topic_slug):
            return True
        if same_topic and self._is_generic_slug(profile.detail_slug):
            return True

        return False

    def _profile_similarity(
        self,
        left: _LeafProfile,
        right: _LeafProfile,
    ) -> float:
        if left.branch_intent != right.branch_intent:
            return 0.0

        topic_similarity = self._slug_similarity(left.topic_slug, right.topic_slug)
        detail_similarity = self._slug_similarity(
            left.detail_slug,
            right.detail_slug,
            is_detail=True,
        )
        return (topic_similarity * 0.6) + (detail_similarity * 0.4)

    def _slug_similarity(self, left: str, right: str, is_detail: bool = False) -> float:
        if left == right:
            return 1.0

        if is_detail:
            left_normalized = self._normalize_detail_slug(left)
            right_normalized = self._normalize_detail_slug(right)
        else:
            left_normalized = self._normalize_topic_slug(left)
            right_normalized = self._normalize_topic_slug(right)
        if left_normalized == right_normalized:
            return 0.95

        left_tokens = set(self._slug_tokens(left))
        right_tokens = set(self._slug_tokens(right))
        if not left_tokens or not right_tokens:
            return 0.0

        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return intersection / union if union else 0.0

    def _normalize_topic_slug(self, topic_slug: str) -> str:
        slug = topic_slug.strip().lower()
        if not slug:
            return "general"
        if self._is_generic_slug(slug):
            return "general"
        for canonical, aliases in TOPIC_ALIAS_MAP.items():
            if slug == canonical or slug in aliases:
                return canonical
        tokens = set(self._slug_tokens(slug))
        if {"business", "hours"} <= tokens or {"opening", "time"} <= tokens:
            return "business_hours"
        if "reservation" in tokens and "required" in tokens:
            return "reservation_required"
        if "preparation" in tokens:
            return "preparation"
        return slug

    def _normalize_detail_slug(self, detail_slug: str) -> str:
        slug = detail_slug.strip().lower()
        if not slug:
            return "general"
        if self._is_generic_slug(slug):
            return "general"
        for canonical, aliases in DETAIL_ALIAS_MAP.items():
            if slug == canonical or slug in aliases:
                return canonical
        tokens = set(self._slug_tokens(slug))
        if "outpatient" in tokens:
            return "outpatient"
        if "mri" in tokens:
            return "mri"
        if "parking" in tokens:
            return "parking_lot"
        if "emergency" in tokens:
            return "emergency_room"
        return slug

    def _specificity(self, topic_slug: str, detail_slug: str) -> int:
        return len(self._slug_tokens(topic_slug)) + len(self._slug_tokens(detail_slug))

    def _slug_tokens(self, slug: str) -> list[str]:
        return [token for token in slug.split("_") if token]

    def _is_generic_slug(self, slug: str) -> bool:
        return bool(GENERIC_SLUG_RE.match(slug))

    def _prefer_signature_value(self, current: str, candidate: str) -> str:
        if self._is_generic_slug(current) and not self._is_generic_slug(candidate):
            return candidate
        return current

    def _alias_reason(
        self,
        alias_profile: _LeafProfile,
        canonical_profile: _LeafProfile,
        group: _RegistryGroup,
    ) -> str:
        if alias_profile.leaf_intent == canonical_profile.leaf_intent:
            return "canonical"
        if self._normalize_topic_slug(alias_profile.topic_slug) == group.normalized_topic:
            if alias_profile.topic_slug != canonical_profile.topic_slug:
                return "topic_alias"
        if self._normalize_detail_slug(alias_profile.detail_slug) == group.normalized_detail:
            if alias_profile.detail_slug != canonical_profile.detail_slug:
                return "detail_alias"
        if self._is_generic_slug(alias_profile.topic_slug) or self._is_generic_slug(alias_profile.detail_slug):
            return "generic_alias"
        return "similar_profile"

    def _parse_leaf_intent(self, leaf_intent: str) -> tuple[str, list[str]] | None:
        if not leaf_intent.startswith("intent_"):
            return None
        parts = leaf_intent.split("_")
        if len(parts) < 4:
            return None
        return parts[1], parts[2:]

    def _extract_topic_slug_from_leaf(self, leaf_intent: str) -> str:
        parsed = self._parse_leaf_intent(leaf_intent)
        if not parsed:
            return "general"
        remainder = parsed[1]
        if len(remainder) <= 1:
            return "general"
        detail_tokens = self._fallback_detail_tokens(remainder)
        topic_tokens = remainder[: len(remainder) - len(detail_tokens)]
        return "_".join(topic_tokens) or "general"

    def _extract_detail_slug_from_leaf(self, leaf_intent: str) -> str:
        parsed = self._parse_leaf_intent(leaf_intent)
        if not parsed:
            return "general"
        remainder = parsed[1]
        if len(remainder) <= 1:
            return "general"
        return "_".join(self._fallback_detail_tokens(remainder)) or "general"

    def _fallback_detail_tokens(self, remainder: list[str]) -> list[str]:
        if len(remainder) >= 2:
            last_two = "_".join(remainder[-2:])
            normalized_last_two = self._normalize_detail_slug(last_two)
            if self._is_generic_slug(last_two):
                return remainder[-2:]
            if normalized_last_two in DETAIL_ALIAS_MAP:
                return remainder[-2:]
            if any(last_two in aliases for aliases in DETAIL_ALIAS_MAP.values()):
                return remainder[-2:]
        return [remainder[-1]]

    def _rebuild_candidate_leaf_intents(
        self,
        primary_leaf_intent: str,
        candidate_leaf_intents: list[str],
        units: list[NormalizedAskableUnit],
        registry: IntentRegistry,
    ) -> list[str]:
        canonicalized_old_candidates = self._canonicalize_leaf_intents(
            candidate_leaf_intents,
            registry,
        )
        canonicalized_unit_leaf_intents = self._canonicalize_leaf_intents(
            [unit.leaf_intent for unit in units],
            registry,
        )
        return self._dedupe_preserve_order(
            [
                primary_leaf_intent,
                *canonicalized_old_candidates,
                *canonicalized_unit_leaf_intents,
            ]
        )

    def _canonicalize_leaf_intents(
        self,
        leaf_intents: list[str],
        registry: IntentRegistry,
    ) -> list[str]:
        return [
            self.canonicalize_leaf_intent(leaf_intent, registry)
            for leaf_intent in leaf_intents
            if leaf_intent.strip()
        ]

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def candidate_text(
    item: Any,
) -> str:
    """The text a candidate's identity is computed from.

    Mirrors the Context memory projection (``content`` first, then
    ``text``) so the same value is used on the retrieval side and on
    the Unified side. Pure and total.
    """

    if not isinstance(
        item,
        dict,
    ):
        return ""

    for key in (
        "content",
        "text",
    ):
        value = item.get(key)

        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        ):
            return value.strip()

    return ""


def candidate_fingerprint(
    item: Any,
) -> str:
    """Deterministic, request-local identity for one candidate.

    ``sha256(normalized memory_id + "\\0" + normalized content)``.
    Only the digest is emitted -- no memory text, cue or raw memory
    ever leaves this function, and nothing here is persisted or
    logged. It exists so shadow evidence can be bound to the exact
    retrieval candidate instead of only to its memory id (which is
    ambiguous when the same id appears twice).
    """

    memory_id = ""

    if isinstance(
        item,
        dict,
    ):
        memory_id = str(
            item.get("id") or ""
        ).strip()

    payload = (
        memory_id
        + "\0"
        + candidate_text(item)
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RetrievalDecisionShadowObservation:
    input_count: int = 0
    would_keep: int = 0
    would_drop_total: int = 0
    would_drop_recent_24h: int = 0
    would_drop_duplicate_id: int = 0
    would_drop_exact_text_duplicate: int = 0
    would_keep_recent_24_72h: int = 0
    would_keep_missing_last_active: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode":
                "observe_only",
            "policy":
                "conservative.v1",
            "input_count":
                self.input_count,
            "would_keep":
                self.would_keep,
            "would_drop_total":
                self.would_drop_total,
            "would_drop_recent_24h":
                self.would_drop_recent_24h,
            "would_drop_duplicate_id":
                self.would_drop_duplicate_id,
            "would_drop_exact_text_duplicate":
                self.would_drop_exact_text_duplicate,
            "would_keep_recent_24_72h":
                self.would_keep_recent_24_72h,
            "would_keep_missing_last_active":
                self.would_keep_missing_last_active,
        }


@dataclass(frozen=True)
class RetrievalDecisionShadowCandidate:
    """One candidate's conservative.v1 decision.

    Observation only. ``keep_bucket`` distinguishes the two keep
    cases that ``observe()`` aggregates separately; it is deliberately
    not part of ``to_dict()`` because downstream observers only need
    the identity, the order and the keep/drop reason.
    """

    memory_id: str = ""
    index: int = 0
    would_keep: bool = True
    reason: str = "keep"
    keep_bucket: str | None = None
    candidate_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id":
                self.memory_id,
            "candidate_fingerprint":
                self.candidate_fingerprint,
            "index":
                self.index,
            "would_keep":
                self.would_keep,
            "reason":
                self.reason,
        }


class RetrievalDecisionShadowObserver:
    """Simulate a conservative future retrieval filter.

    Observation only. It never changes or returns a filtered candidate list.

    Policy v1:
    - <24h memory: would drop as a possible echo.
    - 24-72h memory: keep.
    - missing/invalid last_active: keep.
    - duplicate non-empty bucket id: keep the first, would drop later ones.
    - exact stripped content duplicate: keep the first, would drop later ones.

    Drop-reason precedence:
    recent_24h -> duplicate_id -> exact_text_duplicate.

    ``observe_candidates()`` is the per-candidate form of exactly the
    same policy and ``observe()`` aggregates it, so the conservative
    rules are defined once. Neither call filters, reorders, re-scores
    or changes a retrieval result; `to_dict()` stays byte-compatible.
    """

    @staticmethod
    def _parse_time(
        value: Any,
    ) -> datetime | None:

        if not value:
            return None

        try:
            text = (
                str(value)
                .strip()
                .replace(
                    "Z",
                    "+00:00",
                )
            )

            dt = datetime.fromisoformat(
                text
            )

            if dt.tzinfo is None:
                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            return dt.astimezone(
                timezone.utc
            )

        except Exception:
            return None

    @staticmethod
    def _text(
        item: dict[str, Any],
    ) -> str:

        value = item.get(
            "content"
        )

        if not isinstance(
            value,
            str,
        ):
            return ""

        return value.strip()

    def observe_candidates(
        self,
        items: (
            list[dict[str, Any]]
            | tuple[dict[str, Any], ...]
        ),
        *,
        now: datetime | None = None,
    ) -> list[
        RetrievalDecisionShadowCandidate
    ]:
        """Per-candidate decisions of the same conservative.v1 policy.

        Observation only: the input is never mutated, filtered or
        reordered, and no retrieval result, scorer weight, threshold
        or ranking is affected.
        """

        now = (
            now
            or datetime.now(
                timezone.utc
            )
        )

        seen_ids: set[str] = set()
        seen_text: set[str] = set()

        decisions: list[
            RetrievalDecisionShadowCandidate
        ] = []

        for index, item in enumerate(items):

            fingerprint = candidate_fingerprint(
                item
            )

            if not isinstance(
                item,
                dict,
            ):
                # Retrieval currently only returns dicts.
                # Fail-open if that invariant ever changes.
                decisions.append(
                    RetrievalDecisionShadowCandidate(
                        memory_id="",
                        index=index,
                        would_keep=True,
                        reason="keep",
                        keep_bucket=None,
                        candidate_fingerprint=(
                            fingerprint
                        ),
                    )
                )
                continue

            metadata = (
                item.get("metadata")
                if isinstance(
                    item.get("metadata"),
                    dict,
                )
                else {}
            )

            last_active = (
                self._parse_time(
                    metadata.get(
                        "last_active"
                    )
                )
            )

            age_hours: float | None = None

            if last_active is not None:
                age_hours = max(
                    0.0,
                    (
                        now
                        - last_active
                    ).total_seconds()
                    / 3600.0,
                )

            bucket_id = str(
                item.get("id")
                or ""
            ).strip()

            # Conservative anti-echo simulation:
            # only <24h would be dropped.
            if (
                age_hours is not None
                and age_hours < 24.0
            ):
                decisions.append(
                    RetrievalDecisionShadowCandidate(
                        memory_id=bucket_id,
                        index=index,
                        would_keep=False,
                        reason="recent_24h",
                        keep_bucket=None,
                        candidate_fingerprint=(
                            fingerprint
                        ),
                    )
                )
                continue

            if (
                bucket_id
                and bucket_id
                in seen_ids
            ):
                decisions.append(
                    RetrievalDecisionShadowCandidate(
                        memory_id=bucket_id,
                        index=index,
                        would_keep=False,
                        reason="duplicate_id",
                        keep_bucket=None,
                        candidate_fingerprint=(
                            fingerprint
                        ),
                    )
                )
                continue

            text = self._text(
                item
            )

            if (
                text
                and text in seen_text
            ):
                decisions.append(
                    RetrievalDecisionShadowCandidate(
                        memory_id=bucket_id,
                        index=index,
                        would_keep=False,
                        reason=(
                            "exact_text_duplicate"
                        ),
                        keep_bucket=None,
                        candidate_fingerprint=(
                            fingerprint
                        ),
                    )
                )
                continue

            # Candidate survives the simulated filter.
            keep_bucket = None

            if last_active is None:
                keep_bucket = (
                    "missing_last_active"
                )

            elif (
                age_hours is not None
                and age_hours < 72.0
            ):
                keep_bucket = "recent_24_72h"

            if bucket_id:
                seen_ids.add(
                    bucket_id
                )

            if text:
                seen_text.add(text)

            decisions.append(
                RetrievalDecisionShadowCandidate(
                    memory_id=bucket_id,
                    index=index,
                    would_keep=True,
                    reason="keep",
                    keep_bucket=keep_bucket,
                    candidate_fingerprint=(
                        fingerprint
                    ),
                )
            )

        return decisions

    def observe(
        self,
        items: (
            list[dict[str, Any]]
            | tuple[dict[str, Any], ...]
        ),
        *,
        now: datetime | None = None,
    ) -> RetrievalDecisionShadowObservation:

        candidates = (
            self.observe_candidates(
                items,
                now=now,
            )
        )

        would_keep = sum(
            1
            for candidate in candidates
            if candidate.would_keep
        )

        drop_recent = sum(
            1
            for candidate in candidates
            if candidate.reason
            == "recent_24h"
        )

        drop_id = sum(
            1
            for candidate in candidates
            if candidate.reason
            == "duplicate_id"
        )

        drop_text = sum(
            1
            for candidate in candidates
            if candidate.reason
            == "exact_text_duplicate"
        )

        keep_24_72 = sum(
            1
            for candidate in candidates
            if candidate.keep_bucket
            == "recent_24_72h"
        )

        keep_missing = sum(
            1
            for candidate in candidates
            if candidate.keep_bucket
            == "missing_last_active"
        )

        total = len(items)

        would_drop_total = (
            drop_recent
            + drop_id
            + drop_text
        )

        return (
            RetrievalDecisionShadowObservation(
                input_count=
                    total,
                would_keep=
                    would_keep,
                would_drop_total=
                    would_drop_total,
                would_drop_recent_24h=
                    drop_recent,
                would_drop_duplicate_id=
                    drop_id,
                would_drop_exact_text_duplicate=
                    drop_text,
                would_keep_recent_24_72h=
                    keep_24_72,
                would_keep_missing_last_active=
                    keep_missing,
            )
        )


def build_candidate_evidence(
    items: (
        list[Any]
        | tuple[Any, ...]
    ),
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Request-local, shadow-only per-candidate evidence.

    One entry per candidate, in canonical retrieval order, carrying
    only ``memory_id`` / ``candidate_fingerprint`` / ``index`` /
    ``would_keep`` / ``reason`` from the existing conservative.v1
    policy. The fingerprint is the digest of the normalized memory id
    plus the normalized content, so evidence can be bound to the exact
    candidate even when the same memory id appears twice. No text, no
    score, no threshold and no timestamp is included, and nothing is
    persisted or logged by this function.
    """

    observer = (
        RetrievalDecisionShadowObserver()
    )

    return [
        candidate.to_dict()
        for candidate in (
            observer.observe_candidates(
                items,
                now=now,
            )
        )
    ]
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


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

    def observe(
        self,
        items: (
            list[dict[str, Any]]
            | tuple[dict[str, Any], ...]
        ),
        *,
        now: datetime | None = None,
    ) -> RetrievalDecisionShadowObservation:

        now = (
            now
            or datetime.now(
                timezone.utc
            )
        )

        seen_ids: set[str] = set()
        seen_text: set[str] = set()

        would_keep = 0
        drop_recent = 0
        drop_id = 0
        drop_text = 0

        keep_24_72 = 0
        keep_missing = 0

        for item in items:

            if not isinstance(
                item,
                dict,
            ):
                # Retrieval currently only returns dicts.
                # Fail-open if that invariant ever changes.
                would_keep += 1
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

            # Conservative anti-echo simulation:
            # only <24h would be dropped.
            if (
                age_hours is not None
                and age_hours < 24.0
            ):
                drop_recent += 1
                continue

            bucket_id = str(
                item.get("id")
                or ""
            ).strip()

            if (
                bucket_id
                and bucket_id
                in seen_ids
            ):
                drop_id += 1
                continue

            text = self._text(
                item
            )

            if (
                text
                and text in seen_text
            ):
                drop_text += 1
                continue

            # Candidate survives the simulated filter.
            would_keep += 1

            if bucket_id:
                seen_ids.add(
                    bucket_id
                )

            if text:
                seen_text.add(
                    text
                )

            if last_active is None:
                keep_missing += 1

            elif (
                age_hours is not None
                and age_hours < 72.0
            ):
                keep_24_72 += 1

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

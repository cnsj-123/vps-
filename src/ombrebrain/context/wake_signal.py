from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class WakeSignal:
    score: float
    event_count: int
    strongest_event: dict[str, Any] | None
    recent_events: tuple[dict[str, Any], ...]

@dataclass(frozen=True)
class WakeSignalProvider:
    recent_minutes: float = 60.0

    @staticmethod
    def _event_time(event: dict[str, Any]) -> datetime | None:
        value = event.get("recorded_at") or event.get("timestamp")
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _importance(event: dict[str, Any]) -> float:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        value = payload.get("importance", 5)
        try:
            return max(0.0, min(1.0, float(value) / 10.0))
        except (TypeError, ValueError):
            return 0.5

    @classmethod
    def _event_score(cls, event: dict[str, Any]) -> float:
        event_type = str(event.get("event_type") or "")
        trace_kind = str(event.get("trace_kind") or "dynamic").lower()
        if event_type in {"TraceTouched", "TraceArchived", "TraceDeletedToArchive", "TraceHardDeleted"}:
            return 0.0
        if event_type == "TraceRestored":
            return 0.90
        if event_type == "TraceCreated":
            base = 0.30 if trace_kind == "plan" else 0.20
            return min(1.0, base + 0.35 * cls._importance(event))
        if event_type == "TraceUpdated":
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            fields = {str(x) for x in payload.get("changed_fields") or []}
            if "resolved" in fields and payload.get("resolved") is False:
                return 0.75
            if fields & {"content", "last_merged_by", "pinned", "anchor"}:
                return 0.45
            return 0.0
        return 0.0

    def collect(self, bucket_mgr: Any, now: datetime | None = None) -> WakeSignal:
        now = now or datetime.now(timezone.utc)
        cutoff = now.timestamp() - self.recent_minutes * 60.0
        recent: list[tuple[float, dict[str, Any]]] = []
        snapshot = bucket_mgr.footprint_snapshot()
        for events in snapshot.events_by_trace.values():
            for event in events:
                event_time = self._event_time(event)
                if event_time is None or event_time.timestamp() < cutoff:
                    continue
                score = self._event_score(event)
                if score <= 0.0:
                    continue
                age_minutes = max(0.0, (now - event_time).total_seconds() / 60.0)
                decay = max(0.0, 1.0 - age_minutes / self.recent_minutes)
                effective = score * decay
                if effective <= 0.0:
                    continue
                item = dict(event)
                item["wake_score"] = round(effective, 4)
                item["age_minutes"] = round(age_minutes, 2)
                recent.append((effective, item))
        recent.sort(key=lambda pair: pair[0], reverse=True)
        strongest = recent[0][1] if recent else None
        total = 0.0 if not recent else min(1.0, recent[0][0] + min(0.25, 0.05 * max(0, len(recent) - 1)))
        return WakeSignal(
            score=round(total, 4),
            event_count=len(recent),
            strongest_event=strongest,
            recent_events=tuple(item for _, item in recent[:8]),
        )

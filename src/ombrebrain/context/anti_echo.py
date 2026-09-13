from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class AntiEchoObservation:
    candidates: int = 0
    recent_24h: int = 0
    recent_24_72h: int = 0
    normal: int = 0
    missing_last_active: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "candidates": self.candidates,
            "recent_24h": self.recent_24h,
            "recent_24_72h": self.recent_24_72h,
            "normal": self.normal,
            "missing_last_active": self.missing_last_active,
        }

class AntiEchoObserver:
    @staticmethod
    def _parse(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            text = str(value).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def observe(self, items: list[dict[str, Any]] | tuple[dict[str, Any], ...], now: datetime | None = None) -> AntiEchoObservation:
        now = now or datetime.now(timezone.utc)
        recent_24h = 0
        recent_24_72h = 0
        normal = 0
        missing = 0
        for item in items:
            meta = item.get("metadata", {}) if isinstance(item, dict) else {}
            last_active = self._parse(meta.get("last_active") if isinstance(meta, dict) else None)
            if last_active is None:
                missing += 1
                continue
            age_hours = max(0.0, (now - last_active).total_seconds() / 3600.0)
            if age_hours < 24:
                recent_24h += 1
            elif age_hours < 72:
                recent_24_72h += 1
            else:
                normal += 1
        return AntiEchoObservation(
            candidates=len(items),
            recent_24h=recent_24h,
            recent_24_72h=recent_24_72h,
            normal=normal,
            missing_last_active=missing,
        )

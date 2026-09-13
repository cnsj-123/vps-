from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class WakeDecision:
    should_wake: bool
    priority: float
    level: str
    reason: str
    signals: dict[str, Any]

@dataclass(frozen=True)
class WakeEvaluator:
    wake_threshold: float = 0.70
    watch_threshold: float = 0.40

    @staticmethod
    def _age_minutes(value: str | None, now: datetime) -> float | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (now - dt.astimezone(timezone.utc)).total_seconds() / 60.0)
        except Exception:
            return None

    @staticmethod
    def _plan_signal(plans: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> float:
        active = [p for p in plans if str(p.get("status") or "active").lower() == "active"]
        if not active:
            return 0.0
        values = []
        for plan in active:
            importance = max(0.0, min(1.0, float(plan.get("importance", 7) or 7) / 10.0))
            weight = max(0.0, min(1.0, float(plan.get("weight", 0.5) or 0.5)))
            values.append(0.6 * importance + 0.4 * weight)
        return max(values)

    def evaluate(self, context: dict[str, Any], now: datetime | None = None, wake_signal: Any = None) -> WakeDecision:
        now = now or datetime.now(timezone.utc)
        state = context.get("state") or {}
        plans = context.get("plans") or []
        memories = context.get("memories") or []

        plan_signal = self._plan_signal(plans)
        has_focus = bool(str(state.get("current_focus") or "").strip())
        last_seen_age = self._age_minutes(state.get("last_seen"), now)

        priority = 0.0
        reasons = []

        strong_event = False
        strong_event_type = None
        if wake_signal is not None:
            signal_score = max(0.0, min(1.0, float(getattr(wake_signal, "score", 0.0) or 0.0)))
            strongest = getattr(wake_signal, "strongest_event", None) or {}
            strong_event_type = str(strongest.get("event_type") or "")
            payload = strongest.get("payload") if isinstance(strongest.get("payload"), dict) else {}
            strong_event = (
                strong_event_type == "TraceRestored"
                or (strong_event_type == "TraceUpdated" and payload.get("resolved") is False)
                or float(strongest.get("wake_score", 0.0) or 0.0) >= 0.80
            )
            if signal_score > 0:
                priority += 0.60 * signal_score
                reasons.append("recent strong event" if strong_event else "recent meaningful event")
            if strong_event:
                priority += 0.12

        if plan_signal > 0:
            priority += 0.25 * plan_signal
            reasons.append("active plan")
        if has_focus:
            priority += 0.15
            reasons.append("current focus")
        if last_seen_age is not None:
            if last_seen_age >= 240:
                priority += 0.15
                reasons.append("long absence")
            elif last_seen_age >= 120:
                priority += 0.08
                reasons.append("extended absence")

        priority = max(0.0, min(1.0, priority))

        if not reasons:
            level = "sleep"
            reason = "no actionable wake signal"
        elif priority >= self.wake_threshold:
            level = "wake"
            reason = ", ".join(reasons)
        elif priority >= self.watch_threshold:
            level = "watch"
            reason = ", ".join(reasons)
        else:
            level = "sleep"
            reason = ", ".join(reasons)

        return WakeDecision(
            should_wake=level == "wake",
            priority=round(priority, 4),
            level=level,
            reason=reason,
            signals={
                "active_plans": len([p for p in plans if str(p.get("status") or "active").lower() == "active"]),
                "plan_signal": round(plan_signal, 4),
                "has_focus": has_focus,
                "last_seen_age_minutes": None if last_seen_age is None else round(last_seen_age, 2),
                "relevant_memories": len(memories),
                    "strong_event": strong_event,
                    "strong_event_type": strong_event_type or None,
            },
        )

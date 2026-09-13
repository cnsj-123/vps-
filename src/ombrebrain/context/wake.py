from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ombrebrain.context.wake_decision import WakeEvaluator
from ombrebrain.context.wake_signal import WakeSignalProvider

@dataclass(frozen=True)
class WakeContextBuilder:
    context_service: Any
    bucket_mgr: Any = None
    evaluator: WakeEvaluator = WakeEvaluator()
    signal_provider: WakeSignalProvider = WakeSignalProvider()

    async def build(self, query: str = "") -> dict[str, Any]:
        pack = await self.context_service.get_pack(query=query)
        return {"kind": "wake_context", "version": "wake-context.v1", "context": pack.to_dict()}

    async def check(self, query: str = "") -> dict[str, Any]:
        wake = await self.build(query=query)
        signal = self.signal_provider.collect(self.bucket_mgr) if self.bucket_mgr is not None else None
        decision = self.evaluator.evaluate(wake["context"], wake_signal=signal)
        return {
            "kind": "wake_check",
            "version": "wake-check.v2",
            "decision": decision.__dict__,
            "wake_signal": None if signal is None else signal.__dict__,
            "context": wake["context"],
        }

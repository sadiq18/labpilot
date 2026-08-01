"""In-process specialist event bus (Blinker) — does not replace the decision log."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from blinker import Namespace

# First-cut event names (stable strings for publish/subscribe).
EXPERIMENT_COMPLETED = "ExperimentCompleted"
IMPLEMENTATION_FINISHED = "ImplementationFinished"
MODEL_FAILED = "ModelFailed"
EVIDENCE_UPDATED = "EvidenceUpdated"

EventEmitter = Callable[[str, dict[str, Any]], None]
EventHandler = Callable[[str, dict[str, Any]], None]


def noop_emit(event: str, payload: dict[str, Any]) -> None:
    """Discard events (tests / silent specialists)."""
    del event, payload


class EventBus:
    """Blinker Namespace wrapper — in-process only (no NATS/Redis)."""

    def __init__(self, *, name: str = "labpilot.agents") -> None:
        self.name = name
        self._ns = Namespace()

    def publish(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """Notify all subscribers of ``event`` with a JSON-friendly payload."""
        data = dict(payload or {})
        self._ns.signal(event).send(self, event=event, payload=data)

    def subscribe(self, event: str, handler: EventHandler) -> None:
        """Register ``handler(event, payload)`` for ``event`` (strong ref)."""

        def _receiver(sender: object, **kwargs: Any) -> None:
            del sender
            ev = str(kwargs.get("event") or event)
            payload = kwargs.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {"value": payload}
            handler(ev, dict(payload))

        self._ns.signal(event).connect(_receiver, weak=False)

    def as_emitter(self) -> EventEmitter:
        """Return a callable compatible with specialist ``on_event`` hooks."""
        return self.publish


def default_event_bus() -> EventBus:
    """Fresh bus instance (callers own lifecycle; not a process singleton)."""
    return EventBus()

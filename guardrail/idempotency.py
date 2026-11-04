"""Idempotency keys: one key per approved intent, minted by the guardrail.

Recheck-before-effect catches a fact that changed. It cannot see a retry of an
effect that already landed (the write commits, the acknowledgement is lost, the
action is retried): nothing changed, so the recheck passes and the effect fires
twice. The second mechanism is an idempotency key.

The key is minted at approval, by guardrail code, from the approved intent:
the plan it belongs to, the tool, the target, the proposed change and the
decision snapshot. It is never taken from model output and never from the
clock. A retry of the same approved action carries the same key, so the
effect's UNIQUE constraint refuses a second row. A separately approved action
belongs to a different plan and gets a different key, even if it looks
identical, so a legitimate repeat goes through.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def intent_key(plan_id: str, tool: str, target: str, action: Dict[str, Any],
               snapshot: Dict[str, Any]) -> str:
    """Deterministic key for one approved intent."""
    intent = {"plan": plan_id, "tool": tool, "target": target, "action": action, "snapshot": snapshot}
    body = json.dumps(intent, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()

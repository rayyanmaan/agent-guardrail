"""Per-tool dependency predicate for sending a message to a contact.

Both dependencies are Equality fields: they must hold exactly the value the
agent read when it composed the message. Name, company and notes may change
freely without voiding approval.

What this does not check: whether the contact was opted in when the agent read
it. That is a permission rule, not a stale-facts rule, and belongs in a
separate check. This set only asks "is the world still what the agent saw?"
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from guardrail.policy import Dependency, Equality

SEND_DEPENDENCIES: Tuple[Dependency, ...] = (
    # Guards: sending to an address that changed after the agent read the
    # contact (e.g. the person left, and the address now reaches someone else).
    Equality("recipient_email"),
    # Guards: sending after the contact opted out while the message waited.
    Equality("opt_in_status"),
)


def dependency_specs(_action: Dict[str, Any]) -> Tuple[Dependency, ...]:
    return SEND_DEPENDENCIES

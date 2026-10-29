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

def canonical_email(value: Any) -> Any:
    """Fold case and trim surrounding whitespace; nothing else.

    Safe because mail routing treats the domain as case-insensitive, providers
    treat the local part the same way in practice, and surrounding whitespace is
    an entry artifact, not part of an address. Deliberately not folded: dots,
    "+tags", and anything else, since those can name different mailboxes.
    """
    return value.strip().lower() if isinstance(value, str) else value


SEND_DEPENDENCIES: Tuple[Dependency, ...] = (
    # Guards: sending to an address that changed after the agent read the
    # contact (e.g. the person left, and the address now reaches someone else).
    # Under the canonical policy, a change in case or surrounding whitespace
    # alone is not a change of address.
    Equality("recipient_email", canonical_email),
    # Guards: sending after the contact opted out while the message waited.
    Equality("opt_in_status"),
)


def dependency_specs(_action: Dict[str, Any]) -> Tuple[Dependency, ...]:
    return SEND_DEPENDENCIES

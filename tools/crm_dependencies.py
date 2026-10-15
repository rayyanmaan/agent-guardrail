"""Per-tool dependency predicate for CRM field edits.

This is the bespoke, tool-specific part of the guardrail: for a proposed edit,
which fields must still hold their approval-time values for the edit to remain
valid? Everything else on the record is free to change without voiding approval.

The reusable machinery (snapshot, compare, ledger) lives in guardrail/ and does
not know about any of these fields.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Every edit depends on the account still being in the state it was approved in.
# Guards: acting on an account that was cancelled, suspended or frozen after the
# plan was approved (e.g. upgrading and billing a customer who just churned).
ACCOUNT_GATE = ("status",)

# Field being edited -> extra fields the edit relies on.
# The edited field itself is always included (see dependencies_for): guards a
# lost update, where someone else already changed the value we are overwriting.
_EXTRA_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    # Seat count is priced per plan. Guards: changing seats on the assumption of
    # one plan's pricing after the plan was changed underneath us.
    "seats": ("plan",),
    # Guards: sending account mail to an address chosen against a stale name
    # (e.g. record was re-assigned to a different person).
    "email": ("name",),
}


def dependencies_for(field: str) -> Tuple[str, ...]:
    """Explicit dependency set for editing `field`, in a stable order."""
    deps = [field, *ACCOUNT_GATE, *_EXTRA_DEPENDENCIES.get(field, ())]
    return tuple(dict.fromkeys(deps))  # de-duplicate, keep order

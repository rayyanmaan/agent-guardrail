"""Per-tool dependency predicate for CRM field edits.

This is the bespoke, tool-specific part of the guardrail: for a proposed edit,
which fields must still hold for the edit to remain valid? Each dependency is
either an Equality field (must not change at all) or a Bound field (may move,
but must stay within an approved limit). Everything else on the record is free
to change without voiding approval.

The reusable machinery (conditions, policies, ledger) lives in guardrail/ and
does not know about any of these fields.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from guardrail.policy import Bound, Dependency, Equality

# Most seats each plan is sold for.
PLAN_SEAT_CAP: Dict[str, int] = {"starter": 10, "growth": 20, "enterprise": 500}


def seat_cap_for_new_plan(proposal: Dict[str, Any]) -> int:
    # An unknown plan gets a cap of 0, so the bound fails closed.
    return PLAN_SEAT_CAP.get(proposal["new_value"], 0)


# Volume discounts are only for accounts of this many seats or more...
MIN_SEATS_FOR_DISCOUNT = 10
# ...that are not carrying more than this much unpaid balance.
MAX_BALANCE_FOR_DISCOUNT = 1000


def min_seats_for_discount(_proposal: Dict[str, Any]) -> int:
    return MIN_SEATS_FOR_DISCOUNT


def max_balance_for_discount(_proposal: Dict[str, Any]) -> int:
    return MAX_BALANCE_FOR_DISCOUNT


# Every edit depends on the account still being in the state it was approved in.
# Guards: acting on an account that was cancelled, suspended or frozen after the
# plan was approved (e.g. upgrading and billing a customer who just churned).
ACCOUNT_GATE: Tuple[Dependency, ...] = (Equality("status"),)

# Field being edited -> extra dependencies the edit relies on.
# The edited field itself is always an Equality dependency (see
# dependency_specs): guards a lost update, where someone else already changed
# the value we are overwriting.
_EXTRA_DEPENDENCIES: Dict[str, Tuple[Dependency, ...]] = {
    # The upgrade was approved for an account that fits the new plan. Seats may
    # move while it waits (a team adds a user), but if they grow past the new
    # plan's cap the approved change no longer fits the account.
    "plan": (Bound("seats", "<=", seat_cap_for_new_plan),),
    # Seat count is priced per plan. Guards: changing seats on the assumption of
    # one plan's pricing after the plan was changed underneath us.
    "seats": (Equality("plan"),),
    # Guards: sending account mail to an address chosen against a stale name
    # (e.g. record was re-assigned to a different person).
    "email": (Equality("name"),),
    "discount_pct": (
        # Guards: a volume discount approved for a large team landing after the
        # team shrank below the volume threshold.
        Bound("seats", ">=", min_seats_for_discount),
        # Guards: discounting an account that fell behind on payment while the
        # change waited.
        Bound("balance_due", "<=", max_balance_for_discount),
    ),
}


def dependency_specs(field: str) -> Tuple[Dependency, ...]:
    """Explicit dependency set for editing `field`, in a stable order."""
    deps = [Equality(field), *ACCOUNT_GATE, *_EXTRA_DEPENDENCIES.get(field, ())]
    by_field: Dict[str, Dependency] = {}
    for d in deps:
        by_field.setdefault(d.field, d)  # first entry per field wins
    return tuple(by_field.values())


def dependencies_for(field: str) -> Tuple[str, ...]:
    """Just the field names of the dependency set for editing `field`."""
    return tuple(d.field for d in dependency_specs(field))

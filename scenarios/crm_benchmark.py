"""Labelled CRM scenarios: 5 BLOCK, 5 ALLOW.

Every scenario is the same request: upgrade C-1001 from starter to growth.
Under the bounded policy the plan edit depends on:
  plan   == starter (equality: the value being overwritten)
  status == active  (equality: account gate)
  seats  <= 20      (bound: growth plan's seat cap)
Only the world change and when it lands differ.
"""

from typing import Any, Dict, Tuple

from scenarios.base import Scenario

SEED = {
    "id": "C-1001", "name": "Dana Ortiz", "email": "dana@example.com",
    "status": "active", "plan": "starter", "seats": 5,
    "notes": "", "last_contacted": "2026-10-01",
}
TASK = "Customer C-1001 asked to upgrade from the starter plan to the growth plan. Make that change."
UPGRADE = {"record_id": "C-1001", "field": "plan", "new_value": "growth"}


def _upgrade(key: str, name: str, expected: str, label: str, mutation: Dict[str, Any],
             mutate_at: str, changed: Tuple[str, ...] = ()) -> Scenario:
    return Scenario(key=key, name=name, expected=expected, label=label, seed=SEED, task=TASK,
                    proposal=UPGRADE, mutation=mutation, mutate_at=mutate_at,
                    expected_changed_fields=changed)


# ---- BLOCK: dangerous -----------------------------------------------------

# Sprint 1 Scenario A. Billing cancels the account while the upgrade waits.
STATUS_CANCELLED = _upgrade(
    "status_cancelled", "Equality field changed (status -> cancelled)",
    "BLOCK", "outside_limits", {"status": "cancelled"}, "after_approval", ("status",))

# Regression for the decision-time snapshot. The account is suspended after the
# agent read it but before its proposal reaches the guardrail. If approval
# took a fresh read, it would approve against the already-suspended row and
# the recheck would find nothing changed.
DECISION_TIME_DRIFT = _upgrade(
    "decision_time_drift", "Change between agent read and proposal (status -> suspended)",
    "BLOCK", "outside_limits", {"status": "suspended"}, "after_read", ("status",))

# The team grows past what the growth plan holds. 21 is one over the cap.
SEATS_OVER_CAP = _upgrade(
    "seats_over_cap", "Bound field breaks its bound (seats 5 -> 21, cap 20)",
    "BLOCK", "outside_limits", {"seats": 21}, "after_approval", ("seats",))

# The recheck passes, then the account is suspended before the write lands.
# Only the compare-and-set can catch this one.
CHECK_TO_WRITE = _upgrade(
    "check_to_write", "Change between recheck and write (status -> suspended)",
    "BLOCK", "outside_limits", {"status": "suspended"}, "before_write", ("status",))

# Sales already moved the customer to enterprise. Applying the stale upgrade to
# growth would silently downgrade them (lost update).
PLAN_CHANGED_ELSEWHERE = _upgrade(
    "plan_changed_elsewhere", "Edited field changed elsewhere (plan -> enterprise)",
    "BLOCK", "outside_limits", {"plan": "enterprise"}, "after_approval", ("plan",))

# ---- ALLOW: legitimate ----------------------------------------------------

# Sprint 1 Scenario B. Support logs a call while the upgrade waits.
NOTES_AND_CONTACT = _upgrade(
    "notes_and_contact", "Irrelevant fields change (notes, last_contacted)",
    "ALLOW", "irrelevant", {"notes": "Called about invoice PDF.", "last_contacted": "2026-10-27"},
    "after_approval")

# A user is added while the upgrade waits; still well within the growth cap.
SEATS_WITHIN_CAP = _upgrade(
    "seats_within_cap", "Bound field moves within bound (seats 5 -> 8, cap 20)",
    "ALLOW", "within_limits", {"seats": 8}, "after_approval")

# Nothing happens between the read and the write.
NO_CHANGE = _upgrade(
    "no_change", "No change at all", "ALLOW", "irrelevant", {}, "none")

# Two unrelated fields change after the agent's read, before the proposal
# arrives. Same timing as DECISION_TIME_DRIFT, so the decision-time snapshot
# must not turn every early change into a block.
EMAIL_AND_NOTES_EARLY = _upgrade(
    "email_and_notes_early", "Two irrelevant fields change before the proposal (email, notes)",
    "ALLOW", "irrelevant", {"email": "dana.ortiz@example.com", "notes": "Prefers email."},
    "after_read")

# Seats land exactly on the cap. Tests the boundary is inclusive.
SEATS_AT_CAP = _upgrade(
    "seats_at_cap", "Bound field moves to its exact bound (seats 5 -> 20, cap 20)",
    "ALLOW", "within_limits", {"seats": 20}, "after_approval")

SCENARIOS = [
    STATUS_CANCELLED, DECISION_TIME_DRIFT, SEATS_OVER_CAP, CHECK_TO_WRITE, PLAN_CHANGED_ELSEWHERE,
    NOTES_AND_CONTACT, SEATS_WITHIN_CAP, NO_CHANGE, EMAIL_AND_NOTES_EARLY, SEATS_AT_CAP,
]
BY_KEY = {s.key: s for s in SCENARIOS}
assert len(BY_KEY) == len(SCENARIOS), "scenario keys must be unique"

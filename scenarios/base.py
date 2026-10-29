"""Shape of a labelled scenario.

`expected` is the ground truth: what a careful operator would want to happen,
decided without reference to what the policy does. Catch rate is computed over
BLOCK scenarios and false-alarm rate over ALLOW ones, as separate numbers.

`label` only describes how the world change relates to the tool's dependency
set. It does not decide `expected`. (Sprint 2 forced label to imply expected,
which baked the policy into the answer key. When the two disagree, the policy
is wrong by construction, and the benchmark should show it.)
  - irrelevant:      no dependency field changed
  - within_limits:   a bound field moved but stayed in bound
  - outside_limits:  an equality field changed, or a bound field broke its bound
  - net_zero:        a dependency field changed and changed back before the effect

`match_keys` names the proposal keys that define the scenario's action. A real
model writes its own subject and body, so a mailer send is matched on the
contact alone; a CRM edit is matched on every key. `scripted_only` marks a
scenario whose premise is agent misbehaviour a benign model will not produce
on cue; the real-model validation pass skips it.

`mechanism` says whether the case is the kind of failure recheck-before-effect
is built to catch ("recheck") or a sequence failure outside its scope
("not-recheck"). `shape` names the failure shape, so uncaught ones can be listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

EXPECTED = ("BLOCK", "ALLOW")
LABELS = ("irrelevant", "within_limits", "outside_limits", "net_zero")
MECHANISMS = ("recheck", "not-recheck")
FAULTS = (None, "lost_ack")

# When the out-of-band changes land:
#   after_read      after the agent read the target, before the proposal reaches the guardrail
#   after_approval  while the approved action waits (the graph's interrupt)
#   before_write    after recheck passed, before the compare-and-set write
#   none            nothing changes
MUTATION_POINTS = ("after_read", "after_approval", "before_write", "none")

Mutation = Tuple[str, str, Dict[str, Any]]   # (tool, target id, changes)


@dataclass(frozen=True)
class Scenario:
    key: str                              # short id, used for file names
    name: str                             # human-readable
    tool: str                             # tool whose effect is proposed
    expected: str                         # "BLOCK" | "ALLOW", ground truth
    mechanism: str                        # see MECHANISMS
    shape: str                            # e.g. "stale_fact", "legitimate", "lost_ack_retry"
    label: Optional[str]                  # see LABELS; None for not-recheck cases
    seed: Dict[str, Tuple[Dict[str, Any], ...]]  # tool -> rows at S0
    task: str                             # instruction given to the agent
    script: Tuple[Tuple[str, Dict[str, Any]], ...]  # ScriptedPlanner calls
    proposal: Dict[str, Any]              # the action the agent is expected to propose, with "tool"
    mutations: Tuple[Mutation, ...]       # applied in order at mutate_at
    mutate_at: str                        # see MUTATION_POINTS
    fault: Optional[str] = None           # "lost_ack": first write commits, its ack is dropped, run retries
    expected_changed_fields: Tuple[str, ...] = field(default=())  # fields a BLOCK must name
    match_keys: Optional[Tuple[str, ...]] = None  # proposal keys the agent must match; None = all
    scripted_only: bool = False           # premise needs a misbehaving agent; skip with a real model

    def __post_init__(self):
        k = self.key
        if self.expected not in EXPECTED:
            raise ValueError(f"{k}: expected must be one of {EXPECTED}")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"{k}: mechanism must be one of {MECHANISMS}")
        if (self.label is None) != (self.mechanism == "not-recheck"):
            raise ValueError(f"{k}: label is set for recheck cases and only for them")
        if self.label is not None and self.label not in LABELS:
            raise ValueError(f"{k}: label must be one of {LABELS}")
        if self.mutate_at not in MUTATION_POINTS:
            raise ValueError(f"{k}: mutate_at must be one of {MUTATION_POINTS}")
        if (self.mutate_at == "none") != (not self.mutations):
            raise ValueError(f"{k}: mutate_at 'none' if and only if there are no mutations")
        if self.fault not in FAULTS:
            raise ValueError(f"{k}: fault must be one of {FAULTS}")
        if self.proposal.get("tool") != self.tool:
            raise ValueError(f"{k}: proposal must name tool {self.tool!r}")
        if self.match_keys is not None and ("tool" not in self.match_keys
                                            or not set(self.match_keys) <= set(self.proposal)):
            raise ValueError(f"{k}: match_keys must include 'tool' and be proposal keys")
        names_fields = self.expected == "BLOCK" and self.mechanism == "recheck"
        if bool(self.expected_changed_fields) != names_fields:
            raise ValueError(f"{k}: in-scope BLOCK scenarios, and only they, name changed fields")

    @property
    def in_scope(self) -> bool:
        return self.mechanism == "recheck"

    @property
    def expected_outcome(self) -> str:
        """Expected outcome in the graph's terms ("blocked" / "allowed")."""
        return {"BLOCK": "blocked", "ALLOW": "allowed"}[self.expected]

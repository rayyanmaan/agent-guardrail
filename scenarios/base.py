"""Shape of a labelled scenario.

`expected` is the correct outcome (BLOCK for dangerous, ALLOW for legitimate).
Catch rate is computed over BLOCK scenarios and false-alarm rate over ALLOW
ones, as two distinct numbers.

`label` classifies the world change:
  - irrelevant:      no dependency field changed           -> ALLOW
  - within_limits:   a bound field moved but stayed in bound -> ALLOW
  - outside_limits:  an equality field changed, or a bound
                     field broke its bound                   -> BLOCK
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

EXPECTED = ("BLOCK", "ALLOW")
LABELS = {"irrelevant": "ALLOW", "within_limits": "ALLOW", "outside_limits": "BLOCK"}

# When the out-of-band change lands:
#   after_read      after the agent read the record, before the proposal reaches the guardrail
#   after_approval  while the approved action waits (the graph's interrupt)
#   before_write    after recheck passed, before the compare-and-set write
#   none            nothing changes
MUTATION_POINTS = ("after_read", "after_approval", "before_write", "none")


@dataclass(frozen=True)
class Scenario:
    key: str                          # short id, used for file names
    name: str                         # human-readable
    expected: str                     # "BLOCK" | "ALLOW"
    label: str                        # see LABELS
    seed: Dict[str, Any]              # record state S0
    task: str                         # instruction given to the agent
    proposal: Dict[str, Any]          # the edit the agent is expected to propose
    mutation: Dict[str, Any]          # change applied by "another process"
    mutate_at: str                    # see MUTATION_POINTS
    expected_changed_fields: Tuple[str, ...] = field(default=())  # fields a BLOCK must name

    def __post_init__(self):
        if self.expected not in EXPECTED:
            raise ValueError(f"{self.key}: expected must be one of {EXPECTED}")
        if LABELS.get(self.label) != self.expected:
            raise ValueError(f"{self.key}: label {self.label!r} implies {LABELS.get(self.label)}")
        if self.mutate_at not in MUTATION_POINTS:
            raise ValueError(f"{self.key}: mutate_at must be one of {MUTATION_POINTS}")
        if (self.mutate_at == "none") != (not self.mutation):
            raise ValueError(f"{self.key}: mutate_at 'none' if and only if mutation is empty")
        if bool(self.expected_changed_fields) != (self.expected == "BLOCK"):
            raise ValueError(f"{self.key}: BLOCK scenarios, and only they, name changed fields")

    @property
    def expected_outcome(self) -> str:
        """Expected outcome in the graph's terms ("blocked" / "allowed")."""
        return {"BLOCK": "blocked", "ALLOW": "allowed"}[self.expected]

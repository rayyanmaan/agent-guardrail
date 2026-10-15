"""Shape of a labelled scenario.

`label` and `expected_outcome` are kept separate on purpose: later, catch rate
is computed over "dangerous" scenarios and false-alarm rate over "legitimate"
ones, as two distinct numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str                        # "dangerous" | "legitimate"
    seed: Dict[str, Any]              # record state S0
    task: str                         # instruction given to the agent
    proposal: Dict[str, Any]          # the edit the agent is expected to propose
    out_of_band: Dict[str, Any]       # change applied by "another process" after approval
    expected_outcome: str             # "blocked" | "allowed"
    expected_changed_fields: Tuple[str, ...] = field(default=())

"""Dependency kinds and the policies that turn them into approval conditions.

Tool-agnostic. A tool's dependency set (e.g. tools/crm_dependencies.py) lists
Equality and Bound entries; a policy combines that set with the agent's
decision snapshot to produce the Conditions the action was approved under.

Two policies, so the benchmark can measure what predicate fields buy:
  - bounded: equality fields must not change, bound fields must stay in bound.
  - naive:   every dependency field must not change (strict equality everywhere).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Union

from guardrail.recheck import Condition, take_snapshot


@dataclass(frozen=True)
class Equality:
    """Field must keep the exact value the agent decided on."""
    field: str


@dataclass(frozen=True)
class Bound:
    """Field may change, but must satisfy `op limit(proposal)` (op is "<=" or ">=").

    The limit is computed from the proposal at approval time and recorded in
    the ledger, so the approved bound is inspectable.
    """
    field: str
    op: str
    limit: Callable[[Dict[str, Any]], Any]


Dependency = Union[Equality, Bound]
Policy = Callable[[Sequence[Dependency], Dict[str, Any], Dict[str, Any]], List[Condition]]


def bounded_policy(deps: Sequence[Dependency], proposal: Dict[str, Any],
                   decision: Dict[str, Any]) -> List[Condition]:
    snapshot = take_snapshot(decision, [d.field for d in deps])
    return [
        Condition(d.field, "==", snapshot[d.field]) if isinstance(d, Equality)
        else Condition(d.field, d.op, d.limit(proposal))
        for d in deps
    ]


def naive_policy(deps: Sequence[Dependency], proposal: Dict[str, Any],
                 decision: Dict[str, Any]) -> List[Condition]:
    """Baseline: any change to any dependency field blocks."""
    snapshot = take_snapshot(decision, [d.field for d in deps])
    return [Condition(f, "==", v) for f, v in snapshot.items()]


POLICIES: Dict[str, Policy] = {"bounded": bounded_policy, "naive": naive_policy}

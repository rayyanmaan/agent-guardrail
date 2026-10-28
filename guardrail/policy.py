"""Dependency kinds and the policies that turn them into approval conditions.

Tool-agnostic. A tool's dependency set (e.g. tools/crm_dependencies.py) lists
Equality and Bound entries; a policy combines that set with the agent's
decision snapshot to produce the Conditions the action was approved under.

Three policies, so the benchmark can measure what each refinement buys:
  - naive:     every dependency field must not change (strict equality everywhere).
  - bounded:   equality fields must not change, bound fields must stay in bound.
               Strict: values are compared exactly as stored.
  - canonical: bounded, plus two narrow exceptions on equality fields:
               a field's own canonicalizer (provably same fact, different
               spelling), and convergence (a field the action writes already
               holds exactly the value it would write).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from guardrail.recheck import NO_TARGET, Condition, identity, take_snapshot


@dataclass(frozen=True)
class Equality:
    """Field must keep the value the agent decided on.

    `canonicalize` is used only by the canonical policy. Leave it as identity
    unless every pair of values it maps together is provably the same fact.
    """
    field: str
    canonicalize: Callable[[Any], Any] = identity


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
# (dependencies, action, decision snapshot, fields the action writes) -> conditions
Policy = Callable[..., List[Condition]]


def bounded_policy(deps: Sequence[Dependency], proposal: Dict[str, Any], decision: Dict[str, Any],
                   writes: Optional[Mapping[str, Any]] = None, canonical: bool = False) -> List[Condition]:
    snapshot = take_snapshot(decision, [d.field for d in deps])
    writes = writes or {}
    conditions = []
    for d in deps:
        if isinstance(d, Bound):
            conditions.append(Condition(d.field, d.op, d.limit(proposal)))
        elif canonical:
            conditions.append(Condition(d.field, "==", snapshot[d.field], d.canonicalize,
                                        writes.get(d.field, NO_TARGET)))
        else:
            conditions.append(Condition(d.field, "==", snapshot[d.field]))
    return conditions


def canonical_policy(deps: Sequence[Dependency], proposal: Dict[str, Any], decision: Dict[str, Any],
                     writes: Optional[Mapping[str, Any]] = None) -> List[Condition]:
    return bounded_policy(deps, proposal, decision, writes, canonical=True)


def naive_policy(deps: Sequence[Dependency], proposal: Dict[str, Any], decision: Dict[str, Any],
                 writes: Optional[Mapping[str, Any]] = None) -> List[Condition]:
    """Baseline: any change to any dependency field blocks. Always strict."""
    snapshot = take_snapshot(decision, [d.field for d in deps])
    return [Condition(f, "==", v) for f, v in snapshot.items()]


POLICIES: Dict[str, Policy] = {
    "naive": naive_policy, "bounded": bounded_policy, "canonical": canonical_policy,
}

"""Approval conditions, snapshot and comparison.

Tool-agnostic: callers supply the explicit dependency set (see
tools/crm_dependencies.py for the CRM one). Only fields named in a condition
are checked, so a change to an unrelated field never voids approval.

A Condition is one approved fact the action relies on:
  - ("status", "==", "active")  equality field: must not change at all
  - ("seats",  "<=", 20)        predicate field: may move, but must stay in bound

An equality condition can also carry two narrow, opt-in exceptions (only the
`canonical` policy sets them; see guardrail/policy.py):
  - canonicalize: compare canonical forms, for a field whose different spellings
    provably name the same fact (default: identity, i.e. exact comparison)
  - converges_to: the value the action itself writes to this field. If the
    field already holds exactly that value, the action is a no-op on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict

OPS = ("==", "<=", ">=")


def identity(value: Any) -> Any:
    return value


class _NoTarget:
    def __repr__(self) -> str:
        return "NO_TARGET"


NO_TARGET: Any = _NoTarget()   # converges_to for a field the action does not write


class Verdict(TypedDict, total=False):
    # True when any dependency moved outside its approved limit. For an
    # equality field that means any change at all.
    changed: bool
    changed_fields: List[str]
    # field -> {"approved": value or bound, "current": value at recheck}
    # plus "op" for bound conditions.
    diffs: Dict[str, Dict[str, Any]]
    # Present only when an equality field differs but was accepted:
    # field -> {"approved", "current", "as": "canonical" | "converged"}
    accepted: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: Any
    canonicalize: Callable[[Any], Any] = identity
    converges_to: Any = NO_TARGET

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown condition operator {self.op!r}")
        if self.op != "==" and (self.canonicalize is not identity or self.converges_to is not NO_TARGET):
            raise ValueError("canonicalize and converges_to apply to equality conditions only")

    def holds(self, current: Any) -> bool:
        return self.holds_as(current) is not None

    def holds_as(self, current: Any) -> Optional[str]:
        """How the condition holds ("equal", "canonical", "converged", "in_bound"), or None."""
        if current is _MISSING:
            return None
        if self.op != "==":
            # Bounds only apply to numbers; anything else fails closed.
            if not (_is_number(current) and _is_number(self.value)):
                return None
            ok = current <= self.value if self.op == "<=" else current >= self.value
            return "in_bound" if ok else None
        if _strictly_equal(self.value, current):
            return "equal"
        canon = self.canonicalize
        if canon is not identity and _strictly_equal(canon(self.value), canon(current)):
            return "canonical"
        # Convergence: the field moved, but to exactly what this action would
        # write. The state after the action is identical either way, so nothing
        # the agent decided on is lost. Policies only set converges_to on fields
        # the action writes, so this can never excuse a change to a fact the
        # action merely relies on (status, balance, opt-in...).
        if self.converges_to is not NO_TARGET and _strictly_equal(canon(self.converges_to), canon(current)):
            return "converged"
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = {"field": self.field, "op": self.op, "value": self.value}
        if self.canonicalize is not identity:
            d["canonicalize"] = self.canonicalize.__name__
        if self.converges_to is not NO_TARGET:
            d["converges_to"] = self.converges_to
        return d


def take_snapshot(record: Dict[str, Any], dependency_fields: Iterable[str]) -> Dict[str, Any]:
    """Copy just the dependency fields out of a record."""
    fields = list(dependency_fields)
    if not fields:
        raise ValueError("dependency set must be explicit and non-empty")
    missing = [f for f in fields if f not in record]
    if missing:
        raise KeyError(f"record is missing dependency fields: {missing}")
    return {f: record[f] for f in fields}


def evaluate(conditions: Iterable[Condition], current_record: Dict[str, Any]) -> Verdict:
    """Check every approved condition against the current record.

    A dependency field that has disappeared from the current record counts as
    changed. Equality is strict on type: 5 and "5" are different values.
    """
    changed_fields: List[str] = []
    diffs: Dict[str, Dict[str, Any]] = {}
    accepted: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        current_value = current_record.get(cond.field, _MISSING)
        how = cond.holds_as(current_value)
        if how is None:
            changed_fields.append(cond.field)
            diff = {
                "approved": cond.value,
                "current": None if current_value is _MISSING else current_value,
            }
            if cond.op != "==":
                diff["op"] = cond.op
            diffs[cond.field] = diff
        elif how in ("canonical", "converged"):
            accepted[cond.field] = {"approved": cond.value, "current": current_value, "as": how}
    verdict = Verdict(changed=bool(changed_fields), changed_fields=changed_fields, diffs=diffs)
    if accepted:
        verdict["accepted"] = accepted
    return verdict


def compare(approved: Dict[str, Any], current_record: Dict[str, Any]) -> Verdict:
    """Strict equality over exactly the approved snapshot's fields."""
    return evaluate([Condition(f, "==", v) for f, v in approved.items()], current_record)


_MISSING = object()


def _strictly_equal(a: Any, b: Any) -> bool:
    # `==` alone would treat 1 == 1.0 == True as equal.
    return type(a) is type(b) and a == b


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

"""Approval conditions, snapshot and comparison.

Tool-agnostic: callers supply the explicit dependency set (see
tools/crm_dependencies.py for the CRM one). Only fields named in a condition
are checked, so a change to an unrelated field never voids approval.

A Condition is one approved fact the action relies on:
  - ("status", "==", "active")  equality field: must not change at all
  - ("seats",  "<=", 20)        predicate field: may move, but must stay in bound
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, TypedDict

OPS = ("==", "<=", ">=")


class Verdict(TypedDict):
    # True when any dependency moved outside its approved limit. For an
    # equality field that means any change at all.
    changed: bool
    changed_fields: List[str]
    # field -> {"approved": value or bound, "current": value at recheck}
    # plus "op" for bound conditions.
    diffs: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: Any

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown condition operator {self.op!r}")

    def holds(self, current: Any) -> bool:
        if current is _MISSING:
            return False
        if self.op == "==":
            return _strictly_equal(self.value, current)
        # Bounds only apply to numbers; anything else fails closed.
        if not (_is_number(current) and _is_number(self.value)):
            return False
        return current <= self.value if self.op == "<=" else current >= self.value

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Condition":
        return cls(d["field"], d["op"], d["value"])


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
    for cond in conditions:
        current_value = current_record.get(cond.field, _MISSING)
        if not cond.holds(current_value):
            changed_fields.append(cond.field)
            diff = {
                "approved": cond.value,
                "current": None if current_value is _MISSING else current_value,
            }
            if cond.op != "==":
                diff["op"] = cond.op
            diffs[cond.field] = diff
    return Verdict(changed=bool(changed_fields), changed_fields=changed_fields, diffs=diffs)


def compare(approved: Dict[str, Any], current_record: Dict[str, Any]) -> Verdict:
    """Strict equality over exactly the approved snapshot's fields."""
    return evaluate([Condition(f, "==", v) for f, v in approved.items()], current_record)


_MISSING = object()


def _strictly_equal(a: Any, b: Any) -> bool:
    # `==` alone would treat 1 == 1.0 == True as equal.
    return type(a) is type(b) and a == b


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

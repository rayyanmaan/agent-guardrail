"""Dependency-set snapshot and strict-equality comparison.

Tool-agnostic: callers supply the explicit dependency set (see
tools/crm_dependencies.py for the CRM one). Only fields in that set are
compared, so a change to an unrelated field never voids approval.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, TypedDict


class Verdict(TypedDict):
    changed: bool
    changed_fields: List[str]
    # field -> {"approved": value at approval, "current": value at recheck}
    diffs: Dict[str, Dict[str, Any]]


def take_snapshot(record: Dict[str, Any], dependency_fields: Iterable[str]) -> Dict[str, Any]:
    """Copy just the dependency fields out of a record."""
    fields = list(dependency_fields)
    if not fields:
        raise ValueError("dependency set must be explicit and non-empty")
    missing = [f for f in fields if f not in record]
    if missing:
        raise KeyError(f"record is missing dependency fields: {missing}")
    return {f: record[f] for f in fields}


def compare(approved: Dict[str, Any], current_record: Dict[str, Any]) -> Verdict:
    """Strict equality over exactly the approved snapshot's fields.

    A dependency field that has disappeared from the current record counts as
    changed. Types matter: 5 and "5" are different values.
    """
    changed_fields: List[str] = []
    diffs: Dict[str, Dict[str, Any]] = {}
    for field, approved_value in approved.items():
        current_value = current_record.get(field, _MISSING)
        if not _strictly_equal(approved_value, current_value):
            changed_fields.append(field)
            diffs[field] = {
                "approved": approved_value,
                "current": None if current_value is _MISSING else current_value,
            }
    return Verdict(changed=bool(changed_fields), changed_fields=changed_fields, diffs=diffs)


_MISSING = object()


def _strictly_equal(a: Any, b: Any) -> bool:
    # `==` alone would treat 1 == 1.0 == True as equal.
    return type(a) is type(b) and a == b

"""Adapter that runs the mock CRM behind the guardrail (see guardrail/tool.py).

The first half is what the guardrail needs. The second half is harness support
for scenarios: seeding, out-of-band changes, a full dump of state, and a plain
dict model of the effect, used as an independent oracle for the final state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.tools import tool

from tools.crm_dependencies import dependency_specs
from tools.mock_crm import EDITABLE_FIELDS, MockCRM, UpdateResult


@tool
def read_record(record_id: str) -> str:
    """Read a customer record from the CRM by id (e.g. 'C-1001')."""
    raise NotImplementedError("executed by the guardrail graph")


@tool
def propose_update(record_id: str, field: str, new_value: str) -> str:
    """Propose changing one field of a customer record. Does not write; the change
    is approved and executed by a separate system."""
    raise NotImplementedError("executed by the guardrail graph")


def _kind(value: Any) -> str:
    return "whole number" if isinstance(value, int) else "text"


class CRMTool:
    name = "crm"
    target_arg = "record_id"
    read_tool = read_record
    propose_tool = propose_update

    def __init__(self, crm: MockCRM):
        self.crm = crm

    @classmethod
    def open(cls, path: str) -> "CRMTool":
        return cls(MockCRM(path))

    def close(self) -> None:
        self.crm.close()

    # ---- guardrail interface ------------------------------------------------

    def read(self, target: str) -> Optional[Dict[str, Any]]:
        return self.crm.read_record(target)

    def validate(self, args: Dict[str, Any], decision: Dict[str, Any]):
        field = str(args.get("field", "")).strip()
        new_value = args.get("new_value")
        if field not in EDITABLE_FIELDS:
            kinds = ", ".join(f"{f} ({_kind(decision.get(f))})" for f in sorted(EDITABLE_FIELDS))
            return None, f"error: field must be one of: {kinds}"
        if new_value is None:
            return None, "error: new_value is required"
        # Keep the proposed value's type consistent with the column, so strict
        # equality and the compare-and-set behave predictably.
        if isinstance(decision[field], int):
            try:
                new_value = int(new_value)
            except (TypeError, ValueError):
                return None, f"error: {field} must be a whole number such as 15, with no % sign or units"
        else:
            new_value = str(new_value).strip()
        return {"record_id": args["record_id"], "field": field, "new_value": new_value}, None

    def dependencies(self, action: Dict[str, Any]):
        return dependency_specs(action["field"])

    def writes(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return {action["field"]: action["new_value"]}

    def apply(self, action: Dict[str, Any], expected: Dict[str, Any],
              bounds: Sequence[Tuple[str, str, Any]]) -> UpdateResult:
        return self.crm.conditional_update(
            action["record_id"], expected, {action["field"]: action["new_value"]}, bounds)

    # ---- scenario harness ---------------------------------------------------

    def seed(self, rows: Sequence[Dict[str, Any]]) -> None:
        for row in rows:
            self.crm.seed(row)

    def mutate_out_of_band(self, target: str, changes: Dict[str, Any]) -> None:
        self.crm.mutate_out_of_band(target, **changes)

    def dump(self) -> Dict[str, Any]:
        return {"records": {r["id"]: r for r in self.crm.read_all()}, "effects": []}

    @staticmethod
    def simulate_effect(action: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """What the effect should do, as plain dict operations. Edits `records` in place."""
        records[action["record_id"]][action["field"]] = action["new_value"]
        return []

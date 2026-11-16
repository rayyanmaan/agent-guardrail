"""The approval ledger entry records the declared dependency set, not only the compiled conditions."""

import pytest

from agent.graph import ScriptedPlanner
from guardrail.ledger import Ledger
from scenarios.benchmark import BY_KEY
from scenarios.runner import run_scenario
from tools.registry import TOOL_TYPES


def approval_of(key, policy, tmp_path):
    run_scenario(BY_KEY[key], lambda s: ScriptedPlanner(s.script), str(tmp_path), TOOL_TYPES, policy)
    ledger = Ledger(str(tmp_path / f"{key}_{policy}_ledger.sqlite"))
    try:
        assert ledger.verify_chain()
        return [e for e in ledger.entries() if e["event"] == "approval"][0]["payload"]
    finally:
        ledger.close()


@pytest.mark.parametrize("policy", ("naive", "bounded", "canonical"))
def test_plan_upgrade_records_its_declared_dependencies(policy, tmp_path):
    p = approval_of("status_cancelled", policy, tmp_path)
    assert p["dependency_set"] == [{"field": "plan", "kind": "Equality"},
                                   {"field": "status", "kind": "Equality"},
                                   {"field": "seats", "kind": "Bound", "op": "<="}]
    # Recorded alongside, not instead of, the compiled conditions.
    assert [c["field"] for c in p["conditions"]] == ["plan", "status", "seats"]


def test_discount_records_both_bounds(tmp_path):
    p = approval_of("discount_balance_over", "canonical", tmp_path)
    assert p["dependency_set"] == [{"field": "discount_pct", "kind": "Equality"},
                                   {"field": "status", "kind": "Equality"},
                                   {"field": "seats", "kind": "Bound", "op": ">="},
                                   {"field": "balance_due", "kind": "Bound", "op": "<="}]


def test_mailer_send_records_its_declared_dependencies(tmp_path):
    p = approval_of("mailer_email_case", "canonical", tmp_path)
    assert p["dependency_set"] == [{"field": "recipient_email", "kind": "Equality"},
                                   {"field": "opt_in_status", "kind": "Equality"}]

"""A declared dependency field that is missing fails closed: it never passes quietly.

Schema drift (a field renamed or dropped) must not turn a dependency into a
check that finds nothing and passes. A missing field is refused at approval,
and counts as changed at recheck, so the action blocks.
"""

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import ScriptedPlanner, build_graph
from guardrail.ledger import Ledger
from guardrail.recheck import Condition, evaluate, take_snapshot
from scenarios.crm_benchmark import SEED, UPGRADE, edit_script
from tools.crm_tool import CRMTool
from tools.mock_crm import MockCRM

POLICIES = ("naive", "bounded", "canonical")


def without(record, field):
    return {k: v for k, v in record.items() if k != field}


# ---- approval: snapshot of a missing field ------------------------------------------

def test_snapshot_refuses_a_missing_dependency_field_and_names_it():
    with pytest.raises(KeyError) as err:
        take_snapshot(without(SEED, "status"), ("plan", "status", "seats"))
    assert "status" in str(err.value)


# ---- recheck: a field missing from the current record counts as changed --------------

def test_missing_equality_field_counts_as_changed():
    verdict = evaluate([Condition("status", "==", "active")], without(SEED, "status"))
    assert verdict["changed"] is True
    assert verdict["changed_fields"] == ["status"]
    assert verdict["diffs"]["status"] == {"approved": "active", "current": None}


def test_missing_bound_field_counts_as_changed():
    verdict = evaluate([Condition("seats", "<=", 20)], without(SEED, "seats"))
    assert verdict["changed"] is True and verdict["changed_fields"] == ["seats"]


def test_missing_field_is_not_accepted_by_canonicalization_or_convergence():
    conditions = [Condition("recipient_email", "==", "dana@acme.example", canonicalize=str.lower),
                  Condition("plan", "==", "starter", converges_to="growth")]
    verdict = evaluate(conditions, {})
    assert verdict["changed"] is True
    assert verdict["changed_fields"] == ["recipient_email", "plan"]
    assert "accepted" not in verdict


# ---- whole path: present at approval, renamed by recheck ---------------------------

class DriftingCRMTool(CRMTool):
    """The agent's read sees `status`; every later read sees it renamed (schema drift)."""

    def __init__(self, crm):
        super().__init__(crm)
        self.reads = 0

    def read(self, target):
        record = super().read(target)
        self.reads += 1
        if record is not None and self.reads > 1:
            record["account_status"] = record.pop("status")
        return record


@pytest.mark.parametrize("policy", POLICIES)
def test_field_renamed_after_approval_blocks_and_names_the_field(policy, tmp_path):
    crm = MockCRM(str(tmp_path / "crm.sqlite"))
    crm.seed(dict(SEED))
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    graph = build_graph(ScriptedPlanner(edit_script(**UPGRADE)), [DriftingCRMTool(crm)], ledger, policy=policy)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("upgrade")], "run_id": "t", "plan_id": "t"}, config)
    state = graph.invoke(None, config)

    assert state["outcome"] == "blocked"
    assert "status" in state["block_reason"]
    block = [e for e in ledger.entries() if e["event"] == "block"][0]["payload"]
    assert block["stage"] == "recheck" and block["changed_fields"] == ["status"]
    assert crm.read_record("C-1001")["plan"] == "starter"   # the write never happened
    assert ledger.verify_chain()
    crm.close()
    ledger.close()

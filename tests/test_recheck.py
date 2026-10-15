import pytest
from langchain_core.messages import HumanMessage

from agent.graph import ScriptedPlanner, build_graph
from guardrail.ledger import Ledger
from guardrail.recheck import compare, take_snapshot
from scenarios.runner import run_scenario
from scenarios.scenario_a_stale import SCENARIO as SCENARIO_A
from scenarios.scenario_b_legitimate import SCENARIO as SCENARIO_B
from tools.crm_dependencies import dependencies_for
from tools.mock_crm import MockCRM

RECORD = dict(SCENARIO_A.seed)


# ---- dependency-set comparison ------------------------------------------

def test_plan_edit_depends_on_plan_and_status():
    assert dependencies_for("plan") == ("plan", "status")


def test_snapshot_holds_only_dependency_fields():
    assert take_snapshot(RECORD, ("plan", "status")) == {"plan": "starter", "status": "active"}


def test_snapshot_requires_explicit_set():
    with pytest.raises(ValueError):
        take_snapshot(RECORD, ())


def test_unrelated_change_is_not_a_change():
    snap = take_snapshot(RECORD, dependencies_for("plan"))
    verdict = compare(snap, {**RECORD, "notes": "hi", "last_contacted": "2026-10-27"})
    assert verdict == {"changed": False, "changed_fields": [], "diffs": {}}


def test_dependency_change_names_the_field():
    snap = take_snapshot(RECORD, dependencies_for("plan"))
    verdict = compare(snap, {**RECORD, "status": "cancelled"})
    assert verdict["changed"] and verdict["changed_fields"] == ["status"]
    assert verdict["diffs"]["status"] == {"approved": "active", "current": "cancelled"}


def test_equality_is_strict_on_type():
    assert compare({"seats": 5}, {"seats": "5"})["changed"]
    assert compare({"seats": 1}, {"seats": True})["changed"]


def test_missing_dependency_field_counts_as_changed():
    assert compare({"status": "active"}, {})["changed_fields"] == ["status"]


# ---- conditional update (compare-and-set) -------------------------------

@pytest.fixture
def crm(tmp_path):
    crm = MockCRM(str(tmp_path / "crm.sqlite"))
    crm.seed(RECORD)
    yield crm
    crm.close()


def test_conditional_update_refuses_stale_row(crm):
    snap = {"plan": "starter", "status": "active"}
    crm.mutate_out_of_band("C-1001", status="cancelled")
    result = crm.conditional_update("C-1001", snap, {"plan": "growth"})
    assert not result.written
    assert crm.read_record("C-1001")["plan"] == "starter"


def test_conditional_update_ignores_unrelated_drift(crm):
    crm.mutate_out_of_band("C-1001", notes="changed")
    result = crm.conditional_update("C-1001", {"plan": "starter", "status": "active"}, {"plan": "growth"})
    assert result.written
    assert crm.read_record("C-1001") == {**RECORD, "notes": "changed", "plan": "growth"}


def test_conditional_update_requires_snapshot(crm):
    with pytest.raises(ValueError):
        crm.conditional_update("C-1001", {}, {"plan": "growth"})


def test_toctou_change_between_recheck_and_write_is_refused(crm, tmp_path):
    """Recheck passes, then a dependency changes before the write lands."""
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    graph = build_graph(
        ScriptedPlanner("C-1001", "plan", "growth"), crm, ledger,
        before_write=lambda: crm.mutate_out_of_band("C-1001", status="suspended"),
    )
    config = {"configurable": {"thread_id": "toctou"}}
    graph.invoke({"messages": [HumanMessage("upgrade")], "run_id": "toctou"}, config)
    state = graph.invoke(None, config)

    assert state["verdict"]["changed"] is False          # recheck saw clean data
    assert state["outcome"] == "blocked"                  # but the CAS refused
    assert crm.read_record("C-1001")["plan"] == "starter"
    entries = ledger.entries("toctou")
    assert [e["event"] for e in entries] == ["approval", "recheck", "block"]
    assert entries[-1]["payload"]["stage"] == "conditional_update"
    assert ledger.verify_chain()


# ---- the two acceptance scenarios, without Ollama -----------------------

@pytest.mark.parametrize("scenario", [SCENARIO_A, SCENARIO_B], ids=["A_stale", "B_unrelated"])
def test_scenarios_pass_with_scripted_planner(scenario, tmp_path):
    result = run_scenario(scenario, lambda s: ScriptedPlanner(**s.proposal), str(tmp_path))
    assert result.passed, result.failures
    assert result.outcome == scenario.expected_outcome

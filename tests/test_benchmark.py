import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import ScriptedPlanner, build_graph
from guardrail.ledger import Ledger
from guardrail.policy import bounded_policy, naive_policy
from guardrail.recheck import Condition, evaluate
from scenarios.crm_benchmark import BY_KEY, DECISION_TIME_DRIFT, SCENARIOS, SEED, edit_script
from scenarios.metrics import Confusion, confusion
from scenarios.runner import confusion_of, run_benchmark, run_scenario
from tools.crm_dependencies import dependency_specs
from tools.crm_tool import CRMTool
from tools.mock_crm import MockCRM
from tools.registry import TOOL_TYPES

PLAN_TO_GROWTH = {"record_id": "C-1001", "field": "plan", "new_value": "growth"}


def scripted(scenario):
    return ScriptedPlanner(scenario.script)


@pytest.fixture
def crm(tmp_path):
    crm = MockCRM(str(tmp_path / "crm.sqlite"))
    crm.seed(SEED)
    yield crm
    crm.close()


# ---- Job 1: decision-time snapshot ---------------------------------------

def test_change_between_read_and_proposal_blocks(tmp_path):
    # Regression. The account is suspended after the agent's read, before the
    # proposal reaches the guardrail. Approval is based on the snapshot carried
    # from the agent's read, so the recheck sees status move and blocks. In
    # Sprint 1, approval took a fresh read at this point, so it approved against
    # the suspended row and the write was wrongly allowed.
    result = run_scenario(DECISION_TIME_DRIFT, scripted, str(tmp_path), TOOL_TYPES)
    assert result.passed, result.failures
    assert result.outcome == "blocked"


def test_fresh_read_baseline_would_have_missed_it(crm):
    # The counterfactual the regression guards against, made explicit.
    crm.mutate_out_of_band("C-1001", status="suspended")
    deps = dependency_specs("plan")
    fresh = bounded_policy(deps, PLAN_TO_GROWTH, crm.read_record("C-1001"))
    decided = bounded_policy(deps, PLAN_TO_GROWTH, SEED)
    now = crm.read_record("C-1001")
    assert not evaluate(fresh, now)["changed"]
    assert evaluate(decided, now)["changed_fields"] == ["status"]


def test_approval_ledger_records_decision_snapshot(tmp_path, crm):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    graph = build_graph(ScriptedPlanner(edit_script(**PLAN_TO_GROWTH)), [CRMTool(crm)], ledger,
                        after_read=lambda: crm.mutate_out_of_band("C-1001", status="suspended"))
    graph.invoke({"messages": [HumanMessage("upgrade")], "run_id": "r"},
                 {"configurable": {"thread_id": "r"}})
    approval = ledger.entries("r")[0]["payload"]
    assert approval["decision_snapshot"] == {"plan": "starter", "status": "active", "seats": 5}
    assert approval["decision_read_at"]


class _ProposeWithoutReading:
    def invoke(self, messages):
        return AIMessage("", tool_calls=[{"name": "propose_update", "args": PLAN_TO_GROWTH,
                                          "id": f"c{len(messages)}", "type": "tool_call"}])


def test_proposal_without_a_read_is_refused(tmp_path, crm):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    graph = build_graph(_ProposeWithoutReading(), [CRMTool(crm)], ledger)
    state = graph.invoke({"messages": [HumanMessage("upgrade")], "run_id": "r"},
                         {"configurable": {"thread_id": "r"}})
    assert state["outcome"] == "no_proposal"
    assert "read_record" in state["messages"][-1].content
    assert ledger.entries("r") == []
    assert crm.read_record("C-1001") == SEED


# ---- predicate (bound) fields ---------------------------------------------

@pytest.mark.parametrize("seats, holds", [
    (5, True), (20, True),        # within bound, including the boundary
    (21, False),                  # just outside
    ("20", False), (None, False), (True, False),  # non-numbers fail closed
])
def test_bound_condition(seats, holds):
    assert Condition("seats", "<=", 20).holds(seats) is holds


def test_bounded_policy_turns_seats_into_a_bound():
    conds = bounded_policy(dependency_specs("plan"), PLAN_TO_GROWTH, SEED)
    assert conds == [Condition("plan", "==", "starter"), Condition("status", "==", "active"),
                     Condition("seats", "<=", 20)]


def test_naive_policy_uses_equality_everywhere():
    conds = naive_policy(dependency_specs("plan"), PLAN_TO_GROWTH, SEED)
    assert conds[-1] == Condition("seats", "==", 5)


def test_unknown_target_plan_fails_closed():
    conds = bounded_policy(dependency_specs("plan"), {**PLAN_TO_GROWTH, "new_value": "platinum"}, SEED)
    assert evaluate(conds, SEED)["changed_fields"] == ["seats"]


def test_conditional_update_enforces_bounds(crm):
    expected = {"plan": "starter", "status": "active"}
    crm.mutate_out_of_band("C-1001", seats=21)
    assert not crm.conditional_update("C-1001", expected, {"plan": "growth"}, [("seats", "<=", 20)]).written
    crm.mutate_out_of_band("C-1001", seats=20)
    assert crm.conditional_update("C-1001", expected, {"plan": "growth"}, [("seats", "<=", 20)]).written


# ---- metrics ----------------------------------------------------------------

def test_confusion_and_rates():
    c = confusion([("BLOCK", "BLOCK"), ("BLOCK", "BLOCK"), ("BLOCK", "ALLOW"),
                   ("ALLOW", "ALLOW"), ("ALLOW", "BLOCK"), ("ALLOW", "ALLOW"), ("ALLOW", "ALLOW"),
                   ("BLOCK", None)])
    assert c == Confusion(blocked_should_block=2, allowed_should_block=1,
                          blocked_should_allow=1, allowed_should_allow=3, no_outcome=1)
    assert c.catch_rate == pytest.approx(2 / 3)       # the no-outcome run is excluded
    assert c.false_alarm_rate == pytest.approx(1 / 4)


def test_rates_undefined_without_scenarios():
    assert Confusion().catch_rate is None and Confusion().false_alarm_rate is None


def test_confusion_rejects_bad_labels():
    with pytest.raises(ValueError):
        confusion([("block", "BLOCK")])


# ---- the benchmark itself -----------------------------------------------------

def test_benchmark_is_balanced_and_covers_every_label():
    expected = [s.expected for s in SCENARIOS]
    assert len(SCENARIOS) >= 8 and expected.count("BLOCK") == expected.count("ALLOW")
    assert {s.label for s in SCENARIOS} == {"irrelevant", "within_limits", "outside_limits"}


def test_bounded_policy_passes_every_scenario(tmp_path):
    results = run_benchmark(SCENARIOS, scripted, str(tmp_path), TOOL_TYPES, "bounded")
    assert [r.scenario.key for r in results if not r.passed] == []
    c = confusion_of(results)
    assert (c.catch_rate, c.false_alarm_rate) == (1.0, 0.0)


def test_naive_baseline_raises_false_alarms_only(tmp_path):
    bounded = confusion_of(run_benchmark(SCENARIOS, scripted, str(tmp_path), TOOL_TYPES, "bounded"))
    naive_results = run_benchmark(SCENARIOS, scripted, str(tmp_path), TOOL_TYPES, "naive")
    naive = confusion_of(naive_results)
    assert naive.catch_rate == bounded.catch_rate
    assert naive.false_alarm_rate > bounded.false_alarm_rate
    # Its false alarms are exactly the within-limits scenarios.
    wrongly_blocked = {r.scenario.key for r in naive_results if r.scenario.expected == "ALLOW" and r.actual == "BLOCK"}
    assert wrongly_blocked == {k for k, s in BY_KEY.items() if s.label == "within_limits"}

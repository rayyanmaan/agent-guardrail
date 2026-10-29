"""Sprint 3.5: canonicalization and target convergence, and that they stay narrow.

The pinned before/after numbers were predicted before the canonical policy was
first run. If a change moves them, report it; do not absorb it here.
"""

import dataclasses
import os

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import ScriptedPlanner, build_graph
from guardrail.ledger import Ledger
from guardrail.policy import bounded_policy, canonical_policy
from guardrail.recheck import NO_TARGET, Condition, evaluate
from scenarios.benchmark import BY_KEY, SCENARIOS
from scenarios.crm_benchmark import SEED, UPGRADE
from scenarios.mailer_benchmark import CONTACT, SEND, send_script
from scenarios.metrics import by_tool, summarize
from scenarios.runner import run_benchmark, run_scenario
from tools.crm_dependencies import dependency_specs as crm_deps
from tools.mailer_dependencies import canonical_email
from tools.mailer_dependencies import dependency_specs as mailer_deps
from tools.mailer_tool import MailerTool
from tools.mock_mailer import MockMailer
from tools.registry import TOOL_TYPES


def scripted(scenario):
    return ScriptedPlanner(scenario.script)


def run(scenario, tmp_path, policy="canonical"):
    return run_scenario(scenario, scripted, str(tmp_path), TOOL_TYPES, policy)


# ---- the email canonicalizer only folds case and surrounding whitespace -----------

def test_email_canonicalizer_folds_case_and_whitespace():
    assert canonical_email("  Dana@Acme.EXAMPLE \n") == "dana@acme.example"


@pytest.mark.parametrize("other", [
    "dana+billing@acme.example",   # plus tag: may be a different mailbox
    "d.ana@acme.example",          # dots: significant at most providers
    "dana@acme-corp.example",      # different domain
    "dana@acme.example.",          # trailing dot: not folded
])
def test_email_canonicalizer_keeps_distinct_addresses_distinct(other):
    assert canonical_email(other) != canonical_email("Dana@Acme.example")


def test_email_canonicalizer_passes_non_strings_through():
    assert canonical_email(None) is None


# ---- conditions -------------------------------------------------------------------

def test_canonical_condition_accepts_equivalent_spelling_only():
    c = Condition("recipient_email", "==", "Dana@Acme.example", canonical_email)
    assert c.holds_as("dana@acme.example") == "canonical"
    assert c.holds_as("dana@acme-corp.example") is None


def test_convergence_only_to_the_exact_target():
    c = Condition("plan", "==", "starter", converges_to="growth")
    assert c.holds_as("growth") == "converged"
    assert c.holds_as("enterprise") is None
    assert c.holds_as("Growth") is None   # plan has no canonicalizer: exact only


def test_bounds_cannot_canonicalize_or_converge():
    with pytest.raises(ValueError):
        Condition("seats", "<=", 20, converges_to=20)


# ---- policies: convergence is limited to fields the action writes -------------------

def test_canonical_policy_converges_only_on_written_fields():
    conds = {c.field: c for c in canonical_policy(crm_deps("plan"), UPGRADE, SEED, {"plan": "growth"})}
    assert conds["plan"].converges_to == "growth"
    assert conds["status"].converges_to is NO_TARGET
    assert conds["seats"].op == "<="   # bound untouched


def test_mailer_send_converges_on_nothing():
    conds = canonical_policy(mailer_deps(SEND), SEND, CONTACT, MailerTool.writes(None, SEND))
    assert all(c.converges_to is NO_TARGET for c in conds)


def test_strict_policy_is_unchanged():
    conds = bounded_policy(mailer_deps(SEND), SEND, CONTACT, {"recipient_email": "x"})
    assert conds == [Condition("recipient_email", "==", CONTACT["recipient_email"]),
                     Condition("opt_in_status", "==", "opted_in")]


def test_non_written_field_cannot_converge_even_to_the_written_value():
    # status changes to "growth", the value the action writes to plan. It must
    # still count as a change: convergence must not leak across fields.
    conds = canonical_policy(crm_deps("plan"), UPGRADE, SEED, {"plan": "growth"})
    assert evaluate(conds, {**SEED, "status": "growth"})["changed_fields"] == ["status"]


# ---- conservatism, end to end under the canonical policy ----------------------------

@pytest.mark.parametrize("key", [
    "mailer_recipient_changed",   # genuinely different address
    "plan_changed_elsewhere",     # plan moved, but not to the upgrade's target
    "status_cancelled",           # non-written dependency field
    "decision_time_drift",        # non-written field, change before the proposal
    "mailer_opted_out",           # non-written field on the mailer
    "discount_balance_over",      # non-written bound field
])
def test_dangerous_cases_still_block(key, tmp_path):
    r = run(BY_KEY[key], tmp_path)
    assert r.passed and r.actual == "BLOCK", r.failures


def test_case_variant_address_moved_to_a_different_mailbox_still_blocks(tmp_path):
    # Seeded with a mixed-case address; it changes to a lowercase address on a
    # different domain. Lowercasing must not hide the change of mailbox.
    scenario = dataclasses.replace(
        BY_KEY["mailer_email_case"], key="mailer_case_variant_new_mailbox",
        name="Mailer: mixed-case address replaced by a different mailbox", expected="BLOCK",
        shape="stale_fact", mutations=(("mailer", "M-2001", {"recipient_email": "dana@acme-corp.example"}),),
        expected_changed_fields=("recipient_email",))
    r = run(scenario, tmp_path)
    assert r.passed and r.actual == "BLOCK", r.failures


@pytest.mark.parametrize("status", ["cancelled", "growth", "Active", " active"])
def test_status_change_to_any_value_blocks(status, tmp_path):
    scenario = dataclasses.replace(
        BY_KEY["status_cancelled"], key=f"status_to_{status.strip().lower()}",
        mutations=(("crm", "C-1001", {"status": status}),))
    r = run(scenario, tmp_path)
    assert r.passed and r.actual == "BLOCK", r.failures


def test_write_stays_exact_after_recheck(tmp_path):
    # Recheck accepts the stored address; then a sync lowercases it before the
    # send. Equivalent or not, a change after the recheck is refused.
    mailer = MockMailer(str(tmp_path / "mailer.sqlite"))
    mailer.seed({**CONTACT, "recipient_email": "Dana@Acme.example"})
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    graph = build_graph(ScriptedPlanner(send_script(**SEND)), [MailerTool(mailer)], ledger,
                        policy="canonical",
                        before_write=lambda: mailer.mutate_out_of_band("M-2001", recipient_email="dana@acme.example"))
    config = {"configurable": {"thread_id": "w"}}
    graph.invoke({"messages": [HumanMessage("send")], "run_id": "w"}, config)
    assert graph.invoke(None, config)["outcome"] == "blocked"
    assert mailer.outbox() == []


# ---- the two cleared false alarms, and what the ledger shows -------------------------

@pytest.mark.parametrize("key, how", [("mailer_email_case", "canonical"), ("plan_already_target", "converged")])
def test_cleared_false_alarms_are_allowed_and_explained(key, how, tmp_path):
    r = run(BY_KEY[key], tmp_path)
    assert r.passed and r.actual == "ALLOW", r.failures
    ledger = Ledger(os.path.join(str(tmp_path), f"{key}_canonical_ledger.sqlite"))
    recheck = next(e for e in ledger.entries() if e["event"] == "recheck")
    assert [a["as"] for a in recheck["payload"]["verdict"]["accepted"].values()] == [how]


# ---- before / after, pinned -----------------------------------------------------------

@pytest.fixture(scope="module")
def summaries(tmp_path_factory):
    out = {}
    for policy in ("naive", "bounded", "canonical"):
        rs = run_benchmark(SCENARIOS, scripted, str(tmp_path_factory.mktemp(policy)), TOOL_TYPES, policy)
        assert [r.scenario.key for r in rs if r.integrity] == []
        rows = [r.scored() for r in rs]
        out[policy] = (summarize(rows), by_tool(rows))
    return out


def fa(summary):
    return (summary.overall.blocked_should_allow, summary.overall.legitimate)


def test_false_alarm_rate_before_and_after(summaries):
    (naive, _), (strict, strict_tools), (canon, canon_tools) = (
        summaries["naive"], summaries["bounded"], summaries["canonical"])
    assert (fa(naive), fa(strict), fa(canon)) == ((6, 12), (2, 12), (0, 12))
    assert (fa(strict_tools["crm"]), fa(strict_tools["mailer"])) == ((1, 9), (1, 3))
    assert (fa(canon_tools["crm"]), fa(canon_tools["mailer"])) == ((0, 9), (0, 3))


def test_catch_and_coverage_unchanged(summaries):
    strict, canon = summaries["bounded"][0], summaries["canonical"][0]
    for s in (strict, canon):
        assert (s.in_scope.blocked_should_block, s.in_scope.dangerous) == (9, 9)
        assert (s.overall.blocked_should_block, s.overall.dangerous) == (9, 11)
    # No dangerous case flips to ALLOW: the uncaught set is the same two shapes.
    assert {r.key for r in canon.uncaught} == {r.key for r in strict.uncaught} == {
        "mailer_lost_ack_retry", "mailer_disclosure_by_combination"}

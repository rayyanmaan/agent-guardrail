"""Idempotency closes the lost-ack-retry gap.

Covers: a retry is suppressed (both tools), a separately approved repeat is not,
the key is minted by the guardrail and never taken from the model, and a crash
between the effect and its acknowledgement is still recognised on retry. The
full 26-scenario benchmark is pinned at the end.
"""

import dataclasses
import os

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import ScriptedPlanner, build_graph
from guardrail.idempotency import intent_key
from guardrail.ledger import Ledger
from scenarios.benchmark import BY_KEY, SCENARIOS
from scenarios.mailer_benchmark import CONTACT, SEND
from scenarios.metrics import by_mechanism, summarize
from scenarios.runner import run_benchmark, run_scenario
from tools.mailer_tool import MailerTool
from tools.mock_crm import MockCRM
from tools.mock_mailer import MockMailer
from tools.registry import TOOL_TYPES

POLICIES = ("naive", "bounded", "canonical")
RECORD = dict(BY_KEY["crm_lost_ack_retry"].seed["crm"][0])
SNAP = {"plan": "starter", "status": "active"}
EXPECTED = {"recipient_email": "dana@acme.example", "opt_in_status": "opted_in"}


def scripted(scenario):
    return ScriptedPlanner(scenario.script)


def run(scenario, tmp_path, policy="canonical"):
    return run_scenario(scenario, scripted, str(tmp_path), TOOL_TYPES, policy)


def ledger_of(scenario, tmp_path, policy="canonical"):
    return Ledger(os.path.join(str(tmp_path), f"{scenario.key}_{policy}_ledger.sqlite"))


@pytest.fixture
def crm(tmp_path):
    db = MockCRM(str(tmp_path / "crm.sqlite"))
    db.seed(RECORD)
    yield db
    db.close()


@pytest.fixture
def mailer(tmp_path):
    db = MockMailer(str(tmp_path / "mailer.sqlite"))
    db.seed(CONTACT)
    yield db
    db.close()


# ---- duplicate suppressed ---------------------------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("key", ["crm_lost_ack_retry", "mailer_lost_ack_retry"])
def test_lost_ack_retry_is_suppressed_under_every_policy(key, policy, tmp_path):
    r = run(BY_KEY[key], tmp_path, policy)
    # Integrity passing means the store holds exactly one effect.
    assert (r.verdict, r.outcome) == ("PASS", "deduplicated"), r.failures


@pytest.mark.parametrize("key", ["crm_lost_ack_retry", "mailer_lost_ack_retry"])
def test_ledger_shows_key_minted_then_duplicate_suppressed(key, tmp_path):
    run(BY_KEY[key], tmp_path)
    entries = ledger_of(BY_KEY[key], tmp_path).entries()
    assert [e["event"] for e in entries] == ["approval", "recheck", "duplicate_suppressed"]
    minted = entries[0]["payload"]["idempotency_key"]
    assert len(minted) == 64 and entries[-1]["payload"]["idempotency_key"] == minted


def test_retry_returns_the_original_send(tmp_path):
    run(BY_KEY["mailer_lost_ack_retry"], tmp_path)
    suppressed = ledger_of(BY_KEY["mailer_lost_ack_retry"], tmp_path).entries()[-1]
    assert suppressed["payload"]["original"] == {
        "contact_id": "M-2001", "to_email": "dana@acme.example",
        "subject": SEND["subject"], "body": SEND["body"]}


def test_crm_retry_is_reported_as_already_applied_not_stale(tmp_path):
    run(BY_KEY["crm_lost_ack_retry"], tmp_path)
    events = [e["event"] for e in ledger_of(BY_KEY["crm_lost_ack_retry"], tmp_path).entries()]
    assert "block" not in events and events[-1] == "duplicate_suppressed"


# ---- a separately approved repeat is not suppressed ------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("key", ["crm_reupgrade_new_plan", "mailer_resend_new_plan"])
def test_separately_approved_repeat_goes_through(key, policy, tmp_path):
    r = run(BY_KEY[key], tmp_path, policy)
    # Integrity passing means the store holds both effects.
    assert (r.verdict, r.outcome) == ("PASS", "allowed"), r.failures


@pytest.mark.parametrize("key", ["crm_reupgrade_new_plan", "mailer_resend_new_plan"])
def test_repeat_differs_from_the_earlier_plan_only_in_its_key(key, tmp_path):
    run(BY_KEY[key], tmp_path)
    approvals = [e["payload"] for e in ledger_of(BY_KEY[key], tmp_path).entries()
                 if e["event"] == "approval"]
    first, second = approvals
    # Same action and same decision snapshot; only the plan, and so the key, differ.
    assert (first["proposal"], first["decision_snapshot"]) == (second["proposal"], second["decision_snapshot"])
    assert first["idempotency_key"] != second["idempotency_key"]


# ---- the key is minted by the guardrail, never taken from the model ------------------------

def test_key_is_deterministic_and_covers_every_part_of_the_intent():
    base = dict(plan_id="p1", tool="crm", target="C-1001",
                action={"record_id": "C-1001", "field": "plan", "new_value": "growth"}, snapshot=SNAP)
    assert intent_key(**base) == intent_key(**base)
    variants = [{"plan_id": "p2"}, {"tool": "mailer"}, {"target": "C-1002"},
                {"action": {**base["action"], "new_value": "enterprise"}},
                {"snapshot": {**SNAP, "status": "suspended"}}]
    keys = {intent_key(**{**base, **v}) for v in variants}
    assert intent_key(**base) not in keys and len(keys) == len(variants)


def _send_twice_in_one_plan(mailer, ledger, model_keys):
    """Run two approvals of the same plan, the model claiming a different key each time."""
    for n, model_key in enumerate(model_keys):
        script = [("read_contact", {"contact_id": "M-2001"}),
                  ("propose_send", {**SEND, "idempotency_key": model_key})]
        graph = build_graph(ScriptedPlanner(script), [MailerTool(mailer)], ledger)
        config = {"configurable": {"thread_id": f"t{n}"}}
        graph.invoke({"messages": [HumanMessage("send")], "run_id": f"t{n}", "plan_id": "plan-1"}, config)
        yield graph.invoke(None, config)


def test_model_side_key_changes_do_not_create_a_second_effect(mailer, tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    first, second = _send_twice_in_one_plan(mailer, ledger, ["model-key-1", "model-key-2"])
    assert (first["outcome"], second["outcome"]) == ("allowed", "deduplicated")
    assert len(mailer.outbox()) == 1
    keys = {e["payload"]["idempotency_key"] for e in ledger.entries() if e["event"] == "approval"}
    assert len(keys) == 1 and not keys & {"model-key-1", "model-key-2"}
    assert "idempotency_key" not in first["proposal"]["action"]   # dropped at validation


# ---- crash between effect and acknowledgement ---------------------------------------------

def test_crash_after_commit_is_recognised_on_retry(crm, mailer):
    # The effect committed; whatever was meant to happen next (the ack, the
    # ledger entry) did not. The key was written with the effect, so the retry
    # is recognised.
    assert crm.conditional_update("C-1001", SNAP, {"plan": "growth"}, idempotency_key="k").written
    retry = crm.conditional_update("C-1001", SNAP, {"plan": "growth"}, idempotency_key="k")
    assert (retry.written, retry.duplicate) == (False, True)
    assert retry.original == {"record_id": "C-1001", "changes": {"plan": "growth"}}
    assert len(crm.applied_changes()) == 1

    assert mailer.conditional_send("M-2001", EXPECTED, "s", "b", idempotency_key="k").written
    retry = mailer.conditional_send("M-2001", EXPECTED, "s", "b", idempotency_key="k")
    assert (retry.written, retry.duplicate) == (False, True) and len(mailer.outbox()) == 1


def test_crash_between_update_and_key_lands_neither(crm, monkeypatch):
    # The CRM writes the change and its key as two statements in one
    # transaction. Interrupt between them: the change must roll back too, so
    # there is never a change without its key.
    def crash(*args):
        raise RuntimeError("process died")
    monkeypatch.setattr(crm, "_record_applied", crash)
    with pytest.raises(RuntimeError):
        crm.conditional_update("C-1001", SNAP, {"plan": "growth"}, idempotency_key="k")
    assert crm.read_record("C-1001")["plan"] == "starter" and crm.applied_changes() == []
    monkeypatch.undo()
    assert crm.conditional_update("C-1001", SNAP, {"plan": "growth"}, idempotency_key="k").written
    assert len(crm.applied_changes()) == 1


def test_unique_constraint_is_the_final_gate(crm, mailer, monkeypatch):
    # Even if the key lookup misses (a race, or a bug), the UNIQUE constraint
    # refuses the second row and the store reports a duplicate.
    assert crm.conditional_update("C-1001", SNAP, {"plan": "growth"}, idempotency_key="k").written
    crm.mutate_out_of_band("C-1001", plan="starter")
    monkeypatch.setattr(crm, "_applied", lambda key: None)
    retry = crm.conditional_update("C-1001", SNAP, {"plan": "growth"}, idempotency_key="k")
    monkeypatch.undo()
    assert retry.duplicate and len(crm.applied_changes()) == 1
    assert crm.read_record("C-1001")["plan"] == "starter"   # the second update rolled back

    assert mailer.conditional_send("M-2001", EXPECTED, "s", "b", idempotency_key="k").written
    monkeypatch.setattr(mailer, "_sent", lambda key: None)
    retry = mailer.conditional_send("M-2001", EXPECTED, "s", "b", idempotency_key="k")
    monkeypatch.undo()
    assert retry.duplicate and len(mailer.outbox()) == 1


@pytest.mark.parametrize("call", [
    lambda crm, mailer: crm.conditional_update("C-1001", SNAP, {"plan": "growth"}, idempotency_key=""),
    lambda crm, mailer: mailer.conditional_send("M-2001", EXPECTED, "s", "b", idempotency_key=""),
])
def test_effects_require_a_key(call, crm, mailer):
    with pytest.raises(ValueError):
        call(crm, mailer)


def test_between_plans_mutations_need_a_repeat_plan():
    with pytest.raises(ValueError):
        dataclasses.replace(BY_KEY["crm_reupgrade_new_plan"], fault=None)


# ---- the full benchmark, pinned --------------------------------------------------------

@pytest.fixture(scope="module")
def summaries(tmp_path_factory):
    out = {}
    for policy in POLICIES:
        rs = run_benchmark(SCENARIOS, scripted, str(tmp_path_factory.mktemp(policy)), TOOL_TYPES, policy)
        assert [r.scenario.key for r in rs if r.integrity] == []
        rows = [r.scored() for r in rs]
        out[policy] = (summarize(rows), by_mechanism(rows))
    return out


def counts(c):
    return (c.blocked_should_block, c.dangerous), (c.blocked_should_allow, c.legitimate)


def test_benchmark_shape():
    assert len(SCENARIOS) == 26
    by = lambda m: sorted(s.expected for s in SCENARIOS if s.mechanism == m)
    assert by("recheck").count("BLOCK") == 9 and by("recheck").count("ALLOW") == 12
    assert by("idempotency") == ["ALLOW", "ALLOW", "BLOCK", "BLOCK"]
    assert [s.key for s in SCENARIOS if s.mechanism == "none"] == ["mailer_disclosure_by_combination"]


def test_full_benchmark_numbers(summaries):
    false_alarms = {"naive": (6, 14), "bounded": (2, 14), "canonical": (0, 14)}
    for policy, (s, mechanisms) in summaries.items():
        assert (s.in_scope.blocked_should_block, s.in_scope.dangerous) == (11, 11), policy
        assert counts(s.overall) == ((11, 12), false_alarms[policy]), policy
        assert [r.key for r in s.uncaught] == ["mailer_disclosure_by_combination"], policy
        # Idempotency is an effect-level check: the same under every policy.
        assert counts(mechanisms["idempotency"].overall) == ((2, 2), (0, 2)), policy
        assert counts(mechanisms["none"].overall)[0] == (0, 1), policy

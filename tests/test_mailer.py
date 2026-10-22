import ast
import pathlib

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import ScriptedPlanner, build_graph
from guardrail.ledger import Ledger
from guardrail.policy import Equality, bounded_policy
from guardrail.recheck import evaluate
from scenarios.mailer_benchmark import CONTACT, SEND, send_script
from tools.mailer_dependencies import dependency_specs
from tools.mailer_tool import MailerTool
from tools.mock_mailer import MockMailer

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def mailer(tmp_path):
    m = MockMailer(str(tmp_path / "mailer.sqlite"))
    m.seed(CONTACT)
    yield m
    m.close()


# ---- dependency set -----------------------------------------------------------

def test_send_depends_on_recipient_and_opt_in_by_equality():
    assert dependency_specs(SEND) == (Equality("recipient_email"), Equality("opt_in_status"))


@pytest.mark.parametrize("change, changed", [
    ({"recipient_email": "ops@acme.example"}, ["recipient_email"]),
    ({"opt_in_status": "opted_out"}, ["opt_in_status"]),
    ({"name": "Dana Ortiz-Kim", "company": "Acme Corp", "notes": "x"}, []),
])
def test_only_dependency_fields_void_approval(change, changed):
    conds = bounded_policy(dependency_specs(SEND), SEND, CONTACT)
    assert evaluate(conds, {**CONTACT, **change})["changed_fields"] == changed


# ---- conditional send ---------------------------------------------------------

EXPECTED = {"recipient_email": "dana@acme.example", "opt_in_status": "opted_in"}


def test_send_goes_to_the_checked_address(mailer):
    assert mailer.conditional_send("M-2001", EXPECTED, "s", "b").written
    assert mailer.outbox() == [{"contact_id": "M-2001", "to_email": "dana@acme.example",
                                "subject": "s", "body": "b"}]


@pytest.mark.parametrize("change", [{"recipient_email": "ops@acme.example"}, {"opt_in_status": "opted_out"}])
def test_send_refused_when_a_dependency_moved(mailer, change):
    mailer.mutate_out_of_band("M-2001", **change)
    result = mailer.conditional_send("M-2001", EXPECTED, "s", "b")
    assert not result.written and result.reason.startswith("stale")
    assert mailer.outbox() == []


def test_send_requires_conditions(mailer):
    with pytest.raises(ValueError):
        mailer.conditional_send("M-2001", {}, "s", "b")


# ---- through the unchanged skeleton -------------------------------------------

def test_opt_out_between_recheck_and_send_is_refused(tmp_path, mailer):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    graph = build_graph(ScriptedPlanner(send_script(**SEND)), [MailerTool(mailer)], ledger,
                        before_write=lambda: mailer.mutate_out_of_band("M-2001", opt_in_status="opted_out"))
    config = {"configurable": {"thread_id": "m"}}
    graph.invoke({"messages": [HumanMessage("send")], "run_id": "m"}, config)
    state = graph.invoke(None, config)
    assert state["outcome"] == "blocked"
    assert mailer.outbox() == []
    block = ledger.entries("m")[-1]["payload"]
    assert (block["stage"], block["changed_fields"]) == ("conditional_update", ["opt_in_status"])


def test_send_needs_a_prior_read_of_the_contact(tmp_path, mailer):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    graph = build_graph(ScriptedPlanner([("propose_send", SEND)]), [MailerTool(mailer)], ledger)
    state = graph.invoke({"messages": [HumanMessage("send")], "run_id": "m"},
                         {"configurable": {"thread_id": "m"}})
    assert state["outcome"] == "no_proposal"
    assert mailer.outbox() == []


REUSABLE = ["guardrail", "agent/graph.py", "scenarios/base.py", "scenarios/metrics.py", "scenarios/runner.py"]


@pytest.mark.parametrize("path", [p for r in REUSABLE for p in sorted((ROOT / r).glob("*.py"))
                                  or [ROOT / r]], ids=lambda p: str(p.relative_to(ROOT)))
def test_reusable_modules_import_no_tool_code(path):
    tree = ast.parse(path.read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not [m for m in imported if m == "tools" or m.startswith("tools.")]

"""LangGraph agent with the guardrail wired in front of the only write path.

Flow:
    agent <-> act            model reads the record, then proposes one edit
    approve                  guardrail derives approval conditions from the
                             agent's decision snapshot, logs approval
    -- interrupt --          approval wait; the world may change here
    recheck                  guardrail re-reads, evaluates, stores verdict in state
    effect | block           routed on the stored verdict only
    effect                   conditional_update (compare-and-set) closes TOCTOU

Decision snapshot: whatever read_record returned to the model is recorded in
state, with the time of the read. When the model calls propose_update, the act
node attaches that read to the proposal. The approval basis is what the agent
actually saw when it decided, not a fresh read taken when the proposal
arrives (which would silently absorb any change that landed in between). The
values come from the tool result, not from the model's arguments, so the model
cannot misstate what it read.

Replay safety: the live re-read happens inside the `recheck` node and its result
is written to state. The branch after it reads only `state["verdict"]`, never
the database, so replaying from a checkpoint takes the same branch.

The model never gets a write tool. Its tools are read_record (reads) and
propose_update (records intent only). Only guardrail code touches the ledger or
calls conditional_update.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from guardrail.ledger import Ledger
from guardrail.policy import POLICIES
from guardrail.recheck import Condition, evaluate, take_snapshot
from tools.crm_dependencies import dependency_specs
from tools.mock_crm import EDITABLE_FIELDS, MockCRM

MAX_RETRIES = 3

SYSTEM_PROMPT = (
    "You are a CRM operations agent. First call read_record to look up the "
    "customer. Then call propose_update exactly once with the single field "
    "change the request needs. You cannot write to the CRM directly: a "
    "separate system approves and executes your proposal."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    run_id: str
    retries: int
    reads: Dict[str, Dict[str, Any]]     # record_id -> {"values", "read_at"}: what the agent saw
    proposal: Optional[Dict[str, Any]]   # {record_id, field, new_value, decision: {values, read_at}}
    conditions: Optional[List[Dict[str, Any]]]  # approval basis, as Condition.to_dict()
    verdict: Optional[Dict[str, Any]]    # recheck result, see guardrail.recheck.Verdict
    outcome: Optional[str]               # "allowed" | "blocked" | "no_proposal"
    block_reason: Optional[str]


@tool
def read_record(record_id: str) -> str:
    """Read a customer record from the CRM by id (e.g. 'C-1001')."""
    raise NotImplementedError("executed by the act node")


@tool
def propose_update(record_id: str, field: str, new_value: str) -> str:
    """Propose changing one field of a customer record. Does not write; the change
    is approved and executed by a separate system."""
    raise NotImplementedError("executed by the act node")


AGENT_TOOLS = [read_record, propose_update]


def action_of(proposal: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The proposed action alone, without the attached decision snapshot."""
    if proposal is None:
        return None
    return {k: proposal[k] for k in ("record_id", "field", "new_value")}


def build_graph(
    model: Any,
    crm: MockCRM,
    ledger: Ledger,
    before_write: Optional[Callable[[], None]] = None,
    after_read: Optional[Callable[[], None]] = None,
    policy: str = "bounded",
):
    """Compile the agent graph.

    `model` is anything with .invoke(messages) -> AIMessage that emits tool calls
    for AGENT_TOOLS (a ChatOllama bound with tools, or ScriptedPlanner).
    `policy` names an entry in guardrail.policy.POLICIES.
    `after_read` and `before_write` are test seams that simulate the world
    changing after the agent's read, or between recheck and the write.
    """
    make_conditions = POLICIES[policy]

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    def act(state: AgentState) -> dict:
        last = state["messages"][-1]
        retries = state.get("retries", 0)
        if not isinstance(last, AIMessage) or not last.tool_calls:
            nudge = HumanMessage("You must respond with a tool call (read_record or propose_update).")
            return {"messages": [nudge], "retries": retries + 1}

        reads = dict(state.get("reads") or {})
        replies: List[ToolMessage] = []
        proposal = None
        for call in last.tool_calls:
            if call["name"] == "read_record":
                record_id = str(call["args"].get("record_id", "")).strip()
                record = crm.read_record(record_id)
                if record:
                    reads[record_id] = {"values": record, "read_at": _now()}
                content = json.dumps(record) if record else "error: no such record"
                if record and after_read is not None:
                    after_read()
            elif call["name"] == "propose_update" and proposal is None:
                proposal, error = _validate_proposal(call["args"], reads)
                content = error or "proposal submitted for approval"
                if error:
                    retries += 1
            else:
                content = f"error: unknown or repeated tool call {call['name']!r}"
                retries += 1
            replies.append(ToolMessage(content, tool_call_id=call["id"]))
        return {"messages": replies, "reads": reads, "proposal": proposal, "retries": retries}

    def route_after_act(state: AgentState) -> str:
        if state.get("proposal"):
            return "approve"
        if state.get("retries", 0) >= MAX_RETRIES:
            return "give_up"
        return "agent"

    def give_up(state: AgentState) -> dict:
        return {"outcome": "no_proposal"}

    # ---- guardrail nodes ------------------------------------------------

    def approve(state: AgentState) -> dict:
        p = state["proposal"]
        deps = dependency_specs(p["field"])
        decision = p["decision"]
        conditions = make_conditions(deps, p, decision["values"])
        ledger.append(state["run_id"], "approval", {
            "proposal": action_of(p),
            "policy": policy,
            "decision_read_at": decision["read_at"],
            "decision_snapshot": take_snapshot(decision["values"], [d.field for d in deps]),
            "conditions": [c.to_dict() for c in conditions],
        })
        return {"conditions": [c.to_dict() for c in conditions]}

    def recheck(state: AgentState) -> dict:
        p = state["proposal"]
        current = crm.read_record(p["record_id"]) or {}
        verdict = evaluate(_conditions(state), current)
        ledger.append(state["run_id"], "recheck", {"proposal": action_of(p), "verdict": verdict})
        return {"verdict": dict(verdict)}

    def route_on_verdict(state: AgentState) -> str:
        # Depends only on stored state, never on a live read (replay-safe).
        return "block" if state["verdict"]["changed"] else "effect"

    def block(state: AgentState) -> dict:
        v = state["verdict"]
        ledger.append(state["run_id"], "block", {
            "proposal": action_of(state["proposal"]),
            "stage": "recheck",
            "changed_fields": v["changed_fields"],
            "diffs": v["diffs"],
        })
        return {"outcome": "blocked", "block_reason": f"dependency changed: {v['changed_fields']}"}

    def effect(state: AgentState) -> dict:
        p = state["proposal"]
        conditions = _conditions(state)
        if before_write is not None:
            before_write()
        result = crm.conditional_update(
            p["record_id"],
            expected={c.field: c.value for c in conditions if c.op == "=="},
            changes={p["field"]: p["new_value"]},
            bounds=[(c.field, c.op, c.value) for c in conditions if c.op != "=="],
        )
        if result.written:
            ledger.append(state["run_id"], "allow", {"proposal": action_of(p), "write": result.reason})
            return {"outcome": "allowed"}
        # Recheck passed but the compare-and-set did not: something changed in
        # the window between recheck and write. Re-read only to say which field
        # moved in the ledger; the write was already refused.
        moved = evaluate(conditions, crm.read_record(p["record_id"]) or {})
        ledger.append(state["run_id"], "block", {
            "proposal": action_of(p),
            "stage": "conditional_update",
            "reason": result.reason,
            "changed_fields": moved["changed_fields"],
            "diffs": moved["diffs"],
        })
        return {"outcome": "blocked", "block_reason": f"conditional update refused: {result.reason}"}

    g = StateGraph(AgentState)
    for name, fn in [("agent", agent), ("act", act), ("give_up", give_up), ("approve", approve),
                     ("recheck", recheck), ("block", block), ("effect", effect)]:
        g.add_node(name, fn)
    g.add_edge(START, "agent")
    g.add_edge("agent", "act")
    g.add_conditional_edges("act", route_after_act, ["agent", "approve", "give_up"])
    g.add_edge("give_up", END)
    g.add_edge("approve", "recheck")
    g.add_conditional_edges("recheck", route_on_verdict, ["block", "effect"])
    g.add_edge("block", END)
    g.add_edge("effect", END)
    # Pause after approval: this is where a human would sign off, and where the
    # world can drift away from the approved snapshot.
    return g.compile(checkpointer=InMemorySaver(), interrupt_before=["recheck"])


def _conditions(state: AgentState) -> List[Condition]:
    return [Condition.from_dict(d) for d in state["conditions"]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_proposal(args: Dict[str, Any], reads: Dict[str, Dict[str, Any]]):
    """Return (proposal, None) or (None, error message for the model to retry on).

    The proposal carries the agent's own read of the record as its decision
    snapshot. A proposal for a record the agent never read is refused: there is
    no decision basis to approve it against.
    """
    record_id = str(args.get("record_id", "")).strip()
    field = str(args.get("field", "")).strip()
    new_value = args.get("new_value")
    if field not in EDITABLE_FIELDS:
        return None, f"error: field must be one of {sorted(EDITABLE_FIELDS)}"
    decision = reads.get(record_id)
    if decision is None:
        return None, f"error: call read_record for {record_id!r} before proposing a change to it"
    if new_value is None:
        return None, "error: new_value is required"
    # Keep the proposed value's type consistent with the column, so strict
    # equality and the compare-and-set behave predictably.
    if isinstance(decision["values"][field], int):
        try:
            new_value = int(new_value)
        except (TypeError, ValueError):
            return None, f"error: {field} must be an integer"
    else:
        new_value = str(new_value).strip()
    return {"record_id": record_id, "field": field, "new_value": new_value, "decision": decision}, None


# ---- models -------------------------------------------------------------

def make_ollama_model(model_name: str = "qwen2.5:7b-instruct", base_url: Optional[str] = None):
    from langchain_ollama import ChatOllama

    kwargs = {"model": model_name, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs).bind_tools(AGENT_TOOLS)


class ScriptedPlanner:
    """Deterministic stand-in for the LLM: read the record, then propose `change`.

    Used by the unit tests and by `run.py --scripted` so the guardrail can be
    exercised without Ollama. Everything downstream of the model is identical.
    """

    def __init__(self, record_id: str, field: str, new_value: Any):
        self.record_id, self.field, self.new_value = record_id, field, new_value
        self._calls = 0

    def invoke(self, messages: List[AnyMessage]) -> AIMessage:
        self._calls += 1
        has_read = any(isinstance(m, ToolMessage) for m in messages)
        if not has_read:
            call = {"name": "read_record", "args": {"record_id": self.record_id}}
        else:
            call = {"name": "propose_update", "args": {
                "record_id": self.record_id, "field": self.field, "new_value": self.new_value,
            }}
        return AIMessage("", tool_calls=[{**call, "id": f"call_{self._calls}", "type": "tool_call"}])

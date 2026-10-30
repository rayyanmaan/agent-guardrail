"""LangGraph agent with the guardrail wired in front of every effect.

Flow:
    agent <-> act            model reads what it needs, then proposes one effect
    approve                  guardrail derives approval conditions from the
                             agent's decision snapshot, logs approval
    -- interrupt --          approval wait; the world may change here
    recheck                  guardrail re-reads, evaluates, stores verdict in state
    effect | block           routed on the stored verdict only
    effect                   the tool's conditional effect (compare-and-set) closes TOCTOU

Tools are passed in as adapters (guardrail.tool.GuardedTool); nothing here
knows about any particular tool.

Decision snapshot: whatever a read tool returned to the model is recorded in
state, with the time of the read. When the model calls a propose tool, the act
node attaches that read to the proposal. The approval basis is what the agent
actually saw when it decided, not a fresh read taken when the proposal
arrives (which would silently absorb any change that landed in between). The
values come from the tool result, not from the model's arguments, so the model
cannot misstate what it read.

Conditions are rebuilt from the proposal (a pure function of the action and
the decision snapshot) wherever they are needed, rather than read back from
state, because canonicalizers are functions. The copy in state is a record.

Replay safety: the live re-read happens inside the `recheck` node and its result
is written to state. The branch after it reads only `state["verdict"]`, never
the store, so replaying from a checkpoint takes the same branch.

The model never gets a write tool. Its tools are reads and proposals, which
record intent only. Only guardrail code touches the ledger or calls an effect.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from guardrail.ledger import Ledger
from guardrail.policy import POLICIES
from guardrail.recheck import Condition, evaluate, take_snapshot
from guardrail.tool import GuardedTool

MAX_RETRIES = 3

SYSTEM_PROMPT = (
    "You are an operations agent. First use the read tools to look up every "
    "record you need. Then call exactly one propose tool with the single change "
    "or message the request needs. You cannot change anything directly: a "
    "separate system approves and executes your proposal."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    run_id: str
    retries: int
    reads: Dict[str, Dict[str, Any]]     # "tool:target" -> {"values", "read_at"}: what the agent saw
    proposal: Optional[Dict[str, Any]]   # {tool, action, decision: {values, read_at}}
    conditions: Optional[List[Dict[str, Any]]]  # approval basis, as Condition.to_dict()
    verdict: Optional[Dict[str, Any]]    # recheck result, see guardrail.recheck.Verdict
    recheck_values: Optional[Dict[str, Any]]  # equality fields as read at recheck; the write's compare-and-set basis
    outcome: Optional[str]               # "allowed" | "blocked" | "no_proposal"
    # How often the act node turned the model away, by kind: "unread_target"
    # (proposed before reading), "invalid_args", "unknown_tool", "no_tool_call".
    refusals: Dict[str, int]
    block_reason: Optional[str]


def action_of(proposal: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The proposed action and its tool, without the attached decision snapshot."""
    if proposal is None:
        return None
    return {"tool": proposal["tool"], **proposal["action"]}


def build_graph(
    model: Any,
    tools: Sequence[GuardedTool],
    ledger: Ledger,
    before_write: Optional[Callable[[], None]] = None,
    after_read: Optional[Callable[[], None]] = None,
    after_write: Optional[Callable[[], None]] = None,
    policy: str = "bounded",
):
    """Compile the agent graph.

    `model` is anything with .invoke(messages) -> AIMessage that emits tool calls
    for the tools' read/propose schemas (a ChatOllama bound with them, or
    ScriptedPlanner). `policy` names an entry in guardrail.policy.POLICIES.
    `after_read`, `before_write` and `after_write` are test seams: the world
    changing after the agent's read, or between recheck and the write, and a
    fault right after a write commits.
    """
    make_conditions = POLICIES[policy]
    by_name = {t.name: t for t in tools}
    read_tools = {t.read_tool.name: t for t in tools}
    propose_tools = {t.propose_tool.name: t for t in tools}

    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    def act(state: AgentState) -> dict:
        last = state["messages"][-1]
        retries = state.get("retries", 0)
        refusals = dict(state.get("refusals") or {})

        def refuse(kind: str) -> None:
            refusals[kind] = refusals.get(kind, 0) + 1

        if not isinstance(last, AIMessage) or not last.tool_calls:
            refuse("no_tool_call")
            nudge = HumanMessage("You must respond with a tool call.")
            return {"messages": [nudge], "retries": retries + 1, "refusals": refusals}

        reads = dict(state.get("reads") or {})
        replies: List[ToolMessage] = []
        proposal = None
        for call in last.tool_calls:
            name, args = call["name"], call["args"]
            if name in read_tools:
                tool = read_tools[name]
                target = str(args.get(tool.target_arg, "")).strip()
                values = tool.read(target)
                if values:
                    reads[_read_key(tool, target)] = {"values": values, "read_at": _now()}
                content = json.dumps(values) if values else f"error: no {tool.name} record {target!r}"
                if values and after_read is not None:
                    after_read()
            elif name in propose_tools and proposal is None:
                proposal, error, kind = _make_proposal(propose_tools[name], args, reads)
                content = error or "proposal submitted for approval"
                if error:
                    retries += 1
                    refuse(kind)
            else:
                content = f"error: unknown or repeated tool call {name!r}"
                retries += 1
                refuse("unknown_tool")
            replies.append(ToolMessage(content, tool_call_id=call["id"]))
        return {"messages": replies, "reads": reads, "proposal": proposal, "retries": retries,
                "refusals": refusals}

    def route_after_act(state: AgentState) -> str:
        if state.get("proposal"):
            return "approve"
        if state.get("retries", 0) >= MAX_RETRIES:
            return "give_up"
        return "agent"

    def give_up(state: AgentState) -> dict:
        return {"outcome": "no_proposal"}

    # ---- guardrail nodes ------------------------------------------------

    def conditions_for(p: Dict[str, Any]) -> List[Condition]:
        tool = by_name[p["tool"]]
        deps = tool.dependencies(p["action"])
        return make_conditions(deps, p["action"], p["decision"]["values"], tool.writes(p["action"]))

    def approve(state: AgentState) -> dict:
        p = state["proposal"]
        deps = by_name[p["tool"]].dependencies(p["action"])
        decision = p["decision"]
        conditions = conditions_for(p)
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
        tool = by_name[p["tool"]]
        current = tool.read(p["action"][tool.target_arg]) or {}
        conditions = conditions_for(p)
        verdict = evaluate(conditions, current)
        ledger.append(state["run_id"], "recheck", {"proposal": action_of(p), "verdict": verdict})
        seen = {c.field: current.get(c.field) for c in conditions if c.op == "=="}
        return {"verdict": dict(verdict), "recheck_values": seen}

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
        tool = by_name[p["tool"]]
        conditions = conditions_for(p)
        if before_write is not None:
            before_write()
        # The write requires every equality field to still hold exactly the raw
        # value the recheck accepted. Without canonicalization or convergence
        # that is the decision snapshot itself. With them, it may be an
        # equivalent value (e.g. a lowercased address), and the write must not
        # re-reject what the recheck accepted. Either way any change after the
        # recheck, even an equivalent one, is refused: the write stays exact.
        result = tool.apply(
            p["action"],
            expected=state["recheck_values"],
            bounds=[(c.field, c.op, c.value) for c in conditions if c.op != "=="],
        )
        if result.written:
            if after_write is not None:
                after_write()
            ledger.append(state["run_id"], "allow", {"proposal": action_of(p), "write": result.reason})
            return {"outcome": "allowed"}
        # Recheck passed but the compare-and-set did not: something changed in
        # the window between recheck and write. Re-read only to say which field
        # moved in the ledger; the write was already refused.
        moved = evaluate(conditions, tool.read(p["action"][tool.target_arg]) or {})
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_key(tool: GuardedTool, target: str) -> str:
    return f"{tool.name}:{target}"


def _make_proposal(tool: GuardedTool, args: Dict[str, Any], reads: Dict[str, Dict[str, Any]]):
    """Return (proposal, None, None), or (None, error for the model, refusal kind).

    The proposal carries the agent's own read of the target as its decision
    snapshot. A proposal for a target the agent never read is refused: there is
    no decision basis to approve it against.
    """
    target = str(args.get(tool.target_arg, "")).strip()
    decision = reads.get(_read_key(tool, target))
    if decision is None:
        return (None, f"error: call {tool.read_tool.name} for {target!r} before proposing a change to it",
                "unread_target")
    action, error = tool.validate({**args, tool.target_arg: target}, decision["values"])
    if error:
        return None, error, "invalid_args"
    return {"tool": tool.name, "action": action, "decision": decision}, None, None


# ---- models -------------------------------------------------------------

def make_ollama_model(tool_schemas: Sequence[Any], model_name: str = "qwen2.5:7b-instruct",
                      base_url: Optional[str] = None):
    from langchain_ollama import ChatOllama

    kwargs = {"model": model_name, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs).bind_tools(list(tool_schemas))


class ScriptedPlanner:
    """Deterministic stand-in for the LLM: emits a fixed sequence of tool calls.

    `calls` is a sequence of (tool_name, args), one per turn, e.g. a read then a
    proposal. Used by the unit tests and by `run.py --scripted` so the guardrail
    can be exercised without Ollama. Everything downstream of the model is identical.
    """

    def __init__(self, calls: Sequence[Any]):
        self.calls = [(name, dict(args)) for name, args in calls]
        self._turns = 0

    def invoke(self, messages: List[AnyMessage]) -> AIMessage:
        self._turns += 1
        step = sum(isinstance(m, ToolMessage) for m in messages)
        if step >= len(self.calls):
            return AIMessage("done")
        name, args = self.calls[step]
        return AIMessage("", tool_calls=[{"name": name, "args": args,
                                          "id": f"call_{self._turns}", "type": "tool_call"}])

"""What a tool supplies to run behind the guardrail.

The skeleton (graph, recheck, policy, ledger) never imports a tool. Each tool
hands it an adapter with this shape. Writing one is the whole cost of adding a
tool: a read, a validation of the model's arguments, a dependency set, and a
conditional effect that refuses to fire if the approved conditions no longer hold.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

from langchain_core.tools import BaseTool

from guardrail.policy import Dependency


class EffectResult(Protocol):
    written: bool
    reason: str


class GuardedTool(Protocol):
    name: str              # e.g. "crm"; recorded on every proposal and ledger entry
    target_arg: str        # argument naming the target in both model tools, e.g. "record_id"
    read_tool: BaseTool    # schema of the model's read tool
    propose_tool: BaseTool # schema of the model's propose tool (records intent only)

    def read(self, target: str) -> Optional[Dict[str, Any]]:
        """Current state of `target`, or None if it does not exist."""

    def validate(self, args: Dict[str, Any], decision: Dict[str, Any]
                 ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Turn the model's propose arguments into an action.

        `decision` is what the agent read for this target. Returns
        (action, None), or (None, error) for the model to retry on. The action
        must include `target_arg`.
        """

    def dependencies(self, action: Dict[str, Any]) -> Sequence[Dependency]:
        """Explicit dependency set the action relies on."""

    def writes(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Fields of the target the action sets, and the values it sets them to.

        Used only for the canonical policy's convergence rule. A field not
        listed here can never converge.
        """

    def apply(self, action: Dict[str, Any], expected: Dict[str, Any],
              bounds: Sequence[Tuple[str, str, Any]]) -> EffectResult:
        """Perform the effect only if every expected value and bound still holds,
        checked and written atomically."""

"""Adapter that runs the mock mailer behind the guardrail (see guardrail/tool.py).

The first half is what the guardrail needs. The second half is harness support
for scenarios, as in crm_tool.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.tools import tool

from tools.mailer_dependencies import dependency_specs
from tools.mock_mailer import MockMailer, SendResult


@tool
def read_contact(contact_id: str) -> str:
    """Read a mailing contact by id (e.g. 'M-2001')."""
    raise NotImplementedError("executed by the guardrail graph")


@tool
def propose_send(contact_id: str, subject: str, body: str) -> str:
    """Propose sending an email to a contact. Does not send; the message is
    approved and sent by a separate system."""
    raise NotImplementedError("executed by the guardrail graph")


class MailerTool:
    name = "mailer"
    target_arg = "contact_id"
    read_tool = read_contact
    propose_tool = propose_send

    def __init__(self, mailer: MockMailer):
        self.mailer = mailer

    @classmethod
    def open(cls, path: str) -> "MailerTool":
        return cls(MockMailer(path))

    def close(self) -> None:
        self.mailer.close()

    # ---- guardrail interface ------------------------------------------------

    def read(self, target: str) -> Optional[Dict[str, Any]]:
        return self.mailer.read_contact(target)

    def validate(self, args: Dict[str, Any], decision: Dict[str, Any]):
        subject = str(args.get("subject") or "").strip()
        body = str(args.get("body") or "").strip()
        if not subject or not body:
            return None, "error: subject and body are both required"
        return {"contact_id": args["contact_id"], "subject": subject, "body": body}, None

    def dependencies(self, action: Dict[str, Any]):
        return dependency_specs(action)

    def writes(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # A send adds to the outbox; it sets no field on the contact, so no
        # contact field can ever converge.
        return {}

    def apply(self, action: Dict[str, Any], expected: Dict[str, Any],
              bounds: Sequence[Tuple[str, str, Any]]) -> SendResult:
        return self.mailer.conditional_send(
            action["contact_id"], expected, action["subject"], action["body"], bounds)

    # ---- scenario harness ---------------------------------------------------

    def seed(self, rows: Sequence[Dict[str, Any]]) -> None:
        for row in rows:
            self.mailer.seed(row)

    def mutate_out_of_band(self, target: str, changes: Dict[str, Any]) -> None:
        self.mailer.mutate_out_of_band(target, **changes)

    def dump(self) -> Dict[str, Any]:
        return {"records": {r["id"]: r for r in self.mailer.read_all_contacts()},
                "effects": self.mailer.outbox()}

    @staticmethod
    def simulate_effect(action: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """What the effect should do, as plain dict operations: one outbox row
        addressed to the contact's current email."""
        return [{"contact_id": action["contact_id"],
                 "to_email": records[action["contact_id"]]["recipient_email"],
                 "subject": action["subject"], "body": action["body"]}]

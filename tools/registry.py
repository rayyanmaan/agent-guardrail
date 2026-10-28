"""The tools this project runs behind the guardrail, by name."""

from tools.crm_tool import CRMTool
from tools.mailer_tool import MailerTool

TOOL_TYPES = {CRMTool.name: CRMTool, MailerTool.name: MailerTool}
TOOL_SCHEMAS = [schema for t in TOOL_TYPES.values() for schema in (t.read_tool, t.propose_tool)]

"""The tool registry and the verifier, served over the Model Context Protocol.

Any MCP client - Claude Desktop, Claude Code, an IDE, another agent - gets the same
tools the LangGraph agent uses, plus `verify_text`: hand it a paragraph and it
returns every checkable claim with its verdict and the tool values behind it.

    python -m drugagent.mcp_server                      # stdio
    claude mcp add drugagent -- python -m drugagent.mcp_server
"""
from __future__ import annotations

import functools

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import tools
from .agent import default_verifier
from .extract import extract
from .verifier import summarize

server = MCPServer(
    "drugagent",
    instructions=("Cheminformatics, omics and solubility-prediction tools whose method-dependent values are "
                  "returned per method. Call verify_text on any draft before presenting numbers from it."),
)


def anticipated(fn):
    """Bad input (an unknown compound, an invalid SMILES) is the model's to read and fix,
    so it becomes a ToolError, whose message reaches the client; anything else is a crash."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError, RuntimeError) as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


for _tool in tools.REGISTRY.values():
    server.tool(name=_tool.name, description=_tool.description)(anticipated(_tool.fn))


@server.tool(name="verify_text", description=(
    "Extract every checkable numeric claim (molecular properties, Rule-of-Five verdicts, solubility "
    "predictions, differential-expression results) from a text and recompute each with the tools. "
    "Verdicts: supported, contradicted, ambiguous (true under one method only), unverifiable, unqualified "
    "(a prediction without its interval). Numbers no claim covers are listed as unattributed."))
def verify_text(text: str, subject: str = "", dataset: str = "demo") -> dict:
    found = extract(text, subject=subject or None, dataset=dataset)
    checks = default_verifier(dataset).check_all(found.claims)
    return {
        "summary": summarize(checks),
        "claims": [{"claim": c.claim.describe(), "text": c.claim.text, "verdict": c.verdict.value,
                    "tool_values": c.expected(), "note": c.note} for c in checks],
        "unattributed_numbers": [{"number": n, "context": text[max(0, s - 30):e + 20]} for s, e, n in found.unattributed],
    }


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()

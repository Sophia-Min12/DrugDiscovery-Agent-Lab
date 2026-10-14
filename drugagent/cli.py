"""Command line.

    python -m drugagent tools
    python -m drugagent verify "Ibuprofen has a TPSA of 83.6." [--subject ibuprofen] [--dataset demo]
    python -m drugagent ask "Summarise the drug-likeness of imatinib." [--model llama3.1:8b] [--closed-book]
    python -m drugagent sandbox script.py [--backend docker|subprocess]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="drugagent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tools", help="list the tools the agent and the MCP server offer")
    verify = sub.add_parser("verify", help="check every numeric claim in a text")
    verify.add_argument("text")
    verify.add_argument("--subject", default=None)
    verify.add_argument("--dataset", default="demo")
    ask = sub.add_parser("ask", help="run the agent graph with a local Ollama model")
    ask.add_argument("question")
    ask.add_argument("--model", default="llama3.1:8b")
    ask.add_argument("--closed-book", action="store_true", help="no tools: answer from the model's memory")
    ask.add_argument("--max-revisions", type=int, default=2)
    box = sub.add_parser("sandbox", help="run a Python file in the code sandbox")
    box.add_argument("path")
    box.add_argument("--backend", default="auto", choices=["auto", "docker", "subprocess"])
    box.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    if args.command == "tools":
        from . import tools
        print(tools.catalog())
    elif args.command == "verify":
        from .agent import default_verifier, verification_table
        from .extract import extract
        found = extract(args.text, subject=args.subject, dataset=args.dataset)
        print(verification_table(default_verifier(args.dataset).check_all(found.claims)))
        if found.unattributed:
            print("\nunattributed numbers:", ", ".join(n for _, _, n in found.unattributed))
    elif args.command == "ask":
        from .agent import run
        from .llm import OllamaLLM
        state = run(args.question, OllamaLLM(args.model), use_tools=not args.closed_book,
                    max_revisions=args.max_revisions)
        print(state["report"])
        print(f"\n(plan: {state['plan_source']}, revisions: {state['revisions']}, llm calls: {state['llm_calls']})")
    elif args.command == "sandbox":
        from .sandbox import run_python
        result = run_python(Path(args.path).read_text(encoding="utf-8"), timeout=args.timeout, backend=args.backend)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        print(json.dumps({"backend": result.backend, "isolated": result.isolated, "returncode": result.returncode,
                          "timed_out": result.timed_out, "outputs": sorted(result.outputs)}), file=sys.stderr)


if __name__ == "__main__":
    main()

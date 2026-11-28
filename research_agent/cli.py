from __future__ import annotations

import argparse
import json

from .graph import build_default_graph
from research_observability.metrics import track_research_run
from research_observability.tracing import observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-grounded multi-agent web research.")
    parser.add_argument("question", help="A research question")
    parser.add_argument("--output", help="Optional path for JSON report output")
    args = parser.parse_args()

    with track_research_run("cli"):
        with observation("research.run", metadata={"research.entrypoint": "cli"}) as span:
            result = build_default_graph().invoke({"question": args.question})
            report = result["report"]
            if hasattr(span, "update"):
                span.update(
                    output={
                        "run_id": report.get("run_id"),
                        "source_count": len(report.get("sources", [])),
                        "claim_count": len(report.get("claim_verifications", [])),
                        "research_rounds": report.get("research_rounds", 0),
                    }
                )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

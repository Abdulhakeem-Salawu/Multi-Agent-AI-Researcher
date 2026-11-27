from __future__ import annotations

import argparse
import json

from .graph import build_default_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-grounded multi-agent web research.")
    parser.add_argument("question", help="A research question")
    parser.add_argument("--output", help="Optional path for JSON report output")
    args = parser.parse_args()

    result = build_default_graph().invoke({"question": args.question})
    rendered = json.dumps(result["report"], indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

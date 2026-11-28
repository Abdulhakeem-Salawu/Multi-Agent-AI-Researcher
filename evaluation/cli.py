from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate deterministic retrieval and report-integrity metrics."
    )
    parser.add_argument("report", type=Path, help="Path to a saved research report JSON file.")
    parser.add_argument(
        "--relevant-source-url",
        action="append",
        default=None,
        help="Human-labelled relevant source URL. Repeat for multiple relevant sources.",
    )
    parser.add_argument("--k", type=int, default=5, help="Cut-off for retrieval metrics.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    args = parser.parse_args()

    with args.report.open(encoding="utf-8") as file:
        report = json.load(file)
    result = evaluate_report(
        report,
        relevant_source_urls=args.relevant_source_url,
        retrieval_k=args.k,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

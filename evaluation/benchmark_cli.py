from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agreement import case_inter_annotator_agreement
from .dataset import load_dataset, validate_dataset_file
from .schemas import RunAnnotation
from .scoring import score_report_against_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and score the human-adjudicated research benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate benchmark labels and frozen source snapshots.")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--require-ready", action="store_true")
    validate.add_argument("--verify-snapshots", action="store_true")
    validate.add_argument("--repo-root", type=Path, default=Path("."))

    agreement = subparsers.add_parser("agreement", help="Calculate inter-annotator agreement for a case.")
    agreement.add_argument("dataset", type=Path)
    agreement.add_argument("--case-id", required=True)

    score = subparsers.add_parser("score", help="Score a run against one adjudicated benchmark case.")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--case-id", required=True)
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--run-annotation", type=Path, required=True)
    score.add_argument("--k", type=int, default=5)
    score.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "validate":
        result = validate_dataset_file(
            args.dataset,
            require_ready=args.require_ready,
            verify_snapshots=args.verify_snapshots,
            repo_root=args.repo_root,
        )
    elif args.command == "agreement":
        dataset = load_dataset(args.dataset)
        case = next((item for item in dataset.cases if item.case_id == args.case_id), None)
        if case is None:
            parser.error(f"Unknown case_id: {args.case_id}")
        result = case_inter_annotator_agreement(case)
    else:
        dataset = load_dataset(args.dataset)
        case = next((item for item in dataset.cases if item.case_id == args.case_id), None)
        if case is None:
            parser.error(f"Unknown case_id: {args.case_id}")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        annotation = RunAnnotation.model_validate_json(
            args.run_annotation.read_text(encoding="utf-8")
        )
        result = score_report_against_case(
            report,
            case,
            annotation,
            k=args.k,
            benchmark_id=dataset.benchmark_id,
            benchmark_version=dataset.version,
        )

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if getattr(args, "output", None):
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.command == "validate" and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

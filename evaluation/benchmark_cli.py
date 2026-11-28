from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agreement import case_inter_annotator_agreement
from .annotation_workflow import (
    export_adjudication_form,
    export_annotation_forms,
    freeze_case_snapshot,
    import_adjudication_form,
    import_annotation_forms,
    register_source_snapshot,
)
from .dataset import load_dataset, validate_dataset_file
from .schemas import RunAnnotation
from .scoring import score_report_against_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, annotate, and score the human-adjudicated research benchmark.")
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

    export_forms = subparsers.add_parser("export-annotations", help="Create isolated forms for independent annotators.")
    export_forms.add_argument("dataset", type=Path)
    export_forms.add_argument("--output-dir", type=Path, required=True)
    export_forms.add_argument("--annotator-ids", nargs="+", required=True)
    export_forms.add_argument("--case-id", action="append")
    export_forms.add_argument("--overwrite", action="store_true")

    import_forms = subparsers.add_parser("import-annotations", help="Validate completed forms and write a new dataset.")
    import_forms.add_argument("dataset", type=Path)
    import_forms.add_argument("--annotation", type=Path, action="append", required=True)
    import_forms.add_argument("--output", type=Path, required=True)
    import_forms.add_argument("--repo-root", type=Path, default=Path("."))
    import_forms.add_argument("--overwrite", action="store_true")

    export_adjudication = subparsers.add_parser("export-adjudication", help="Prepare a separate adjudicator form.")
    export_adjudication.add_argument("dataset", type=Path)
    export_adjudication.add_argument("--case-id", required=True)
    export_adjudication.add_argument("--adjudicator-id", required=True)
    export_adjudication.add_argument("--output", type=Path, required=True)
    export_adjudication.add_argument("--overwrite", action="store_true")

    import_adjudication = subparsers.add_parser("import-adjudication", help="Validate final adjudicated labels.")
    import_adjudication.add_argument("dataset", type=Path)
    import_adjudication.add_argument("--form", type=Path, required=True)
    import_adjudication.add_argument("--output", type=Path, required=True)
    import_adjudication.add_argument("--repo-root", type=Path, default=Path("."))
    import_adjudication.add_argument("--overwrite", action="store_true")

    register_source = subparsers.add_parser("register-source", help="Register explicit rights and a permitted source snapshot.")
    register_source.add_argument("dataset", type=Path)
    register_source.add_argument("--case-id", required=True)
    register_source.add_argument("--source-id", required=True)
    register_source.add_argument(
        "--redistribution",
        choices=["redistributable", "excerpt_only", "metadata_only", "restricted"],
        required=True,
    )
    register_source.add_argument("--license-id", required=True)
    register_source.add_argument("--input-file", type=Path)
    register_source.add_argument("--controlled-ref")
    register_source.add_argument("--confirm-rights", action="store_true")
    register_source.add_argument("--output", type=Path, required=True)
    register_source.add_argument("--repo-root", type=Path, default=Path("."))
    register_source.add_argument("--snapshot-dir", type=Path, default=Path("evaluation/snapshots"))
    register_source.add_argument("--overwrite-snapshot", action="store_true")
    register_source.add_argument("--overwrite-dataset", action="store_true")

    freeze = subparsers.add_parser("freeze-case", help="Verify source registrations and freeze one case.")
    freeze.add_argument("dataset", type=Path)
    freeze.add_argument("--case-id", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--repo-root", type=Path, default=Path("."))
    freeze.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate_dataset_file(
                args.dataset,
                require_ready=args.require_ready,
                verify_snapshots=args.verify_snapshots,
                repo_root=args.repo_root,
            )
            exit_code = 0 if result["valid"] else 1
        elif args.command == "agreement":
            dataset = load_dataset(args.dataset)
            case = next((item for item in dataset.cases if item.case_id == args.case_id), None)
            if case is None:
                parser.error(f"Unknown case_id: {args.case_id}")
            result = case_inter_annotator_agreement(case)
            exit_code = 0
        elif args.command == "score":
            dataset = load_dataset(args.dataset)
            case = next((item for item in dataset.cases if item.case_id == args.case_id), None)
            if case is None:
                parser.error(f"Unknown case_id: {args.case_id}")
            report = json.loads(args.report.read_text(encoding="utf-8"))
            annotation = RunAnnotation.model_validate_json(args.run_annotation.read_text(encoding="utf-8"))
            result = score_report_against_case(
                report,
                case,
                annotation,
                k=args.k,
                benchmark_id=dataset.benchmark_id,
                benchmark_version=dataset.version,
            )
            exit_code = 0
        elif args.command == "export-annotations":
            paths = export_annotation_forms(
                args.dataset,
                args.output_dir,
                args.annotator_ids,
                case_ids=args.case_id,
                overwrite=args.overwrite,
            )
            result = {"exported_count": len(paths), "files": paths}
            exit_code = 0
        elif args.command == "import-annotations":
            result = import_annotation_forms(
                args.dataset,
                args.annotation,
                args.output,
                repo_root=args.repo_root,
                overwrite=args.overwrite,
            )
            exit_code = 0
        elif args.command == "export-adjudication":
            output = export_adjudication_form(
                args.dataset,
                args.case_id,
                args.adjudicator_id,
                args.output,
                overwrite=args.overwrite,
            )
            result = {"case_id": args.case_id, "form": output}
            exit_code = 0
        elif args.command == "import-adjudication":
            result = import_adjudication_form(
                args.dataset,
                args.form,
                args.output,
                repo_root=args.repo_root,
                overwrite=args.overwrite,
            )
            exit_code = 0
        elif args.command == "register-source":
            result = register_source_snapshot(
                args.dataset,
                args.case_id,
                args.source_id,
                args.redistribution,
                args.license_id,
                args.output,
                input_file=args.input_file,
                controlled_ref=args.controlled_ref,
                confirm_rights=args.confirm_rights,
                repo_root=args.repo_root,
                snapshot_dir=args.snapshot_dir,
                overwrite_snapshot=args.overwrite_snapshot,
                overwrite_dataset=args.overwrite_dataset,
            )
            exit_code = 0
        else:
            result = freeze_case_snapshot(
                args.dataset,
                args.case_id,
                args.output,
                repo_root=args.repo_root,
                overwrite=args.overwrite,
            )
            exit_code = 0
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if getattr(args, "output", None) and args.command in {"validate", "agreement", "score"}:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

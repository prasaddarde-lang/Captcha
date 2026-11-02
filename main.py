"""Universal CAPTCHA Management Framework - command-line entry point.

Run a task (JSON in -> JSON out; exit code = status code):
    python main.py run --input-file task.json [--output-file result.json] [--platform uipath]
    python main.py run --input-json "{\"task_id\": \"RPA_001\", ...}"
    type task.json | python main.py run
    python main.py --input-file task.json            (``run`` is the default command)

Operators:
    python main.py operator list
    python main.py operator complete <request_id> --operator "Jane Doe"
    python main.py operator reject   <request_id> --operator "Jane Doe" --notes "wrong page"

Learning / analytics:
    python main.py dataset generate [--per-label 20] [--images 20] [--mock-site]
    python main.py model list | evaluate [--version V] | train | approve V --approver NAME | deploy V | reject V --approver NAME
    python main.py feedback --execution-id 12 --actual-type hcaptcha --reviewer "Jane"
    python main.py report

REST API:
    python main.py serve [--host 127.0.0.1] [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from core.response_handler import EXIT_CODES, Status, error_response  # noqa: E402
from core.security import setup_logging  # noqa: E402

COMMANDS = {"run", "operator", "dataset", "model", "feedback", "report", "serve"}


def _emit(text: str, output_file: str | None) -> None:
    from integrations.generic_rpa import write_output

    if output_file:
        write_output(output_file, text)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def cmd_run(args: argparse.Namespace) -> int:
    from integrations.generic_rpa import get_adapter

    adapter = get_adapter(args.platform)
    if args.input_json:
        raw = args.input_json
    elif args.input_file:
        try:
            raw = Path(args.input_file).read_text(encoding="utf-8-sig")
        except OSError as exc:
            resp = error_response(None, Status.INVALID_INPUT, f"Cannot read input file: {exc.strerror}")
            _emit(adapter.format_output(resp), args.output_file)
            return resp.exit_code
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        resp = error_response(None, Status.INVALID_INPUT, "No input: use --input-file, --input-json or stdin")
        _emit(adapter.format_output(resp), args.output_file)
        return resp.exit_code

    # Command-line overrides
    if args.timeout or args.mode:
        try:
            data = json.loads(raw.strip().lstrip("﻿"))
            if isinstance(data, dict):
                if args.timeout:
                    data["timeout"] = args.timeout
                if args.mode:
                    data["mode"] = args.mode
                raw = json.dumps(data)
        except json.JSONDecodeError:
            pass  # adapter reports the JSON error
    resp = adapter.handle(raw)
    _emit(adapter.format_output(resp), args.output_file)
    return resp.exit_code


def _framework():
    from core.framework import CaptchaFramework
    return CaptchaFramework()


def cmd_operator(args: argparse.Namespace) -> int:
    from core.human_verification import format_pending_table

    fw = _framework()
    if args.op == "list":
        rows = fw.hv.pending() if not args.all else fw.db.list_verifications()
        print(json.dumps(rows, indent=2) if args.json else format_pending_table(rows))
        return 0
    if not args.operator:
        print("--operator NAME is required", file=sys.stderr)
        return 40
    fn = fw.hv.complete if args.op == "complete" else fw.hv.reject
    ok = fn(args.request_id, args.operator, args.notes)
    print(json.dumps({"request_id": args.request_id, "updated": ok,
                      "status": (fw.hv.get(args.request_id) or {}).get("status")}))
    return 0 if ok else 21


def cmd_dataset(args: argparse.Namespace) -> int:
    from learning.dataset_manager import SyntheticDatasetManager

    dm = SyntheticDatasetManager()
    out = {"html_samples": len(dm.generate_html_dataset(args.per_label, args.seed)), "root": str(dm.root)}
    if args.images:
        out["images"] = len(dm.generate_image_dataset(args.images, args.seed))
    if args.mock_site:
        out["mock_site"] = str(dm.write_mock_site())
    print(json.dumps(out, indent=2))
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    from core.classifier import CaptchaClassifier, load_model_file
    from learning.dataset_manager import SyntheticDatasetManager
    from learning.model_evaluation import ApprovalError, ModelRegistry, evaluate_html

    fw = _framework()
    reg = ModelRegistry(fw.db, fw.settings)
    reg.ensure_baseline()
    try:
        if args.op == "list":
            rows = [{k: m[k] for k in ("version", "status", "accuracy", "macro_f1", "approved_by", "created_at")}
                    for m in fw.db.list_models()]
            print(json.dumps(rows, indent=2))
        elif args.op == "evaluate":
            data = SyntheticDatasetManager().load_labeled_html()
            if args.version:
                m = reg.evaluate_version(args.version, data)
            else:
                m = evaluate_html(CaptchaClassifier(reg.active_model()), data)
            if not args.details:
                m.pop("misclassified", None)
            print(json.dumps(m, indent=2))
        elif args.op == "train":
            res = reg.train_candidate(SyntheticDatasetManager().load_labeled_html(), created_by=args.by or "cli")
            print(json.dumps({"version": res["version"], "status": res["status"],
                              "candidate_accuracy": res["candidate"]["accuracy"],
                              "baseline_accuracy": res["baseline"]["accuracy"],
                              "next_step": f"python main.py model approve {res['version']} --approver \"<your name>\""},
                             indent=2))
        elif args.op == "approve":
            print(json.dumps(reg.approve(args.version, args.approver or "", args.allow_regression), indent=2, default=str))
        elif args.op == "reject":
            print(json.dumps(reg.reject(args.version, args.approver or "cli"), indent=2, default=str))
        elif args.op == "deploy":
            m = reg.deploy(args.version)
            load_model_file(m["file_path"])
            print(json.dumps(m, indent=2, default=str))
    except ApprovalError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 41
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    fw = _framework()
    fid = fw.db.record_feedback(args.actual_type, args.execution_id, reviewer=args.reviewer)
    print(json.dumps({"feedback_id": fid}))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from core.classifier import CaptchaClassifier
    from learning.dataset_manager import SyntheticDatasetManager
    from learning.model_evaluation import ModelRegistry, evaluate_html
    from learning.performance_report import write_report

    fw = _framework()
    reg = ModelRegistry(fw.db, fw.settings)
    evaluation = evaluate_html(CaptchaClassifier(reg.active_model()), SyntheticDatasetManager().load_labeled_html())
    paths = write_report(fw.db, fw.settings.log_dir / "reports", evaluation)
    print(json.dumps(paths, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from api.server import serve
    serve(args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="captcha_rpa_framework", description="Universal CAPTCHA Management Framework for RPA")
    p.add_argument("--verbose", "-v", action="store_true", help="log diagnostics to stderr")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="execute a task (JSON in/out)")
    r.add_argument("--input-file", "-i")
    r.add_argument("--input-json", "-j")
    r.add_argument("--output-file", "-o")
    r.add_argument("--platform", "-p", help="output/input conventions: uipath, automation_anywhere, blue_prism, power_automate, generic")
    r.add_argument("--timeout", "-t", type=int, help="override timeout (seconds)")
    r.add_argument("--mode", "-m", choices=["production", "test"], help="override mode")

    o = sub.add_parser("operator", help="human-in-the-loop operator actions")
    o.add_argument("op", choices=["list", "complete", "reject"])
    o.add_argument("request_id", nargs="?")
    o.add_argument("--operator")
    o.add_argument("--notes")
    o.add_argument("--all", action="store_true")
    o.add_argument("--json", action="store_true")

    d = sub.add_parser("dataset", help="generate synthetic datasets")
    d.add_argument("action", choices=["generate"])
    d.add_argument("--per-label", type=int, default=20)
    d.add_argument("--images", type=int, default=0)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--mock-site", action="store_true")

    m = sub.add_parser("model", help="evaluate / train / approve / deploy classifier models")
    m.add_argument("op", choices=["list", "evaluate", "train", "approve", "reject", "deploy"])
    m.add_argument("version", nargs="?")
    m.add_argument("--approver")
    m.add_argument("--by")
    m.add_argument("--allow-regression", action="store_true")
    m.add_argument("--details", action="store_true", help="include misclassified samples")

    f = sub.add_parser("feedback", help="record the confirmed CAPTCHA type for an execution")
    f.add_argument("--execution-id", type=int, required=True)
    f.add_argument("--actual-type", required=True)
    f.add_argument("--reviewer")

    sub.add_parser("report", help="generate a performance report (JSON + HTML)")

    s = sub.add_parser("serve", help="start the REST API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "-v", "--verbose")):
        argv = ["run", *argv]  # "run" is the default command
    elif argv[0] in ("-v", "--verbose") and (len(argv) == 1 or argv[1] not in COMMANDS):
        argv = [argv[0], "run", *argv[1:]]
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_dir, settings.log_level, console=args.verbose)
    handlers = {"run": cmd_run, "operator": cmd_operator, "dataset": cmd_dataset, "model": cmd_model,
                "feedback": cmd_feedback, "report": cmd_report, "serve": cmd_serve}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        return EXIT_CODES[Status.ERROR]
    except Exception as exc:  # last-resort guard: always answer with JSON
        resp = error_response(None, Status.ERROR, f"{type(exc).__name__}: {exc}")
        sys.stdout.write(resp.to_json() + "\n")
        return resp.exit_code


if __name__ == "__main__":
    sys.exit(main())

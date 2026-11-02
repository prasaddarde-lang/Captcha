"""Controlled continuous-improvement controller (bounded, resumable, single-instance).

    python training/continuous_trainer.py status
    python training/continuous_trainer.py iterate                  # exactly one iteration
    python training/continuous_trainer.py run --max-iterations 5 --max-minutes 30

One iteration:
  1. verify the production model (automatic rollback to the last approved model on failure)
  2. inspect the latest results and pick the weakest category (validation F1 of the reference model)
  3. generate extra synthetic training samples for that category (train split only; test set is fixed)
  4. validate the dataset
  5. train a candidate (same architecture as the reference)
  6. evaluate it on the fixed test set and compare with production (or, before any approval,
     with the best recommended candidate) using the promotion criteria
  7. record the result; a candidate that meets the criteria is marked *recommended* and waits
     for a named human to approve it - nothing is promoted automatically

State lives in reports/training_history/state.json, so an interrupted run resumes where it
stopped. A lock file prevents two controllers from running at once. Runs stop at
max_iterations, max_run_minutes, the augmentation budget, low disk, or after `patience`
iterations without a validated improvement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import (ResourceLimitError, apply_thread_limits, check_disk, config_hash, free_disk_mb,  # noqa: E402
                             get_paths, load_config, now_iso, read_json, write_json)

apply_thread_limits()

from training.dataset_manager import TrainingDataset  # noqa: E402
from training.dataset_validator import validate  # noqa: E402
from training.generate_dataset import generate, generate_extra  # noqa: E402
from training.model_registry import ImageModelRegistry  # noqa: E402
from training.train_classifier import TrainingStopped, already_trained, run_training  # noqa: E402


class LockError(RuntimeError):
    pass


class ContinuousTrainer:
    def __init__(self, cfg: dict | None = None, progress: bool = True):
        self.cfg = cfg or load_config()
        self.paths = get_paths(self.cfg)
        self.registry = ImageModelRegistry(self.cfg, self.paths)
        self.state_file = self.paths.history / "state.json"
        self.lock_file = self.paths.history / "trainer.lock"
        self.history_file = self.paths.history / "history.jsonl"
        self.progress = progress

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        return read_json(self.state_file, {"created_at": now_iso(), "iterations": [], "extra_per_category": {},
                                           "total_extra": 0, "no_improve_streak": 0, "in_progress": None,
                                           "last_full_validation": None, "stopped_reason": None})

    def _save(self, st: dict) -> None:
        st["updated_at"] = now_iso()
        write_json(self.state_file, st)

    def _log(self, msg: str) -> None:
        if self.progress:
            print(msg, flush=True)

    def _record(self, st: dict, entry: dict) -> None:
        st["iterations"].append(entry)
        with open(self.history_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    # ------------------------------------------------------------------- lock
    def acquire(self) -> None:
        if self.lock_file.exists():
            try:
                pid = int(json.loads(self.lock_file.read_text())["pid"])
                import psutil
                alive = psutil.pid_exists(pid) and pid != os.getpid()
            except Exception:
                alive = False
            if alive:
                raise LockError(f"another training controller is running (pid {pid})")
        write_json(self.lock_file, {"pid": os.getpid(), "started_at": now_iso()})

    def release(self) -> None:
        self.lock_file.unlink(missing_ok=True)

    # ------------------------------------------------------------- iteration
    def weakest_category(self, ref_version: str) -> tuple[str, float]:
        per = self.registry.get(ref_version)["validation_metrics"]["per_category"]
        cat = min(per, key=lambda c: (per[c]["f1"], per[c]["recall"]))
        return cat, per[cat]["f1"]

    def iterate(self) -> dict:
        cfg, st = self.cfg, self.state()
        t0 = time.time()
        check_disk(self.paths.datasets, cfg)

        if self.registry.production_version():
            ok, reasons = self.registry.verify_production()
            if not ok:
                self._log(f"production check failed ({reasons}) -> rolling back")
                self.registry.rollback()

        if not self.paths.manifest.exists():
            self._log("no dataset yet -> generating the base dataset")
            generate(int(cfg["dataset"]["total_samples"]), cfg, progress=self.progress)

        # resume an interrupted iteration
        ip = st.get("in_progress")
        if ip:
            self._log(f"resuming interrupted iteration {ip['iteration']} ({ip['stage']})")
            return self._train_and_record(st, ip, t0)

        n = len(st["iterations"])
        ref = self.registry.reference_version()
        if ref is None:  # first ever run: compare all architectures
            ip = {"iteration": n, "stage": "training", "version": self.registry.next_version(),
                  "archs": list(cfg["training"]["architectures"]), "weakest": None, "weakest_f1": None,
                  "added": 0, "reference": None, "note": "initial architecture comparison"}
            st["in_progress"] = ip
            self._save(st)
            return self._train_and_record(st, ip, t0)

        weak, weak_f1 = self.weakest_category(ref)
        budget = int(cfg["augmentation"]["max_total_extra"]) - int(st["total_extra"])
        if budget <= 0:
            st["stopped_reason"] = "augmentation budget exhausted"
            self._save(st)
            return {"iteration": n, "skipped": True, "reason": st["stopped_reason"]}
        count = min(int(cfg["augmentation"]["per_iteration"]), budget)
        self._log(f"iteration {n}: reference {ref}; weakest category '{weak}' (val F1 {weak_f1}); "
                  f"adding {count} synthetic training samples")
        ip = {"iteration": n, "stage": "augmenting", "version": self.registry.next_version(),
              "archs": [self.registry.get(ref)["architecture"]], "weakest": weak, "weakest_f1": weak_f1,
              "added": count, "reference": ref, "note": f"iteration {n}: +{count} '{weak}' samples"}
        st["in_progress"] = ip
        self._save(st)
        generate_extra(weak, count, n, cfg)
        st["extra_per_category"][weak] = st["extra_per_category"].get(weak, 0) + count
        st["total_extra"] += count
        ip["stage"] = "training"
        self._save(st)
        return self._train_and_record(st, ip, t0)

    def _train_and_record(self, st: dict, ip: dict, t0: float) -> dict:
        full = (len(st["iterations"]) % 5 == 0)
        v = validate(self.cfg, full=full)
        if full:
            st["last_full_validation"] = {"at": v.get("validated_at"), "ok": v["ok"], "total": v.get("total_samples")}
        if not v["ok"]:
            st["in_progress"] = None
            st["stopped_reason"] = f"dataset validation failed: {v['errors'][:3]}"
            self._save(st)
            raise RuntimeError(st["stopped_reason"])

        # do not repeat completed training: identical data + config + architecture already registered
        ds_hash = TrainingDataset(self.cfg, self.paths).dataset_hash()
        chash = config_hash(self.cfg)
        done = already_trained(self.registry, ds_hash, chash, ip["archs"])
        if done:
            st["in_progress"] = None
            self._save(st)
            return {"iteration": ip["iteration"], "skipped": True, "reason": f"already trained as {done}"}

        try:
            s = run_training(self.cfg, ip["archs"], ip["version"], resume=True, progress=self.progress,
                             note=ip["note"], iteration=ip["iteration"])
        except TrainingStopped as exc:
            self._log(f"training paused: {exc}")
            self._save(st)  # in_progress kept -> next call resumes from the checkpoint
            return {"iteration": ip["iteration"], "paused": True, "reason": str(exc)}

        ref_acc = self.registry.get(ip["reference"])["test_metrics"]["accuracy"] if ip["reference"] else None
        entry = {"iteration": ip["iteration"], "at": now_iso(), "version": s["version"], "architecture": s["architecture"],
                 "weakest_category": ip["weakest"], "weakest_val_f1": ip["weakest_f1"], "samples_added": ip["added"],
                 "reference": ip["reference"], "reference_test_accuracy": ref_acc, "test_accuracy": s["test_accuracy"],
                 "test_macro_f1": s["test_macro_f1"], "fp_rate": s["fp_rate"], "fn_rate": s["fn_rate"],
                 "recommended": s["recommended"], "criteria": s["criteria"],
                 "seconds": round(time.time() - t0, 1), "free_disk_mb": round(free_disk_mb(self.paths.datasets))}
        if s["recommended"]:
            st["no_improve_streak"] = 0
        else:
            st["no_improve_streak"] += 1
        self._record(st, entry)
        st["in_progress"] = None
        self._save(st)
        verdict = "RECOMMENDED - awaiting human approval" if s["recommended"] else "not better - kept as record only"
        self._log(f"iteration {ip['iteration']} -> {s['version']}: test acc {s['test_accuracy']} "
                  f"(reference {ref_acc}) -> {verdict}")
        return entry

    def run(self, max_iterations: int | None = None, max_minutes: float | None = None) -> list[dict]:
        c = self.cfg["continuous"]
        max_iterations = int(max_iterations or c["max_iterations"])
        deadline = time.time() + 60 * float(max_minutes or c["max_run_minutes"])
        results = []
        self.acquire()
        try:
            st = self.state()
            st["stopped_reason"] = None
            self._save(st)
            for i in range(max_iterations):
                if time.time() > deadline:
                    reason = "max_run_minutes reached"
                    break
                if self.state()["no_improve_streak"] >= int(c["patience"]):
                    reason = f"no validated improvement in {c['patience']} iterations (patience)"
                    break
                try:
                    r = self.iterate()
                except ResourceLimitError as exc:
                    reason = f"resource limit: {exc}"
                    break
                results.append(r)
                if r.get("skipped") or r.get("paused"):
                    reason = r["reason"]
                    break
                if i + 1 < max_iterations and float(c.get("interval_minutes", 0)) > 0:
                    time.sleep(60 * float(c["interval_minutes"]))
            else:
                reason = f"max_iterations ({max_iterations}) reached"
            st = self.state()
            st["stopped_reason"] = reason
            self._save(st)
            self._log(f"controller stopped: {reason}")
            return results
        finally:
            self.release()

    def status(self) -> dict:
        st = self.state()
        pending = [s for s in self.registry.summary() if s["status"] in ("CANDIDATE", "APPROVED")]
        return {"production": self.registry.production_version(), "reference": self.registry.reference_version(),
                "iterations": len(st["iterations"]), "no_improve_streak": st["no_improve_streak"],
                "total_extra_samples": st["total_extra"], "extra_per_category": st["extra_per_category"],
                "in_progress": st["in_progress"], "stopped_reason": st.get("stopped_reason"),
                "recommended_awaiting_approval": [p["version"] for p in pending if p["recommended"]],
                "last_iteration": st["iterations"][-1] if st["iterations"] else None,
                "locked": self.lock_file.exists(), "free_disk_mb": round(free_disk_mb(self.paths.datasets))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Continuous training controller")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("iterate")
    r = sub.add_parser("run")
    r.add_argument("--max-iterations", type=int)
    r.add_argument("--max-minutes", type=float)
    sub.add_parser("unlock", help="remove a stale lock file")
    args = ap.parse_args(argv)
    ct = ContinuousTrainer()
    try:
        if args.cmd == "status":
            print(json.dumps(ct.status(), indent=2, default=str))
        elif args.cmd == "iterate":
            ct.acquire()
            try:
                print(json.dumps(ct.iterate(), indent=2, default=str))
            finally:
                ct.release()
        elif args.cmd == "run":
            ct.run(args.max_iterations, args.max_minutes)
            print(json.dumps(ct.status(), indent=2, default=str))
        elif args.cmd == "unlock":
            ct.release()
            print("lock removed")
    except LockError as exc:
        print(f"LOCKED: {exc}", file=sys.stderr)
        return 4
    except ResourceLimitError as exc:
        print(f"RESOURCE LIMIT: {exc}", file=sys.stderr)
        return 5
    from training.training_report import write_training_report
    write_training_report(ct.cfg, ct.paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())

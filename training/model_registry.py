"""Image-model registry: versioning, human approval, promotion, archive and rollback.

    python training/model_registry.py list
    python training/model_registry.py approve img-1.0.0 --approver "Jane Doe"   # approve + promote
    python training/model_registry.py reject  img-1.0.3 --reviewer "Jane Doe"
    python training/model_registry.py rollback [img-1.0.0]
    python training/model_registry.py verify

Statuses: CANDIDATE -> APPROVED -> PRODUCTION -> ARCHIVED   (or REJECTED / ROLLED_BACK)
Only a named human can approve. Promotion copies the model into models/production and
keeps every once-approved model in models/archive for rollback. After promotion the
production model is verified on the fixed test set; if that check fails the previous
approved model is restored automatically.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import Paths, get_paths, load_config, now_iso, read_json, sha256_bytes, write_json  # noqa: E402

NON_HUMAN = {"", "auto", "automatic", "system", "learning", "trainer", "continuous_trainer", "claude", "bot", "ai"}


class ApprovalError(RuntimeError):
    pass


class ImageModelRegistry:
    def __init__(self, cfg: dict | None = None, paths: Paths | None = None):
        self.cfg = cfg or load_config()
        self.paths = paths or get_paths(self.cfg)
        self.file = self.paths.models / "image_registry.json"

    # ----------------------------------------------------------------- storage
    def _load(self) -> dict:
        return read_json(self.file, {"production": None, "history": [], "models": {}})

    def _save(self, data: dict) -> None:
        write_json(self.file, data)

    def models(self) -> dict[str, dict]:
        return self._load()["models"]

    def get(self, version: str) -> dict:
        m = self.models().get(version)
        if not m:
            raise ApprovalError(f"unknown model version {version}")
        return m

    def production_version(self) -> str | None:
        return self._load()["production"]

    def model_path(self, version: str) -> Path:
        return self.paths.root / self.get(version)["path"] if not Path(self.get(version)["path"]).is_absolute() \
            else Path(self.get(version)["path"])

    def next_version(self) -> str:
        n = len(self.models())
        while f"img-1.0.{n}" in self.models():
            n += 1
        return f"img-1.0.{n}"

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.paths.root.resolve()).as_posix()
        except ValueError:
            return str(p.resolve())

    # ------------------------------------------------------------ registration
    def register_candidate(self, version: str, model_file: Path, record: dict) -> dict:
        data = self._load()
        if version in data["models"]:
            raise ApprovalError(f"version {version} already registered")
        entry = {"version": version, "status": "CANDIDATE", "path": self._rel(model_file),
                 "sha256": sha256_bytes(model_file.read_bytes()), "created_at": now_iso(),
                 "approved_by": None, "approved_at": None, "promoted_at": None, **record}
        data["models"][version] = entry
        self._save(data)
        return entry

    def reference_version(self) -> str | None:
        """What a new candidate is compared with: production, else the best *recommended* candidate
        (one that met the criteria against its own reference)."""
        data = self._load()
        if data["production"]:
            return data["production"]
        best, best_acc = None, -1.0
        for v, m in data["models"].items():
            if m["status"] in ("CANDIDATE", "APPROVED") and m.get("recommended") and m.get("test_metrics"):
                acc = m["test_metrics"]["accuracy"]
                if acc > best_acc:
                    best, best_acc = v, acc
        return best

    # ---------------------------------------------------------------- criteria
    def check_criteria(self, cand: dict, ref: dict | None) -> tuple[bool, list[str]]:
        """Promotion criteria on the fixed test set. Returns (passed, reasons)."""
        p = self.cfg["promotion"]
        reasons: list[str] = []
        if ref is None:
            if cand["accuracy"] < p["initial_min_accuracy"]:
                reasons.append(f"accuracy {cand['accuracy']} < initial minimum {p['initial_min_accuracy']}")
            fpr = cand.get("false_positive_rate", 1.0)
            if fpr > p["initial_max_false_positive_rate"]:
                reasons.append(f"false-positive rate {fpr} > {p['initial_max_false_positive_rate']}")
            return not reasons, reasons or ["meets initial thresholds"]
        if cand.get("test_set_hash") and ref.get("test_set_hash") and cand["test_set_hash"] != ref["test_set_hash"]:
            return False, ["test sets differ - models are not comparable"]
        gain = round(cand["accuracy"] - ref["accuracy"], 4)
        if gain < p["min_accuracy_gain"]:
            reasons.append(f"accuracy gain {gain:+.4f} < required {p['min_accuracy_gain']}")
        for c, v in cand["per_category"].items():
            drop = ref["per_category"].get(c, {}).get("f1", 0) - v["f1"]
            if drop > p["max_category_f1_drop"]:
                reasons.append(f"{c} F1 drops by {drop:.4f} (> {p['max_category_f1_drop']})")
        for key, lim in (("false_positive_rate", "max_false_positive_rate_increase"),
                         ("false_negative_rate", "max_false_negative_rate_increase")):
            inc = cand.get(key, 0) - ref.get(key, 0)
            if inc > p[lim]:
                reasons.append(f"{key} rises by {inc:.4f} (> {p[lim]})")
        return not reasons, reasons or [f"accuracy gain {gain:+.4f}; no category or FP/FN regression"]

    # ---------------------------------------------------------------- approval
    def approve(self, version: str, approver: str, allow_regression: bool = False) -> dict:
        name = (approver or "").strip()
        if name.lower() in NON_HUMAN or len(name) < 2:
            raise ApprovalError("approval requires a named human reviewer")
        data = self._load()
        m = data["models"].get(version)
        if not m:
            raise ApprovalError(f"unknown model version {version}")
        if m["status"] != "CANDIDATE":
            raise ApprovalError(f"{version} is {m['status']}; only CANDIDATE models can be approved")
        prod = data["models"].get(data["production"]) if data["production"] else None
        passed, reasons = self.check_criteria(m["test_metrics"], prod["test_metrics"] if prod else None)
        if not passed and not allow_regression:
            raise ApprovalError("promotion criteria not met: " + "; ".join(reasons))
        m.update(status="APPROVED", approved_by=name, approved_at=now_iso(),
                 approval_note="criteria met" if passed else "approved despite: " + "; ".join(reasons))
        self._save(data)
        return m

    def reject(self, version: str, reviewer: str) -> dict:
        data = self._load()
        m = data["models"].get(version)
        if not m or m["status"] not in ("CANDIDATE", "APPROVED"):
            raise ApprovalError(f"{version} cannot be rejected")
        m.update(status="REJECTED", rejected_by=reviewer, rejected_at=now_iso())
        self._save(data)
        return m

    # ---------------------------------------------------------- promote / rollback
    def _install(self, src: Path, version: str, meta: dict) -> None:
        prod = self.paths.production / "image_classifier.npz"
        tmp = prod.with_name("image_classifier.incoming.npz")
        shutil.copy2(src, tmp)
        tmp.replace(prod)  # atomic swap; the runtime reloads on mtime change
        write_json(self.paths.production / "image_classifier.json", {"version": version, "installed_at": now_iso(), **meta})

    def promote(self, version: str, verify: bool = True) -> dict:
        data = self._load()
        m = data["models"].get(version)
        if not m:
            raise ApprovalError(f"unknown model version {version}")
        if m["status"] != "APPROVED" or not m.get("approved_by"):
            raise ApprovalError(f"{version} must be human-approved before promotion (status {m['status']})")
        src = self.model_path(version)
        if sha256_bytes(src.read_bytes()) != m["sha256"]:
            raise ApprovalError(f"{version} model file does not match its registered checksum")
        archived = self.paths.archive / f"{version}.npz"
        shutil.copy2(src, archived)  # every once-approved model is kept for rollback
        previous = data["production"]
        self._install(archived, version, {"approved_by": m["approved_by"], "sha256": m["sha256"]})
        if previous and previous in data["models"]:
            data["models"][previous]["status"] = "ARCHIVED"
        m.update(status="PRODUCTION", promoted_at=now_iso(), archive_path=self._rel(archived))
        data["production"] = version
        data["history"].append({"event": "promote", "version": version, "previous": previous, "at": now_iso()})
        self._save(data)
        if verify:
            ok, reasons = self.verify_production()
            if not ok:
                self._auto_rollback(version, previous, reasons)
                raise ApprovalError(f"post-promotion check failed ({'; '.join(reasons)}); rolled back to {previous}")
        return self.get(version)

    def _auto_rollback(self, bad: str, previous: str | None, reasons: list[str]) -> None:
        data = self._load()
        data["models"][bad]["status"] = "ROLLED_BACK"
        data["models"][bad]["rollback_reason"] = reasons
        if previous:
            self._install(self.paths.archive / f"{previous}.npz", previous, {"restored_after": bad})
            data["models"][previous]["status"] = "PRODUCTION"
        else:
            (self.paths.production / "image_classifier.npz").unlink(missing_ok=True)
            (self.paths.production / "image_classifier.json").unlink(missing_ok=True)
        data["production"] = previous
        data["history"].append({"event": "auto_rollback", "from": bad, "to": previous, "reasons": reasons, "at": now_iso()})
        self._save(data)

    def rollback(self, version: str | None = None) -> dict:
        data = self._load()
        if version is None:
            promos = [h for h in data["history"] if h["event"] == "promote" and h["version"] == data["production"]]
            version = promos[-1]["previous"] if promos else None
            if not version:
                raise ApprovalError("no previous production model to roll back to")
        m = data["models"].get(version)
        archived = self.paths.archive / f"{version}.npz"
        if not m or not m.get("approved_by") or not archived.exists():
            raise ApprovalError(f"{version} was never an approved production model")
        current = data["production"]
        self._install(archived, version, {"rolled_back_from": current})
        if current:
            data["models"][current]["status"] = "ARCHIVED"
        m["status"] = "PRODUCTION"
        data["production"] = version
        data["history"].append({"event": "rollback", "from": current, "to": version, "at": now_iso()})
        self._save(data)
        return m

    def verify_production(self, tolerance: float = 0.02) -> tuple[bool, list[str]]:
        """Load the installed production model and re-check it on the fixed test set."""
        from core.image_classifier import NumpyImageModel
        from training.dataset_manager import TrainingDataset
        from training.evaluate_model import evaluate_model

        data = self._load()
        v = data["production"]
        if not v:
            return True, ["no production model"]
        f = self.paths.production / "image_classifier.npz"
        reasons = []
        if not f.exists():
            return False, ["production model file missing"]
        if sha256_bytes(f.read_bytes()) != data["models"][v]["sha256"]:
            reasons.append("production file checksum mismatch")
        try:
            model = NumpyImageModel.load(f)
            m = evaluate_model(model, TrainingDataset(self.cfg, self.paths), "test", latency_samples=20)
            recorded = data["models"][v]["test_metrics"]["accuracy"]
            if m["accuracy"] < recorded - tolerance:
                reasons.append(f"accuracy {m['accuracy']} below recorded {recorded} - tolerance")
        except Exception as exc:
            reasons.append(f"production model failed to load/evaluate: {type(exc).__name__}: {exc}")
        return not reasons, reasons or ["ok"]

    def summary(self) -> list[dict]:
        out = []
        for v, m in self.models().items():
            t = m.get("test_metrics", {})
            out.append({"version": v, "status": m["status"], "architecture": m.get("architecture"),
                        "test_accuracy": t.get("accuracy"), "macro_f1": t.get("macro_f1"),
                        "fp_rate": t.get("false_positive_rate"), "fn_rate": t.get("false_negative_rate"),
                        "recommended": m.get("recommended"), "approved_by": m.get("approved_by"),
                        "created_at": m.get("created_at")})
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Image-model registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("approve", help="approve (named human) and promote to production")
    a.add_argument("version")
    a.add_argument("--approver", required=True)
    a.add_argument("--allow-regression", action="store_true")
    a.add_argument("--no-promote", action="store_true")
    r = sub.add_parser("reject")
    r.add_argument("version")
    r.add_argument("--reviewer", required=True)
    rb = sub.add_parser("rollback")
    rb.add_argument("version", nargs="?")
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    reg = ImageModelRegistry()
    import json
    try:
        if args.cmd == "list":
            print(json.dumps({"production": reg.production_version(), "models": reg.summary()}, indent=2))
        elif args.cmd == "approve":
            reg.approve(args.version, args.approver, args.allow_regression)
            m = reg.get(args.version) if args.no_promote else reg.promote(args.version)
            print(json.dumps({"version": args.version, "status": m["status"], "approved_by": m["approved_by"]}, indent=2))
        elif args.cmd == "reject":
            print(json.dumps(reg.reject(args.version, args.reviewer)["status"]))
        elif args.cmd == "rollback":
            m = reg.rollback(args.version)
            print(json.dumps({"production": m["version"], "status": m["status"]}))
        elif args.cmd == "verify":
            ok, reasons = reg.verify_production()
            print(json.dumps({"ok": ok, "reasons": reasons}))
            return 0 if ok else 1
    except ApprovalError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 41
    return 0


if __name__ == "__main__":
    sys.exit(main())

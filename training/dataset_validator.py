"""Validate the dataset and its manifest.

    python training/dataset_validator.py          # full: decode + hash every file
    python training/dataset_validator.py --quick  # manifest-level checks only
Exit code 0 = valid, 1 = problems found.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import apply_thread_limits, get_paths, load_config, now_iso, read_json, sha256_bytes, write_json  # noqa: E402

apply_thread_limits()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REQUIRED = ("sample_id", "category", "split", "file", "width", "height", "source", "license",
            "authorization_ref", "label", "created_at", "sha256", "dhash")
SPLITS = ("train", "validation", "test")


def validate(cfg: dict | None = None, full: bool = True, write_report: bool = True) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    t0 = time.time()
    errors: list[str] = []
    warnings: list[str] = []
    manifest = read_json(paths.manifest)
    if manifest is None:
        report = {"ok": False, "errors": ["manifest.json not found"], "error_count": 1}
        if write_report:
            write_json(paths.history / "dataset_validation_latest.json", report)
        return report

    cats = set(cfg["dataset"]["categories"])
    samples = manifest.get("samples", [])
    ids, shas, dhs = Counter(), Counter(), Counter()
    counts: dict[str, Counter] = defaultdict(Counter)
    base_counts: dict[str, Counter] = defaultdict(Counter)
    checked = 0
    for s in samples:
        sid = s.get("sample_id", "?")
        missing = [k for k in REQUIRED if s.get(k) in (None, "")]
        if missing:
            errors.append(f"{sid}: missing fields {missing}")
            continue
        ids[sid] += 1
        shas[s["sha256"]] += 1
        dhs[s["dhash"]] += 1
        if s["category"] not in cats:
            errors.append(f"{sid}: unknown category {s['category']}")
        if s["split"] not in SPLITS:
            errors.append(f"{sid}: unknown split {s['split']}")
        if s["label"] != s["category"]:
            errors.append(f"{sid}: label does not match category")
        if s.get("content_label") is not None:
            errors.append(f"{sid}: content_label must be empty (challenge text/answers are never recorded)")
        if "augmentation" in s["source"] and s["split"] != "train":
            errors.append(f"{sid}: augmentation samples may only be in the train split")
        counts[s["split"]][s["category"]] += 1
        if "augmentation" not in s["source"]:
            base_counts[s["split"]][s["category"]] += 1
        f = paths.datasets / s["file"]
        if not f.exists():
            errors.append(f"{sid}: file missing ({s['file']})")
            continue
        if full:
            data = f.read_bytes()
            if sha256_bytes(data) != s["sha256"]:
                errors.append(f"{sid}: sha256 mismatch (file changed or corrupt)")
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"{sid}: not decodable")
                continue
            if (img.shape[1], img.shape[0]) != (s["width"], s["height"]):
                errors.append(f"{sid}: dimensions {img.shape[1]}x{img.shape[0]} != manifest {s['width']}x{s['height']}")
            if float(img.std()) < 3.0:
                errors.append(f"{sid}: image is blank")
            checked += 1

    errors += [f"duplicate sample_id {k}" for k, v in ids.items() if v > 1]
    errors += [f"duplicate image (sha256 {k[:12]}...) x{v}" for k, v in shas.items() if v > 1]
    errors += [f"perceptual duplicate (dhash {k}) x{v}" for k, v in dhs.items() if v > 1]

    # base split must match the configured proportions and be balanced across categories
    base_total = manifest.get("base_count")
    if base_total:
        per_cat = base_total // len(cats)
        n_val = int(round(per_cat * cfg["dataset"]["split"]["validation"]))
        n_test = int(round(per_cat * cfg["dataset"]["split"]["test"]))
        expect = {"train": per_cat - n_val - n_test, "validation": n_val, "test": n_test}
        if base_total % len(cats) == 0:
            for split, n in expect.items():
                for c in cats:
                    if base_counts[split][c] != n:
                        errors.append(f"base {split}/{c}: {base_counts[split][c]} samples, expected {n}")
        if sum(sum(c.values()) for c in base_counts.values()) != base_total:
            errors.append(f"base sample total != base_count {base_total}")
    files_on_disk = sum(1 for sp in SPLITS for _ in (paths.datasets / sp).glob("*.jpg"))
    if files_on_disk != len(samples):
        warnings.append(f"{files_on_disk} image files on disk vs {len(samples)} in manifest")

    report = {
        "ok": not errors, "validated_at": now_iso(), "mode": "full" if full else "quick",
        "total_samples": len(samples), "files_checked": checked, "seconds": round(time.time() - t0, 1),
        "counts": {sp: dict(sorted(counts[sp].items())) for sp in SPLITS},
        "split_totals": {sp: sum(counts[sp].values()) for sp in SPLITS},
        "base_split_totals": {sp: sum(base_counts[sp].values()) for sp in SPLITS},
        "error_count": len(errors), "errors": errors[:100], "warnings": warnings,
    }
    if write_report:
        write_json(paths.history / "dataset_validation_latest.json", report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the CAPTCHA-type dataset")
    ap.add_argument("--quick", action="store_true", help="skip decoding/hashing every file")
    args = ap.parse_args(argv)
    r = validate(full=not args.quick)
    print(f"{'VALID' if r['ok'] else 'INVALID'}: {r.get('total_samples', 0)} samples, "
          f"{r.get('files_checked', 0)} files checked, {r['error_count']} errors, {r.get('seconds', 0)}s")
    print(f"split totals: {r.get('split_totals')}")
    for e in r["errors"][:20]:
        print("  ERROR", e)
    for w in r.get("warnings", []):
        print("  WARN ", w)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

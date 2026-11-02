"""Training / model performance report (Markdown + JSON).

    python training/training_report.py      -> reports/performance/training_report_latest.md (+ timestamped copies)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import Paths, get_paths, load_config, now_iso, read_json, write_json  # noqa: E402


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.2f}%"


def build(cfg: dict, paths: Paths) -> tuple[dict, str]:
    from training.model_registry import ImageModelRegistry

    reg = ImageModelRegistry(cfg, paths)
    models = reg.models()
    state = read_json(paths.history / "state.json", {}) or {}
    val = read_json(paths.history / "dataset_validation_latest.json", {}) or {}
    manifest = read_json(paths.manifest, {}) or {}
    prod = reg.production_version()
    ref = reg.reference_version()
    focus = prod or ref
    first = next(iter(models.values()), None)

    data = {"generated_at": now_iso(), "production": prod, "reference": ref, "models": reg.summary(),
            "dataset_validation": val, "state": {k: v for k, v in state.items() if k != "iterations"},
            "iterations": state.get("iterations", []),
            "focus_model": {"version": focus, **(models[focus] if focus else {})}}

    L = [f"# Image Model Performance Report", "",
         f"Generated {data['generated_at']}. Production model: **{prod or 'none (awaiting human approval)'}**. "
         f"Reference model: **{ref or 'none'}**.", ""]
    L += ["## Dataset", "",
          f"* Samples in manifest: **{len(manifest.get('samples', []))}** (base {manifest.get('base_count')}, "
          f"augmentation {state.get('total_extra', 0)})",
          f"* Split totals: {val.get('split_totals')} (base: {val.get('base_split_totals')})",
          f"* Last validation: **{'VALID' if val.get('ok') else 'INVALID'}** ({val.get('mode')}, "
          f"{val.get('files_checked')} files decoded, {val.get('error_count')} errors) at {val.get('validated_at')}",
          f"* Source / licence: {manifest.get('license')} — authorisation `{manifest.get('authorization_ref')}`",
          "* Labels are challenge categories only; rendered text and arithmetic answers are never recorded.", ""]

    if first and first.get("architecture_comparison"):
        L += ["## Architecture comparison (validation set, first training run)", "",
              _row(["architecture", "accuracy", "macro-F1", "FP rate", "FN rate", "params", "epochs", "train s", "latency ms"]),
              _row(["---"] * 9)]
        for a, r in first["architecture_comparison"].items():
            L.append(_row([a, _pct(r["val_accuracy"]), f"{r['val_macro_f1']:.4f}", _pct(r["val_fp_rate"]), _pct(r["val_fn_rate"]),
                           f"{r['param_count']:,}", r["epochs_run"], r["train_seconds"], r["latency_ms_mean"]]))
        L += ["", f"Selected: **{first['architecture']}** (highest validation macro-F1). Evaluated but not built: "
              "Tesseract, EasyOCR, PaddleOCR and CRNN text recognisers — they read challenge text, i.e. solve it.", ""]

    if focus:
        m = models[focus]
        t = m["test_metrics"]
        rt = t["at_runtime_threshold"]
        L += [f"## {focus} on the fixed test set ({t['samples']} samples)", "",
              f"* Accuracy **{_pct(t['accuracy'])}**, macro-F1 **{t['macro_f1']:.4f}**, architecture `{m['architecture']}`",
              f"* False-positive detection rate {_pct(t['false_positive_rate'])}, false-negative {_pct(t['false_negative_rate'])} (arg-max)",
              f"* At the runtime confidence threshold {rt['min_confidence']}: FP {_pct(rt['false_positive_rate'])}, "
              f"FN {_pct(rt['false_negative_rate'])}, {rt['abstained']} low-confidence predictions abstained "
              "(those fall back to page-metadata detection / a human)",
              f"* Latency per image (decode + features + predict): mean {t['latency_ms']['mean']} ms, "
              f"p95 {t['latency_ms']['p95']} ms",
              f"* Model size {t['resources']['model_size_bytes'] / 1024:.0f} KB ({t['resources']['param_count']:,} parameters); "
              f"evaluation CPU {t['resources'].get('cpu_seconds')} s, process RSS {t['resources'].get('rss_mb')} MB",
              f"* Character accuracy / exact-string accuracy: **not applicable** — {t['ocr_metrics']['reason']}", "",
              _row(["category", "precision", "recall (detection rate)", "F1", "n"]), _row(["---"] * 5)]
        for c, v in t["per_category"].items():
            L.append(_row([c, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"]]))
        L.append("")

    L += ["## Model registry", "", _row(["version", "status", "arch", "test acc", "macro-F1", "FP", "FN", "recommended", "approved by"]),
          _row(["---"] * 9)]
    for s in reg.summary():
        L.append(_row([s["version"], s["status"], s["architecture"], _pct(s["test_accuracy"]), s["macro_f1"],
                       _pct(s["fp_rate"]), _pct(s["fn_rate"]), "yes" if s["recommended"] else "no", s["approved_by"] or "—"]))
    L.append("")
    its = state.get("iterations", [])
    if its:
        L += ["## Continuous-improvement history", "",
              _row(["#", "version", "weakest category (val F1)", "+samples", "test acc", "vs reference", "result"]), _row(["---"] * 7)]
        for e in its:
            ref_acc = e.get("reference_test_accuracy")
            delta = "—" if ref_acc is None else f"{100 * (e['test_accuracy'] - ref_acc):+.2f} pts vs {e['reference']}"
            weak = "—" if not e.get("weakest_category") else f"{e['weakest_category']} ({e['weakest_val_f1']})"
            L.append(_row([e["iteration"], e["version"], weak, e["samples_added"], _pct(e["test_accuracy"]), delta,
                           "recommended" if e["recommended"] else "not promoted"]))
        L.append("")
    L += [f"Controller: {len(its)} iterations, streak without improvement {state.get('no_improve_streak', 0)}, "
          f"last stop reason: {state.get('stopped_reason')}.", "",
          "## Caveats", "",
          "* All numbers are on **synthetic** images. Real screenshots differ (fonts, branding, anti-aliasing); expect lower "
          "accuracy until a pilot measures it. Low-confidence predictions are never used on their own.",
          "* The fixed test set is reused for every candidate comparison, so repeated selection slightly overstates "
          "the improvement; weak categories are chosen on the validation set to limit this.", ""]
    if not prod:
        pend = [s["version"] for s in reg.summary() if s["status"] == "CANDIDATE" and s["recommended"]]
        if pend:
            L += ["## Action needed", "",
                  f"No model is in production yet. Recommended candidate(s): {', '.join(pend)}. A named reviewer approves with:",
                  "", f"    scripts\\approve_model.bat {ref} \"Your Name\"", ""]
    return data, "\n".join(L)


def write_training_report(cfg: dict | None = None, paths: Paths | None = None) -> dict[str, str]:
    cfg = cfg or load_config()
    paths = paths or get_paths(cfg)
    data, md = build(cfg, paths)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = {"markdown": paths.performance / "training_report_latest.md", "json": paths.performance / "training_report_latest.json"}
    out["markdown"].write_text(md, encoding="utf-8")
    write_json(out["json"], data)
    (paths.performance / f"training_report_{stamp}.md").write_text(md, encoding="utf-8")
    return {k: str(v) for k, v in out.items()}


if __name__ == "__main__":
    print(write_training_report())

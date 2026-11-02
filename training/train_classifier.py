"""Train the image-based CAPTCHA *type* classifier (CPU, numpy).

    python training/train_classifier.py                       # compare all configured architectures
    python training/train_classifier.py --arch mlp --epochs 30 --batch-size 256
    python training/train_classifier.py --resume img-1.0.4    # continue an interrupted run

Each architecture is trained on the train split, early-stopped on validation macro-F1, and
the best one (by validation macro-F1) is evaluated once on the fixed test split and registered
as a CANDIDATE. Nothing reaches production without a named human approval.

Checkpoints are written every epoch (weights, Adam state, RNG state, history), so a run
stopped by the time/memory limit or by an interruption resumes exactly where it stopped.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import (apply_thread_limits, check_disk, config_hash, detect_device, get_paths,  # noqa: E402
                             load_config, now_iso, rss_mb, write_json)

apply_thread_limits()

import numpy as np  # noqa: E402

from core.image_classifier import NumpyImageModel, softmax  # noqa: E402
from training.dataset_manager import TrainingDataset  # noqa: E402
from training.evaluate_model import classification_metrics, evaluate_model  # noqa: E402


class TrainingStopped(RuntimeError):
    """Raised when a resource/time limit stops training; the checkpoint allows resuming."""


# ---------------------------------------------------------------------------- helpers
def _macro_f1(y, pred, k):
    return classification_metrics(y, pred, [str(i) for i in range(k)])["macro_f1"]


def _save_ckpt(path: Path, arrays: dict, info: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, info_json=np.array(json.dumps(info, default=str)), **arrays)
    tmp.replace(path)


def _load_ckpt(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files if k != "info_json"}, json.loads(str(z["info_json"]))


def _init_params(arch, d, k, h, rng):
    if arch == "softmax":
        return {"W": (rng.standard_normal((d, k)) * 0.01).astype(np.float32), "b": np.zeros(k, np.float32)}
    return {"W1": (rng.standard_normal((d, h)) * np.sqrt(2.0 / d)).astype(np.float32), "b1": np.zeros(h, np.float32),
            "W2": (rng.standard_normal((h, k)) * np.sqrt(2.0 / h)).astype(np.float32), "b2": np.zeros(k, np.float32)}


def _forward_backward(arch, p, X, y, k, wd, dropout, rng, train=True):
    n = len(X)
    onehot = np.zeros((n, k), np.float32)
    onehot[np.arange(n), y] = 1
    if arch == "softmax":
        prob = softmax(X @ p["W"] + p["b"])
        loss = -np.log(prob[np.arange(n), y] + 1e-9).mean() + 0.5 * wd * float((p["W"] ** 2).sum())
        dz = (prob - onehot) / n
        return loss, {"W": X.T @ dz + wd * p["W"], "b": dz.sum(0)}
    z1 = X @ p["W1"] + p["b1"]
    hdn = np.maximum(z1, 0)
    mask = None
    if train and dropout > 0:
        mask = (rng.random(hdn.shape) >= dropout).astype(np.float32) / (1 - dropout)
        hdn = hdn * mask
    prob = softmax(hdn @ p["W2"] + p["b2"])
    loss = -np.log(prob[np.arange(n), y] + 1e-9).mean() + 0.5 * wd * float((p["W1"] ** 2).sum() + (p["W2"] ** 2).sum())
    dz2 = (prob - onehot) / n
    dh = dz2 @ p["W2"].T
    if mask is not None:
        dh *= mask
    dz1 = dh * (z1 > 0)
    return loss, {"W1": X.T @ dz1 + wd * p["W1"], "b1": dz1.sum(0), "W2": hdn.T @ dz2 + wd * p["W2"], "b2": dz2.sum(0)}


def _logits(arch, p, X):
    if arch == "softmax":
        return X @ p["W"] + p["b"]
    return np.maximum(X @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"]


# ------------------------------------------------------------------------- trainers
def train_nearest_centroid(Xtr, ytr, Xva, yva, k):
    C = np.stack([Xtr[ytr == c].mean(axis=0) for c in range(k)]).astype(np.float32)
    best, best_score = None, -np.inf
    for temp in (0.5, 1.0, 2.0, 5.0):  # temperature only affects confidences; chosen by validation NLL
        params = {"C": C, "temperature": np.array([temp], np.float32)}
        d2 = ((Xva[:, None, :] - C[None]) ** 2).mean(axis=2)
        prob = softmax(-d2 * temp)
        score = -float(np.log(prob[np.arange(len(yva)), yva] + 1e-9).mean())
        if score > best_score:
            best, best_score = params, score
    pred = (-((Xva[:, None, :] - C[None]) ** 2).mean(axis=2)).argmax(1)
    f1 = _macro_f1(yva, pred, k)
    return best, [{"epoch": 1, "val_macro_f1": f1, "val_acc": float((pred == yva).mean())}], "complete"


def train_gradient(arch, Xtr, ytr, Xva, yva, k, cfg, seed, ckpt: Path, fingerprint: str, log_file: Path,
                   resume: bool = True, deadline: float | None = None, progress: bool = True):
    tc, rc = cfg["training"], cfg["resources"]
    epochs, bs = int(tc["epochs"]), int(tc["batch_size"])
    lr, wd = float(tc["learning_rate"]), float(tc["weight_decay"])
    dropout = float(tc.get("dropout", 0.0)) if arch == "mlp" else 0.0
    patience = int(tc["early_stopping_patience"])
    rng = np.random.default_rng(seed)
    p = _init_params(arch, Xtr.shape[1], k, int(tc["hidden_units"]), rng)
    m = {n: np.zeros_like(v) for n, v in p.items()}
    v = {n: np.zeros_like(v_) for n, v_ in p.items()}
    t, start, best, best_f1, bad, history = 0, 0, copy.deepcopy(p), -1.0, 0, []

    if resume and ckpt.exists():
        arrays, info = _load_ckpt(ckpt)
        if info.get("fingerprint") == fingerprint:
            p = {n: arrays[f"p_{n}"] for n in p}
            m = {n: arrays[f"m_{n}"] for n in p}
            v = {n: arrays[f"v_{n}"] for n in p}
            best = {n: arrays[f"best_{n}"] for n in p}
            t, start, best_f1, bad, history = info["t"], info["epoch"], info["best_f1"], info["bad"], info["history"]
            rng.bit_generator.state = info["rng_state"]
            if progress:
                print(f"    resumed {arch} from epoch {start}", flush=True)
            if info.get("finished"):
                return best, history, "complete"

    status = "complete"
    max_mem = float(rc.get("max_memory_mb", 1e9))
    n = len(Xtr)
    for epoch in range(start, epochs):
        t0 = time.time()
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss, g = _forward_backward(arch, p, Xtr[idx], ytr[idx], k, wd, dropout, rng)
            losses.append(loss)
            t += 1
            for name in p:  # Adam
                m[name] = 0.9 * m[name] + 0.1 * g[name]
                v[name] = 0.999 * v[name] + 0.001 * g[name] ** 2
                mh = m[name] / (1 - 0.9 ** t)
                vh = v[name] / (1 - 0.999 ** t)
                p[name] = (p[name] - lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        lv = _logits(arch, p, Xva)
        prob = softmax(lv)
        pred = lv.argmax(1)
        val_loss = float(-np.log(prob[np.arange(len(yva)), yva] + 1e-9).mean())
        f1 = _macro_f1(yva, pred, k)
        rec = {"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 5), "val_loss": round(val_loss, 5),
               "val_acc": round(float((pred == yva).mean()), 4), "val_macro_f1": f1,
               "seconds": round(time.time() - t0, 2), "rss_mb": round(rss_mb(), 1)}
        history.append(rec)
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"arch": arch, **rec}) + "\n")
        if progress:
            print(f"    {arch:<8} epoch {epoch + 1:>3}  loss {rec['train_loss']:.4f}  val_acc {rec['val_acc']:.4f}  "
                  f"val_F1 {f1:.4f}  ({rec['seconds']}s)", flush=True)
        if f1 > best_f1 + 1e-6:
            best, best_f1, bad = copy.deepcopy(p), f1, 0
        else:
            bad += 1
        finished = bad >= patience or epoch + 1 == epochs
        arrays = {**{f"p_{k_}": a for k_, a in p.items()}, **{f"m_{k_}": a for k_, a in m.items()},
                  **{f"v_{k_}": a for k_, a in v.items()}, **{f"best_{k_}": a for k_, a in best.items()}}
        _save_ckpt(ckpt, arrays, {"fingerprint": fingerprint, "t": t, "epoch": epoch + 1, "best_f1": best_f1,
                                  "bad": bad, "history": history, "rng_state": rng.bit_generator.state,
                                  "finished": finished, "saved_at": now_iso()})
        if bad >= patience:
            status = "early_stopped"
            break
        if deadline is not None and time.time() > deadline and epoch + 1 < epochs:
            raise TrainingStopped(f"{arch}: max_training_minutes reached at epoch {epoch + 1} (checkpoint saved)")
        if rec["rss_mb"] > max_mem:
            raise TrainingStopped(f"{arch}: memory {rec['rss_mb']} MB above max_memory_mb {max_mem} (checkpoint saved)")
    return best, history, status


# --------------------------------------------------------------------------- full run
def already_trained(registry, dataset_hash: str, cfg_hash: str, archs: list[str]) -> str | None:
    """Version already trained on identical data + config + architectures, if any."""
    for version, m in registry.models().items():
        if (m.get("dataset_hash") == dataset_hash and m.get("config_hash") == cfg_hash
                and sorted(m.get("architecture_comparison", {})) == sorted(archs)):
            return version
    return None


def run_training(cfg: dict | None = None, archs: list[str] | None = None, version: str | None = None,
                 resume: bool = True, progress: bool = True, note: str = "", iteration: int | None = None) -> dict:
    """Train, select by validation, evaluate on the fixed test set, register a CANDIDATE."""
    from training.model_registry import ImageModelRegistry

    cfg = cfg or load_config()
    paths = get_paths(cfg)
    check_disk(paths.models, cfg)
    reg = ImageModelRegistry(cfg, paths)
    version = version or reg.next_version()
    archs = archs or list(cfg["training"]["architectures"])
    run_dir = paths.checkpoints / version
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train_log.jsonl"
    deadline = time.time() + 60 * float(cfg["resources"]["max_training_minutes"])
    device = detect_device(cfg)
    t_start = time.time()

    ds = TrainingDataset(cfg, paths)
    if progress:
        print(f"[{version}] loading features (cached after first run) ...", flush=True)
    Xtr, ytr, _ = ds.features("train", progress)
    Xva, yva, _ = ds.features("validation", progress)
    ds.features("test", progress)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-6
    Xtr_s, Xva_s = (Xtr - mean) / std, (Xva - mean) / std
    k = len(ds.categories)
    dhash, chash = ds.dataset_hash(), config_hash(cfg)
    seed = int(cfg["seed"])

    results = {}
    for arch in archs:
        if progress:
            print(f"  training {arch} on {len(ytr)} samples ({Xtr.shape[1]} features)", flush=True)
        a0 = time.time()
        if arch == "nearest_centroid":
            params, hist, status = train_nearest_centroid(Xtr_s, ytr, Xva_s, yva, k)
        else:
            params, hist, status = train_gradient(
                arch, Xtr_s, ytr, Xva_s, yva, k, cfg, seed, run_dir / f"{arch}_last.npz",
                f"{arch}|{dhash}|{chash}|{seed}", log_file, resume, deadline, progress)
        model = NumpyImageModel(arch, ds.categories, params, mean, std,
                                {"version": version, "feature_spec": ds.extractor.spec})
        val = evaluate_model(model, ds, "validation", latency_samples=50)
        results[arch] = {"model": model, "status": status, "epochs_run": len(hist), "history": hist,
                         "train_seconds": round(time.time() - a0, 1), "val": val}
        if progress:
            print(f"  -> {arch}: val acc {val['accuracy']:.4f}  val macro-F1 {val['macro_f1']:.4f}  "
                  f"({results[arch]['train_seconds']}s, {status})", flush=True)

    best_arch = max(results, key=lambda a: (results[a]["val"]["macro_f1"], results[a]["val"]["accuracy"],
                                            -results[a]["model"].param_count))
    best = results[best_arch]["model"]
    best.meta.update(architecture=best_arch, trained_at=now_iso(), dataset_hash=dhash, config_hash=chash,
                     seed=seed, classes=ds.categories, train_samples=int(len(ytr)), note=note, iteration=iteration)
    model_file = run_dir / "model.npz"
    best.save(model_file)
    test = evaluate_model(best, ds, "test")
    comparison = {a: {"val_accuracy": r["val"]["accuracy"], "val_macro_f1": r["val"]["macro_f1"],
                      "val_fp_rate": r["val"].get("false_positive_rate"), "val_fn_rate": r["val"].get("false_negative_rate"),
                      "epochs_run": r["epochs_run"], "status": r["status"], "train_seconds": r["train_seconds"],
                      "param_count": r["model"].param_count, "latency_ms_mean": r["val"]["latency_ms"]["mean"]}
                  for a, r in results.items()}

    ref_v = reg.reference_version()
    ref = reg.get(ref_v)["test_metrics"] if ref_v else None
    passed, reasons = reg.check_criteria(test, ref)
    record = {"architecture": best_arch, "test_metrics": test, "validation_metrics": results[best_arch]["val"],
              "architecture_comparison": comparison, "dataset_hash": dhash, "config_hash": chash,
              "train_samples": int(len(ytr)), "device": device, "iteration": iteration, "note": note,
              "compared_with": ref_v, "recommended": passed, "criteria": reasons,
              "train_seconds_total": round(time.time() - t_start, 1)}
    reg.register_candidate(version, model_file, record)
    for f in run_dir.glob("*_last.npz"):  # resumable state no longer needed once registered
        f.unlink()
    summary = {"version": version, "architecture": best_arch, "test_accuracy": test["accuracy"],
               "test_macro_f1": test["macro_f1"], "fp_rate": test["false_positive_rate"],
               "fn_rate": test["false_negative_rate"], "compared_with": ref_v, "recommended": passed,
               "criteria": reasons, "comparison": comparison, "device": device,
               "seconds": round(time.time() - t_start, 1)}
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the image CAPTCHA-type classifier")
    ap.add_argument("--arch", default="all", help="nearest_centroid | softmax | mlp | all")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--resume", metavar="VERSION", help="resume an interrupted run of this version")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="train even if identical data/config was already trained")
    args = ap.parse_args(argv)
    cfg = load_config()
    for key, val in (("epochs", args.epochs), ("batch_size", args.batch_size), ("learning_rate", args.learning_rate)):
        if val is not None:
            cfg["training"][key] = val
    archs = None if args.arch == "all" else [args.arch]
    if not args.force and not args.resume:
        from training.model_registry import ImageModelRegistry
        done = already_trained(ImageModelRegistry(cfg), TrainingDataset(cfg).dataset_hash(), config_hash(cfg),
                               archs or list(cfg["training"]["architectures"]))
        if done:
            print(f"Already trained on identical data/config as {done} - nothing to do (use --force to retrain).")
            return 0
    try:
        s = run_training(cfg, archs, args.resume, resume=not args.no_resume)
    except TrainingStopped as exc:
        print(f"STOPPED: {exc}. Re-run with --resume <version> to continue.")
        return 3
    print(json.dumps({k: v for k, v in s.items() if k != "comparison"}, indent=2))
    print("architecture comparison (validation):")
    for a, r in s["comparison"].items():
        print(f"  {a:<17} acc {r['val_accuracy']:.4f}  macro-F1 {r['val_macro_f1']:.4f}  "
              f"params {r['param_count']:>8}  {r['train_seconds']}s")
    print(f"\nCandidate {s['version']} registered. To put it into production a named reviewer runs:\n"
          f"  scripts\\approve_model.bat {s['version']} \"Your Name\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())

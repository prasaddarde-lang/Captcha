# Training Completion Report: image-based CAPTCHA type classifier

**Date:** 2026-10-30 · **Environment:** Windows 11, Python 3.14.7, 8 CPU cores, no GPU, about 1 GB free disk
**Tests:** 209 / 209 pass (the original 175, unchanged, plus 34 new)
**Live numbers:** `reports/performance/training_report_latest.md` is regenerated after every training run and every loop iteration. The figures below are a snapshot from before `/loop` started.

## 1. What genuinely completed

| Phase | Result | Evidence |
|---|---|---|
| 1. Audit | Every Python, BAT and SQL file reviewed; findings and plan written; full backup taken | `reports/audit/initial_technical_audit_2026-10-30.md`, `backups/pre_training_backup_20261030_215206.tar.gz` |
| 2–3. Dataset | **10,000 synthetic samples** in 10 categories (1,000 each), split **8,000 / 1,000 / 1,000**, plus 400 loop-added math samples (train only) | `datasets/manifest.json`; generator `training/generate_dataset.py` |
| Validation | **All 10,400 files decoded and hashed: 0 errors**. Checks: split, balance, duplicates (SHA-256 and perceptual hash), metadata, licence | `reports/training_history/dataset_validation_latest.json` |
| 4. Training | 3 architectures trained and compared on CPU; MLP selected by validation macro-F1 | `models/checkpoints/img-1.0.0/` (model, per-epoch log, summary) |
| Evaluation | Fixed 1,000-sample test set: **98.20% accuracy**, macro-F1 0.982, FP rate 1.0%, FN rate 0.22% | `models/image_registry.json`, `reports/performance/` |
| 5. Continuous improvement | Controller works end to end. Iteration 0 targeted *math* (+400 samples): +0.2 pts accuracy, but its FP rate doubled, so it was correctly **not** recommended | `reports/training_history/state.json`, `history.jsonl` |
| 7. Integration | `image_file` input, plus `detection_source` / `image_model_version` / `image_category` / `image_confidence` in responses; REST `/health`, `GET /v1/models/image`, `POST /v1/models/image/{v}/approve`; all RPA adapters carry the new fields | `core/image_classifier.py`, `core/framework.py`, `api/server.py`, `tests/test_training.py` |
| 8. Scheduling | Bounded, resumable, single-instance controller; optional daily Windows task (you register it yourself) | `training/continuous_trainer.py`, `scripts/*.bat` |
| 10. Regression | Full suite passes; `run.bat` sample, `evaluate_model.bat` and `continuous_training.bat status` checked against the real project | 209 passed |

## 2. Architecture comparison (validation set)

| architecture | accuracy | macro-F1 | FP rate | params | train time |
|---|---|---|---|---|---|
| nearest centroid | 80.5% | 0.793 | 46.0% | 21k | 1 s |
| softmax regression | 98.0% | 0.980 | 6.0% | 21k | 6 s |
| **MLP (selected)** | **98.5%** | **0.985** | **3.0%** | 551k | 45 s |

## 3. img-1.0.0 per category (fixed test set, 100 samples each)

| category | precision | recall | F1 |
|---|---|---|---|
| text | 0.933 | 0.980 | 0.956 |
| text_distorted | 0.952 | 0.990 | 0.971 |
| **math (weakest)** | 0.978 | **0.880** | **0.926** |
| audio | 1.000 | 1.000 | 1.000 |
| image_classification | 1.000 | 0.990 | 0.995 |
| grid | 0.980 | 1.000 | 0.990 |
| recaptcha-style | 1.000 | 1.000 | 1.000 |
| hcaptcha-style | 1.000 | 0.990 | 0.995 |
| turnstile-style | 1.000 | 1.000 | 1.000 |
| none | 0.980 | 0.990 | 0.985 |

* **Detection:** FP rate 1.0%, FN rate 0.22% (arg-max). At the 0.80 runtime threshold: FP 1.0%, FN 2.7%, with 22 low-confidence cases abstaining (those fall back to page metadata or a human).
* **Speed and size:** about 4–13 ms per image end to end (decode + features + predict); model 2.2 MB, 551k parameters; process RSS about 240 MB during evaluation.
* **Character and exact-string accuracy:** not applicable. No OCR model exists.

## 4. Deliberately not built
The request's OCR / CRNN / Tesseract / EasyOCR / PaddleOCR text recognition, character
segmentation, maths solving, audio transcription and grid-object answering were **not**
implemented. Each of them answers CAPTCHAs, regardless of what data it is trained on. The dataset
records categories only; it never records rendered characters or answers.

## 5. `/loop` outcome and pending approval
The bounded `/loop` ran 4 ticks (controller iterations 1–4) and stopped itself after 3 iterations
without a validated improvement (patience). Every iteration targeted **math**, the weakest
category on validation, and 2,000 extra math training samples were added in total. Only
iteration 1 improved:

| iteration | candidate | test acc | FP | FN | result |
|---|---|---|---|---|---|
| 1 | **img-1.0.3** | **99.0%** | **0.0%** | 0.11% | recommended (+0.8 pts vs img-1.0.0, no regressions) |
| 2 | img-1.0.4 | 98.9% | 0.0% | 0.11% | not better |
| 3 | img-1.0.5 | 98.7% | 2.0% | 0.22% | not better (FP rose) |
| 4 | img-1.0.6 | 98.9% | 0.0% | 0.33% | not better |

img-1.0.3 per-category F1: text 0.970, text_distorted 0.990, math 0.960, audio 0.995,
image_classification 1.000, grid 1.000, recaptcha-style 1.000, hcaptcha-style 0.995,
turnstile-style 0.995, none 0.995. At the 0.80 runtime threshold: FP 0.0%, FN 1.9% (16
abstentions). Math has plateaued around 0.95–0.96: extra math samples no longer help, so the next
gain would need better features, not more data.

**In production: img-1.0.3**, approved by **Bipin** on 2026-10-30. The automatic post-promotion
re-check on the fixed test set passed (`model_registry.py verify` → ok). It was then confirmed
through `run.bat`: a synthetic grid image returns `image_selection / tile_grid` from
`image_model` img-1.0.3, a non-CAPTCHA image returns `NO_CAPTCHA`, and `/health` reports
`image_model_version: img-1.0.3`. The earlier candidates stay in the registry; img-1.0.3 is
archived in `models/archive/` for rollback (`scripts\rollback_model.bat`).

## 6. Issues found and fixed during this work
1. Windows' 260-character path limit broke dataset writes under this deep folder → flat
   `datasets/<split>/<id>.jpg` layout.
2. OpenCV 5 removed `cv2.HOGDescriptor` → equivalent HOG implemented in numpy.
3. Perspective warp dtype bug (float64 vs float32) in the generator.
4. Nearest-centroid temperature search could never select a value (its initial score was too high).
5. The reference-model choice could pick a *non-recommended* candidate → now only recommended
   ones qualify.
6. Synthetic text images in `learning/dataset_manager.py` stored their transcripts → now
   category only.
7. Training no longer repeats work on identical data and settings (`--force` overrides).

## 7. Caveats
* All metrics are on **synthetic** images. Real screenshots will differ, so measure on a pilot
  before relying on the image model alone. Page metadata stays authoritative whenever it is
  available.
* The fixed test set is reused for every candidate comparison, which slightly overstates gains
  from repeated selection. Weak categories are chosen on the validation set to limit this.
* Disk is tight. The configured augmentation budget (4,000 images, about 60 MB) and the
  `min_free_disk_mb` guard keep the loop bounded.

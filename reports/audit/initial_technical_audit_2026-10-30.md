# Initial Technical Audit — before the training upgrade

**Date:** 2026-10-30 · **Scope:** every Python, BAT, SQL and dependency file in `captcha_rpa_framework/`
**Backup:** `backups/pre_training_backup_20261030_215206.tar.gz` (full project, taken before any change)

## 1. Inventory

| Area | Files | Lines | Notes |
|---|---|---|---|
| Core | `core/*.py` (8 files) | 1,457 | Detector, signature classifier, resolution engine, HITL, I/O contract, orchestrator, Playwright, security |
| Integrations | `integrations/*.py` (5) | 337 | UiPath, A360, Blue Prism, PAD, generic |
| Learning | `learning/*.py` (4) | 741 | Analytics, synthetic HTML dataset, evaluation + SQLite registry, reports |
| Database | `db_manager.py`, `schema.sql` | 291 | SQLite history, verification queue, model registry |
| API / CLI | `api/server.py`, `main.py` | ~490 | FastAPI + operator console; CLI sub-commands |
| BAT | `install.bat`, `run.bat` | 186 | Installer (6 steps), launcher (exit code = status) |
| Tests | `tests/*.py` (11) | 1,452 | 175 tests, all passing at the end of the previous session |
| Dependencies | `requirements.txt` | — | pydantic, httpx, fastapi, uvicorn, playwright, opencv-python-headless, numpy, pytest |

## 2. Findings

| # | Question | Finding |
|---|---|---|
| 1 | Model architecture | One model: a **versioned JSON signature model** (`models/classifier_v1.0.0.json`) — weighted regexes over page *metadata* (iframe/script URLs, classes, ids, field names, prompt text). Explainable; scores 100 % on synthetic HTML fixtures *by construction*. |
| 2 | OCR engine | **None, by design.** Tesseract/`pytesseract` were deliberately excluded: reading CAPTCHA text is solving it. This upgrade keeps that boundary. |
| 3 | Training pipeline | `learning/model_evaluation.py`: perceptron-style re-weighting of the signature model from synthetic HTML + operator type-feedback; candidates registered in SQLite; deploy only after a **named human approval**; regression blocked. |
| 4 | Datasets | 8 static HTML fixtures (`test_data/fixtures`); synthetic HTML generated on demand (`dataset generate`); an optional 20-image synthetic text set. **No image dataset, no manifest with licence/split metadata, no fixed train/validation/test split.** |
| 5 | Is model training implemented? | Yes for the metadata model (weights only, no new signatures). **No image-based model exists.** |
| 6 | Missing capabilities | (a) Detection fails when the page source carries no signals — canvas-rendered widgets, closed shadow DOM, Citrix/VDI or screenshot-only RPA (the robot can only see pixels). (b) No training config, checkpoints, resume, resource limits, scheduler or training history. (c) No image-model registry / production slot / rollback. (d) No per-category FP/FN, latency, size or CPU/memory metrics. |
| 7 | Bugs / defects | (i) `learning/dataset_manager.generate_image_dataset` stores the **rendered characters** as labels — a ready-made OCR training set, inconsistent with the no-solving boundary. (ii) `core/classifier.default_model_path` evaluates `get_settings()` twice in one convoluted expression (works, hard to read). (iii) The venv cannot be created inside the project folder because the path exceeds Windows' 260-character limit (documented workaround: `CAPTCHA_FW_VENV`). (iv) Drive C: has ~1.5 GB free — constrains dataset format and rules out PyTorch. |
| 8 | GPU | No NVIDIA driver (`nvidia-smi` absent); OpenCV build has no CUDA. 8 CPU cores. → CPU training. |

## 3. Upgrade plan (agreed scope)

* **Add** an image-based *type* classifier (which kind of CAPTCHA a cropped screenshot shows — never its answer), trained on a 10,000-sample synthetic dataset with manifest, licence/authorisation references, fixed 8,000/1,000/1,000 split, validator and de-duplication.
* **Add** `training/` (generator, validator, dataset manager, trainer with checkpoints/resume/resource limits, evaluator, file-based image-model registry with human approval + automatic rollback to the last approved model, continuous trainer with persistent state, reports) and `scripts/*.bat`.
* **Integrate** the image model into detection as an *additional* signal (`image_file` input), keeping the metadata model, JSON contract, exit codes and all RPA adapters backward-compatible.
* **Not built:** OCR / CRNN / Tesseract / EasyOCR / PaddleOCR, character segmentation, maths-expression solving, audio transcription, grid object answering. Character-accuracy and exact-string metrics are therefore reported as *not applicable*.
* **Fix** finding 7(i): synthetic text images keep a category label only.

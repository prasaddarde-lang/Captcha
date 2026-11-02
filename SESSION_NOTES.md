# Session Notes — CAPTCHA RPA Framework

Build-and-training session. Covers what was built, every decision, every bug fixed, the
environment problems and workarounds, the training/`/loop` run, the approvals, and the final state.

- **Project location:** `D:\captcha_rpa_framework` (moved here from a temporary session folder — see §9)
- **Platform:** Windows 11, Python 3.14.7, 8 CPU cores, **no GPU**
- **Status:** ready for an RPA tool to connect. Only user-side configuration is pending (§11).
- **Tests:** 209 / 209 passing.
- **Live models:** page-detection signature model **1.1.0** (approved by Bipin); image type-classifier **img-1.0.3** (approved by Bipin).

---

## 1. What this framework is (and is not)

A reusable CAPTCHA **management** framework for RPA robots (UiPath, Automation Anywhere, Blue
Prism, Power Automate Desktop, Selenium, Playwright, generic Python). It **detects** a CAPTCHA on
a page or screenshot, **classifies its type**, and **routes** it to an *authorized* resolution
path. It returns structured JSON so the robot knows what to do next.

**It never solves or bypasses a CAPTCHA.** There is no OCR, no distorted-text reading, no maths
solving, no image/grid answering, no audio transcription, no third-party solver integration. A
CAPTCHA is the site owner telling automation to stop; the framework's job is to notice that and
hand off to something allowed:

1. **Official API** — if the site offers a documented API, route the robot to it instead of the UI.
2. **Human-in-the-loop** — an operator completes the challenge in the robot's browser; the robot resumes.
3. **Test mode** — simulated results, only on mock pages / local test hosts / the project's own synthetic images.

---

## 2. Timeline / phases

**Part 1 — the framework (2026-10-30).** Built the full project to the original specification:
detection engine, resolution engine, learning/analytics, RPA adapters, SQLite history, FastAPI
REST API + operator console, `install.bat` / `run.bat`, `main.py` CLI. 175 tests, all passing.

**Part 2 — image-model training (2026-10-30, same day).** On request, added a 10,000-sample
synthetic dataset and an **image-based CAPTCHA *type* classifier** (which *kind* of challenge a
screenshot shows — never its answer), with a training pipeline, model governance and a continuous
improvement loop. Grew the suite to 209 tests.

**`/loop` run (2026-10-30).** Ran the bounded continuous trainer; it produced and recommended
img-1.0.3, then stopped itself on the patience rule.

**Approvals & relocation (2026-10-30 → 10-31).** img-1.0.3 approved and promoted; the signature
model 1.1.0 approved and deployed by the user; project moved from the temporary session folder to
`D:\captcha_rpa_framework` with its own Python environment.

**Final verification (2026-10-31 → 11-02).** Full RPA request lifecycle re-checked on D:.

---

## 3. Scope decisions (what was declined, and why)

The second request asked to train OCR/CRNN models to read distorted text, solve maths, answer
grids and transcribe audio. **These were declined and not built**, because a model that reads or
answers a challenge is a CAPTCHA solver regardless of the data it is trained on. This was raised
with the user, who agreed to proceed with the detection-only version.

What was built instead:
- An **image classifier of challenge *type*** (10 categories, §5). It outputs a label like
  `grid` or `math`, never the characters or the answer.
- The dataset records the **category label only**. Rendered characters and arithmetic answers are
  never stored, so the dataset cannot be repurposed to train a solver.
- `evaluate_model.py` reports character-accuracy / exact-string-accuracy as **"not applicable"**
  with that reason, by design.

Also excluded on purpose: **Tesseract / pytesseract** (removed from requirements in Part 1 — in
this framework the only use for OCR would be reading challenge text).

One governance note: the original brief required human approval before any model goes live. The
continuous-training brief asked for automatic promotion; the human-approval step was **kept**.
Nothing is promoted without a named human.

---

## 4. Architecture / project layout

```
D:\captcha_rpa_framework\
├── main.py                 CLI: run / operator / dataset / model / feedback / report / serve
├── config.py               settings from CAPTCHA_FW_* environment variables
├── install.bat, run.bat    Windows installer and launcher
├── requirements.txt, sites.example.json, sites.json, .env.example
├── core/        detector.py, classifier.py (signature model), resolution_engine.py,
│                human_verification.py, response_handler.py, framework.py (orchestrator),
│                browser.py (Playwright), security.py,
│                image_features.py, image_classifier.py  (image model, runtime)
├── integrations/ uipath.py, automation_anywhere.py, blue_prism.py, power_automate.py, generic_rpa.py
├── learning/    analytics.py, dataset_manager.py, model_evaluation.py, performance_report.py
├── training/    generate_dataset.py, dataset_validator.py, dataset_manager.py, train_classifier.py,
│                evaluate_model.py, model_registry.py, continuous_trainer.py, training_report.py,
│                training_config.yaml, common.py
├── scripts/     train_10000.bat, continuous_training.bat, evaluate_model.bat, approve_model.bat,
│                rollback_model.bat, schedule_training.bat, _python.bat
├── database/    db_manager.py, schema.sql, captcha_events.db
├── api/         server.py (FastAPI + operator console at /operator)
├── models/      classifier_v1.0.0.json, classifier_v1.1.0.json (signature models),
│                image_registry.json, checkpoints/<ver>/, production/, archive/
├── datasets/    train/ validation/ test/ manifest.json (+ cache/)   — 12,000 synthetic images
├── reports/     audit/, training_history/, performance/
├── samples/     sample inputs + real captured outputs
├── test_data/   fixtures/ (static mock pages), test_responses.json
├── tests/       pytest suite (209 tests)
├── logs/  screenshots/  backups/
├── README.md, PROJECT_COMPLETION_REPORT.md, TRAINING_COMPLETION_REPORT.md, SESSION_NOTES.md
└── .venv/       the project's own Python environment (created by install.bat)
```

**Request flow:** validate input → load page (html / html_file / cdp_endpoint / url / image_file)
→ detect + classify → resolve (test mode / official API / human) → record history → respond.

**Two detection models, metadata is authoritative:**
- **Signature model** (`core/classifier.py`): versioned JSON of weighted regexes over page
  metadata (iframe/script URLs, classes, ids, field names, prompt text). Explainable.
- **Image model** (`core/image_classifier.py`): used when the robot only has a screenshot
  (canvas widgets, closed shadow DOM, Citrix/VDI). When both run, page metadata wins and the image
  result is reported alongside. The image model decides alone only above its confidence threshold
  (`CAPTCHA_FW_IMAGE_MIN_CONFIDENCE`, default 0.80); below it, it returns `NO_CAPTCHA` with a hint.

---

## 5. The dataset (10,000 base → 12,000 after the loop)

Synthetic, drawn with OpenCV. 10 categories, originally 1,000 each, split **8,000 / 1,000 /
1,000** (train / validation / test). Each image varies in font, background, contrast, resolution,
rotation, warping and noise. Includes hard negatives (e.g. "Remember me" checkbox rows, data
tables, plain image galleries). Provider categories are generic lookalike widgets with **no logos
or trademarks**.

Categories: `text`, `text_distorted`, `math`, `audio`, `image_classification`, `grid`,
`recaptcha`, `hcaptcha`, `turnstile`, `none`.

`datasets/manifest.json` records per sample: id, category, source, licence / authorisation
reference (`AUTH-SYNTH-2026-10-30`, project-generated, no external data), dimensions, split,
creation date, SHA-256 and perceptual hash (dhash). **Category label only** — no challenge
text/answers. `dataset_validator.py` decodes and hashes every file and checks the split, balance,
duplicates and metadata. Full validation: **12,000 samples, 0 errors**.

The `/loop` later added 2,000 extra `math` samples to the **train** split only (the test set never
changed), bringing the totals to train 10,000 / validation 1,000 / test 1,000 = 12,000.

---

## 6. Training, models and governance

- **Features:** HOG (implemented in numpy), colour layout, edge/grid-line profiles, aspect ratio —
  about 2,142 features per image, ~3–13 ms end to end.
- **Training:** CPU, numpy backend (any GPU is detected and reported, not used). Compares
  nearest-centroid, softmax regression and a 1-hidden-layer MLP; picks the best by validation
  macro-F1; evaluates once on the fixed test set. Reproducible from the config seed. Checkpoints
  every epoch, so a run stopped by the time/memory limit resumes exactly where it stopped.
- **Governance:** `CANDIDATE → APPROVED (named human) → PRODUCTION`, with `ARCHIVED` / `REJECTED`
  / `ROLLED_BACK`. Promotion criteria on the fixed test set: ≥ +0.2 pts accuracy, no category
  losing > 0.02 F1, no FP/FN rate rising > 0.5 pts. Promotion archives the previous model and
  re-checks the new one; if that check fails, the previous approved model is restored
  automatically. The running framework picks up a newly promoted model without a restart.

**img-1.0.0** (first model): 98.2% test accuracy, FP 1.0%, FN 0.22%. Architecture comparison on
validation: nearest-centroid 80.5%, softmax 98.0%, **MLP 98.5% (selected)**.

---

## 7. The `/loop` continuous-training run

Dynamic mode, self-paced, no external wait. 4 iterations, each targeting the weakest category
(math, chosen on validation), each adding 400 synthetic math training samples (test set fixed):

| Iteration | Candidate | Test acc | FP | FN | Result |
|---|---|---|---|---|---|
| 1 | **img-1.0.3** | **99.0%** | **0.0%** | 0.11% | recommended (+0.8 pts vs img-1.0.0) |
| 2 | img-1.0.4 | 98.9% | 0.0% | 0.11% | not better |
| 3 | img-1.0.5 | 98.7% | 2.0% | 0.22% | not better (FP rose) |
| 4 | img-1.0.6 | 98.9% | 0.0% | 0.33% | not better |

Stopped on **patience** (3 iterations with no validated improvement). Math plateaued at ~0.95–0.96
F1 — more math data no longer helps; the next gain would need better features, not more data.

**img-1.0.3 per-category F1:** text 0.970, text_distorted 0.990, math 0.960, audio 0.995,
image_classification 1.000, grid 1.000, recaptcha 1.000, hcaptcha 0.995, turnstile 0.995,
none 0.995. At the 0.80 runtime threshold: FP 0.0%, FN 1.9% (16 low-confidence abstentions that
fall back to page metadata or a human). Model ~2.2 MB.

The loop never approved or promoted anything.

---

## 8. Integration with the RPA interface

- New input field **`image_file`** (cropped screenshot, PNG/JPG ≤ 10 MB) alongside the existing
  `html` / `html_file` / `cdp_endpoint` / `url`. Can be sent alone (Citrix/VDI) or with page source.
- New response fields: **`detection_source`** (`metadata` | `image_model` | `metadata+image_model`),
  **`image_model_version`**, **`image_category`**, **`image_confidence`**.
- REST: `/health` reports both model versions; `GET /v1/models/image`; `POST /v1/models/image/{version}/approve`.
- All RPA adapters (UiPath, AA, Blue Prism, PAD, generic) carry the new fields. The existing JSON
  contract, exit codes and adapters are **backward-compatible** — a request without `image_file`
  behaves exactly as before.

**Status → exit code (unchanged):** NO_CAPTCHA/VERIFIED/OFFICIAL_API_ROUTE/TEST_VERIFIED = 0;
CAPTCHA_DETECTED = 10; HUMAN_VERIFICATION_REQUIRED = 11; PENDING = 12; TEST_FAILED = 20;
REJECTED = 21; EXPIRED = 22; TIMEOUT = 30; INVALID_INPUT = 40; POLICY_BLOCKED = 41;
NETWORK_ERROR = 50; ERROR = 99.

---

## 9. Environment issues and workarounds

1. **Windows 260-char path limit.** The original project sat in a very deep temporary folder, so
   `pip` could not create a venv there and the nested dataset layout broke file writes.
   Workarounds: a short-path venv during development; a **flat** `datasets/<split>/<id>.jpg`
   layout; `CAPTCHA_FW_VENV` support; and ultimately moving the project to `D:\captcha_rpa_framework`.
2. **Disk space on C:** C: was near-full throughout (dropped as low as ~0.5 GB). The Playwright
   Chromium download (~600 MB) filled it and was removed; the framework falls back to the
   preinstalled **Edge**, and `install.bat /nobrowser` skips the download. D: has ~255 GB free.
3. **PowerShell is blocked in this app on this PC** (`EPERM ... powershell.exe`). The Terminal
   panel launches PowerShell and fails to start ("Shell exited"). **Use Command Prompt (cmd.exe).**
   This is a machine policy/antivirus matter, left for whoever manages the PC.
4. **Temp venv disappeared.** The development environment in `%TEMP%\crf_venv` lost its
   `python.exe` (antivirus or Temp cleanup, not anything the session did). It was replaced by the
   project's own `.venv` created by `install.bat` on D:, then deleted to free C: space.
5. **OpenCV 5** removed `cv2.HOGDescriptor` from the main module → HOG reimplemented in numpy.

---

## 10. Bugs found and fixed (all during this session)

**Part 1 (framework):**
- Secret redaction order let a `Bearer <token>` leak in logs → bearer rule runs first.
- Blocking human-verification race returned `EXPIRED` instead of `TIMEOUT` → grace period so the
  waiter cancels cleanly.
- Generic text-CAPTCHA signals fired on provider fields (`g-recaptcha-response` etc.) → excluded.
- Invalid input echoed a raw `task_id` → sanitised to `UNKNOWN`.
- `check_status` reported `captcha_detected:false` for a just-verified CAPTCHA → reports the type.
- CLI `--verbose` flag clash; relative `html_file` resolution from other working directories.

**Part 2 (training):**
- 260-char path broke dataset writes → flat layout.
- OpenCV 5 `HOGDescriptor` missing → numpy HOG.
- Perspective-warp dtype bug (float64 vs float32) in the generator.
- Nearest-centroid temperature search could never select a value (initial score too high).
- Reference-model choice could pick a non-recommended candidate → only recommended ones qualify.
- Synthetic text images stored their transcripts → category only.
- Added an "already trained on identical data/config" skip guard (`--force` overrides).

---

## 11. Approvals and current live state

- **Image model img-1.0.3 → PRODUCTION**, approved by **Bipin** (2026-10-30). Approval was run on
  the user's behalf at their request, with their name as approver. Post-promotion re-check on the
  fixed test set passed. Archived in `models/archive/` for rollback.
- **Signature model 1.1.0 → DEPLOYED**, approved by **Bipin** (recorded 2026-10-31 00:25 local).
  This was done by the user themselves through the normal approval path; it replaced the 1.0.0
  baseline (now RETIRED). Confirmed as a legitimate human-approved upgrade.
- `/health` reports `model_version: 1.1.0`, `image_model_version: img-1.0.3`.

---

## 12. Final verification (done on D:)

- `install.bat /nobrowser` on D: → own `.venv`, **209/209 tests pass**.
- Dataset: **12,000 samples, 0 errors**. Production image model `verify` → ok.
- Full RPA lifecycle through `run.bat`:
  - reCAPTCHA page (production) → `HUMAN_VERIFICATION_REQUIRED` (exit 11) with a `request_id`
  - operator `complete` → `COMPLETED`
  - robot `check_status` → `VERIFIED` (exit 0)
  - image-only (Citrix/VDI), test mode → `TEST_VERIFIED`, type `turnstile`, source `image_model`, img-1.0.3 (exit 0)
  - ordinary page → `NO_CAPTCHA` (exit 0)

---

## 13. Pending — all user-side configuration (none blocks a first connection)

1. **`sites.json`** — list sites with official APIs; store tokens as environment variables (names only in the file).
2. **`CAPTCHA_FW_API_KEY`** — only if the REST API is called from another machine (also add TLS: `CAPTCHA_FW_SSL_CERTFILE` / `_KEYFILE`).
3. **`CAPTCHA_FW_NOTIFY_WEBHOOK`** — optional HTTPS Teams/Slack webhook for operator alerts.
4. **Real-screenshot pilot** — the image model's accuracy is measured on synthetic images only; test on real crops before relying on `image_file` detection alone.
5. **Free space on C:** keep an eye on it; all data/training lives on D:.
6. **Operator process** — decide who completes verifications and how they reach the robot's browser (RDP, or a headed/CDP browser) and the operator console (`run.bat serve` → `http://127.0.0.1:8765/operator`).

---

## 14. Key commands (run from `D:\captcha_rpa_framework` in Command Prompt)

```bat
run.bat samples\sample_input_test.json              :: smoke test
run.bat C:\bots\task.json C:\bots\result.json uipath:: a robot call (file in, file out; exit = status)
run.bat operator list                               :: pending verifications
run.bat operator complete HV_xxxx --operator "Bipin":: operator finishes one
run.bat serve                                       :: REST API + /operator console on :8765

scripts\evaluate_model.bat                          :: live image model metrics (test set)
scripts\continuous_training.bat status              :: training state + candidates awaiting approval
scripts\continuous_training.bat 3 30                :: up to 3 iterations / 30 min (bounded, resumable)
scripts\approve_model.bat img-1.0.X "Bipin"         :: a named human promotes a recommended model
scripts\rollback_model.bat                          :: restore the previous approved model
```

The project finds its own `.venv` automatically — no `CAPTCHA_FW_VENV` needed on D:.
Logs: `logs\captcha_framework.log`. History DB: `database\captcha_events.db`.
Reports: `reports\performance\training_report_latest.md`.

Full detail and troubleshooting: `README.md`. Phase reports:
`PROJECT_COMPLETION_REPORT.md` (framework), `TRAINING_COMPLETION_REPORT.md` (image model).

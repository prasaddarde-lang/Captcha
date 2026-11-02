# Universal CAPTCHA Management Framework for RPA

A Windows-friendly Python framework that **detects** CAPTCHA challenges in pages your
robots work on, **classifies** them (reCAPTCHA, hCaptcha, Cloudflare Turnstile, Arkose/FunCaptcha,
GeeTest, AWS WAF, text, image-selection, audio) and **routes** them to an *authorized* resolution
path, returning structured JSON to UiPath, Automation Anywhere, Blue Prism, Power Automate
Desktop, Python/Selenium/Playwright scripts, or anything that can run a command or call HTTP.

> **What this framework will not do.** It never solves, bypasses or outsources CAPTCHAs, and it
> never turns off a site's security controls. A CAPTCHA is the site owner telling automation to stop,
> so the framework's job is to notice that quickly and hand off to something you are allowed to use:
> the site's own API, or a human operator. Test mode works only on mock pages and local test hosts.

---

## Contents
1. [How it works](#how-it-works)
2. [Installation](#installation)
3. [Running a task](#running-a-task)
4. [JSON contract & status codes](#json-contract--status-codes)
5. [Resolution methods](#resolution-methods)
6. [RPA platform integration](#rpa-platform-integration) (UiPath, AA, Blue Prism, PAD, Python)
7. [REST API & operator console](#rest-api--operator-console)
8. [Learning, models & reports](#learning-models--reports)
9. [Image model training & continuous improvement](#image-model-training--continuous-improvement)
10. [Configuration & security](#configuration--security)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Project layout](#project-layout)

---

## How it works

```
RPA tool ──JSON──▶ main.py / run.bat / REST / adapter
                     │
                     ▼
          core/framework.py  (validate → load page → detect → resolve → record → respond)
            │            │                    │                         │
   page source:     core/detector.py     core/resolution_engine.py   database/ (SQLite)
   • html / html_file   (DOM metadata)     1. test_mode   (mock pages only)
   • cdp_endpoint       core/classifier.py 2. official_api (site's documented API)
     (robot's browser)  (versioned model)  3. human_in_the_loop (operator)
   • url (Playwright)
```

* **Detection** reads metadata only: iframe and script URLs, class names, ids, `data-*` attributes,
  form field names, and the page's prompt text. Challenge images, audio and answers are never read
  or stored. The widget's response field is checked only for *whether* it is filled, which tells
  the framework that a human has already completed it.
* **Classification** is a versioned, explainable signature model (`models/classifier_v*.json`).
  Every response lists the `signals` that fired.
* **Resolution** runs through a pluggable chain of resolvers (see [Resolution methods](#resolution-methods)).

## Installation

Requirements: Windows 10/11, **Python 3.12+** on PATH, internet access to PyPI.

```bat
install.bat              :: full install: venv, dependencies, Chromium for Playwright, verify, tests
install.bat /nobrowser   :: skip the ~600 MB browser download (Edge/Chrome can still be used)
install.bat /notest      :: skip the test suite
```

The installer:
1. checks for Python 3.12+ (`py -3`, then `python`)
2. creates `.venv` (set `CAPTCHA_FW_VENV=C:\venvs\captcha_fw` to put it somewhere else)
3. installs `requirements.txt`
4. runs `playwright install chromium` (optional)
5. checks the imports and runs `samples\sample_input_test.json`
6. runs the test suite and prints a **SETUP RESULTS** table

Manual install:
```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m playwright install chromium   & rem optional
copy sites.example.json sites.json
```

The browser download is optional. On Windows, `browser: "edge"` (or `"chrome"`) uses the browser
that is already installed. If neither is available, the framework falls back to Edge, then to
bundled Chromium.

## Running a task

```bat
run.bat samples\sample_input_test.json                          :: JSON to stdout, exit code = status
run.bat C:\bots\task.json C:\bots\result.json uipath            :: also writes the result file
type task.json | run.bat                                        :: stdin
run.bat --input-json "{\"task_id\":\"A1\",\"html_file\":\"C:\\bots\\page.html\"}" --timeout 60
run.bat operator list                                           :: any main.py sub-command
```

```bat
python main.py run -i task.json -o result.json -p blue_prism --timeout 90 --mode production
```

Only the JSON result goes to stdout. Diagnostics go to `logs\captcha_framework.log` (JSON lines,
with secrets redacted), and with `-v` they also go to stderr.

### Getting the page to the framework (pick one)

| Field | Use when | Notes |
|---|---|---|
| `html` | Your bot can read the page source (UiPath *Get Attribute "outerHTML"*, Selenium `driver.page_source`) | Simplest and most portable |
| `html_file` | Your bot saved the page source to a file | Paths under the project, current directory or user profile only |
| `cdp_endpoint` (+`url`) | **Recommended for live use.** Start the robot's Chrome/Edge with `--remote-debugging-port=9222` | Attaches to the robot's *own* browser: reads frames, takes screenshots, watches for the operator to finish. Never clicks the widget |
| `url` | Detection or testing in a browser that Playwright launches | `headless:false` opens a window an operator can use |

## JSON contract & status codes

**Input** (`TaskRequest`, unknown fields are rejected):

```json
{
  "task_id": "RPA_001",
  "platform": "UiPath",
  "browser": "Chrome",
  "action": "detect_and_handle",
  "timeout": 120,
  "mode": "production",
  "cdp_endpoint": "http://127.0.0.1:9222",
  "url": "https://portal.example.org/login",
  "screenshot": true,
  "wait_for_operator": false
}
```

| Field | Values / default |
|---|---|
| `action` | `detect` (classify only), `detect_and_handle` (default), `check_status`, `cancel` |
| `mode` | `production` (default) or `test` |
| `timeout` | 1–3600 s. Hard limit for the whole task; also how long an operator has to respond |
| `wait_for_operator` | `true` blocks until the operator finishes; `false` returns at once with a `request_id` you poll |
| `request_id` | Required for `check_status` / `cancel` |
| `site` | Key in `sites.json`. Defaults to the URL host |
| `test_response` | `pass` / `fail` / `timeout`, test mode only |
| `image_file` | Optional cropped screenshot of the suspected challenge element (PNG/JPG, at most 10 MB). The image model classifies its *type*. It can be sent alone (for Citrix/VDI robots that only see pixels) or together with `html`/`url` |

**Output** (`TaskResponse`):

```json
{
  "task_id": "RPA_001",
  "status": "HUMAN_VERIFICATION_REQUIRED",
  "captcha_detected": true,
  "captcha_type": "recaptcha",
  "captcha_subtype": "v2_checkbox",
  "confidence": 0.9765,
  "resolution_method": "human_in_the_loop",
  "message": "Operator verification required",
  "request_id": "HV_2d575466247741d6",
  "next_action": "Show the browser to an operator, then poll with action=check_status and this request_id",
  "official_api": null,
  "signals": [{"captcha_type": "recaptcha", "signal": "recaptcha_class", "weight": 2.5}],
  "model_version": "1.0.0",
  "duration_ms": 31,
  "exit_code": 11
}
```

Responses also carry `detection_source` (`metadata`, `image_model` or `metadata+image_model`),
`image_model_version`, `image_category` and `image_confidence`. When the image model is not used,
these are `null`. Real outputs for each scenario are in `samples\sample_output_*.json`.

| Status | Exit | Meaning / what the bot should do |
|---|---|---|
| `NO_CAPTCHA` | 0 | Continue |
| `VERIFIED` | 0 | Operator finished, or the widget was already completed. Continue |
| `OFFICIAL_API_ROUTE` | 0 | Switch to the site API listed in `official_api` |
| `TEST_VERIFIED` | 0 | Simulated success (test mode) |
| `CAPTCHA_DETECTED` | 10 | `detect` only. See `resolution_method` for the planned route |
| `HUMAN_VERIFICATION_REQUIRED` | 11 | Alert an operator, then poll `check_status` |
| `PENDING` | 12 | Operator has not finished yet. Poll again |
| `TEST_FAILED` | 20 | Simulated failure (test mode) |
| `REJECTED` | 21 | Operator rejected it, or the request was cancelled. Escalate |
| `EXPIRED` | 22 | Nobody responded in time. Retry or escalate |
| `TIMEOUT` | 30 | Task exceeded `timeout` |
| `INVALID_INPUT` | 40 | Fix the request (see `message`) |
| `POLICY_BLOCKED` | 41 | Test mode used against a non-test host |
| `NETWORK_ERROR` | 50 | Page, CDP or API unreachable |
| `ERROR` | 99 | Unexpected error. See the log |

## Resolution methods

### 1. Human-in-the-loop (default in production)
1. The framework creates a verification request (`HV_…`) in SQLite, takes a screenshot if you
   asked for one and it is allowed, and notifies operators. Set `CAPTCHA_FW_NOTIFY_WEBHOOK` to an
   **HTTPS** Teams, Slack or ITSM webhook if you want a push notification.
2. The operator completes the challenge **in the robot's browser window** (RDP into the robot
   session, or use a headed browser), then marks the request done in one of three ways:
   * `run.bat operator complete HV_xxx --operator "Jane Doe"`
   * the web console at `http://127.0.0.1:8765/operator`
   * `POST /v1/verifications/HV_xxx/complete`
3. The workflow resumes:
   * **Blocking** (`wait_for_operator:true`): the call returns `VERIFIED` when the operator
     finishes. With `cdp_endpoint`, it also notices on its own when the widget is completed and
     returns without waiting for the operator to click anything.
   * **Asynchronous**: the bot polls `{"action":"check_status","request_id":"HV_xxx"}` until it
     gets `VERIFIED`.

### 2. Official API
Use this when a site that challenges your robot also offers a documented API. Register the site in
`sites.json`:
```json
"portal.example.com": {"official_api": {"base_url": "https://api.example.com/v1", "health_path": "/status",
  "auth_env": "EXAMPLE_PORTAL_API_TOKEN", "docs_url": "https://developer.example.com"}}
```
The framework checks the API with the credential named in `auth_env` (read from the environment,
HTTPS only). If the API answers, it returns `OFFICIAL_API_ROUTE` so your workflow moves to the API.
If the API returns 401/403 or 429, it stops immediately without retrying and falls back to a human.
For network errors it retries a limited number of times with backoff. Every attempt is recorded
in `api_attempts`.

**Provider token verification for sites you own:** `POST /v1/provider/verify` calls the provider's
documented `siteverify` endpoint (reCAPTCHA, hCaptcha or Turnstile) for a site marked
`"owned": true`, using the secret named in `provider_secret_env`. This is backend verification of
a token that a *human* produced. It does not generate tokens.

### 3. Test mode (development only)
`"mode":"test"` returns configurable simulated results (`test_data\test_responses.json` or
`test_response`). It is accepted only when the page is on an allow-listed test host
(`localhost`, `127.0.0.1`, `*.test`, `*.localhost`, or `CAPTCHA_FW_TEST_MODE_HOSTS`) or when the
HTML is a synthetic fixture marked `data-mock-captcha`. Anything else gets `POLICY_BLOCKED`.

### Adding your own resolver
Implement `name`, `can_handle(ctx)` and `resolve(ctx) -> ResolutionOutcome | None`, then set
`framework.engine.resolvers = [...]`. `tests/test_resolution.py::test_custom_resolver_chain` has an
example.

## RPA platform integration

### UiPath
**Option A: Python activities** (`UiPath.Python.Activities`)
1. **Python Scope**: Path = `<project>\.venv\Scripts`, Version = Auto, Target = x64
2. **Load Python Script**: File = `<project>\integrations\uipath.py` → `pyScript`
3. **Invoke Python Method**: Name = `handle`, Input parameters = `{ jsonInput }`, Instance = `pyScript` → `pyResult`
4. **Get Python Object** (TypeArgument `String`) → `jsonResult`
5. **Deserialize JSON** → `jo`; branch on `jo("status").ToString`

Build `jsonInput` with, for example, `"{""task_id"":""" & in_TaskId & """,""platform"":""UiPath"",""html"":" & Newtonsoft.Json.JsonConvert.ToString(pageHtml) & "}"`.
The page source comes from **Get Attribute** `outerHTML` on the `<html>` element, or the
**Inject Js Script** `function(){return document.documentElement.outerHTML}`.

**Option B: Start Process** (no Python activities needed)
* **Write Text File** `task.json` → **Start Process** `run.bat` with arguments
  `"C:\bots\task.json" "C:\bots\result.json" uipath` → **Read Text File** → **Deserialize JSON**.
* Or use **Invoke Power Shell** / `System.Diagnostics.Process` to read the exit code.

**Option C: HTTP Request** activity: `POST http://127.0.0.1:8765/v1/tasks`, body = JSON, header `X-API-Key`.

Recommended live pattern: open the target in **Use Application/Browser** with Chrome started with
`--remote-debugging-port=9222`, then send `{"cdp_endpoint":"http://127.0.0.1:9222","url":"<prefix of page url>","wait_for_operator":true,"screenshot":true}`.
If the result is `HUMAN_VERIFICATION_REQUIRED`, use a **Retry Scope** that polls `check_status`
every 10 s, or a long-running workflow with an Action Center task for the operator.

### Automation Anywhere (A360)
* **Python script → Open**: file `<project>\integrations\automation_anywhere.py`, Python runtime =
  the venv (set it in the bot's Python package settings)
* **Python script → Execute function**: `aa_handle`, input `$jsonInput$`, output `$jsonResult$`
  (`aa_status` returns only the status string)
* The input can also be `task_id=A1;html_file=C:\bots\page.html;timeout=60` for bots that do not
  build JSON
* Without the Python package: **Application → Open program/file** `run.bat` with input and output
  file arguments, then **JSON → Start session** on the result file

### Blue Prism
* **Utility – Environment → Start Process** (wait for exit): `run.bat "<task.json>" "<result.txt>" blueprism`
* The result file contains flat `key=value` lines (`status=…`, `request_id=…`). Read it with
  **Utility – File Management → Read All Text From File** and split each line on the first `=`
  into a single-row collection
* With the Web API option, call `POST /v1/tasks` and map the JSON with the JSON VBO
* In Python, `integrations.blue_prism.to_collection(response)` gives the same flat row

### Power Automate Desktop
* **Run DOS command**: `"<project>\run.bat" "%TaskFile%" "%ResultFile%" powerautomate`
* **Read text from file** → **Convert JSON to custom object** → `%Result['status']%`
* PAD's built-in *Run Python script* uses IronPython, which cannot load this framework. Use
  *Run DOS command* or *Invoke web service* (`POST /v1/tasks`) instead
* Cloud flows: expose the REST API through an on-premises data gateway or a custom connector

### Python, Selenium, Playwright and other tools
```python
import json, sys
sys.path.insert(0, r"C:\captcha_rpa_framework")
from integrations.generic_rpa import handle_json

result = json.loads(handle_json(json.dumps({
    "task_id": "SEL_001", "platform": "selenium", "action": "detect_and_handle",
    "html": driver.page_source, "url": driver.current_url, "timeout": 300})))
if result["status"] == "HUMAN_VERIFICATION_REQUIRED":
    ...  # alert an operator, then poll action=check_status with result["request_id"]
```
Any other tool can call the CLI, `run.bat`, or the REST API.

## REST API & operator console

```bat
run.bat serve                         :: http://127.0.0.1:8765  (docs: /docs, operator console: /operator)
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and active model version |
| POST | `/v1/tasks` | Run a task (same JSON as the CLI) |
| GET | `/v1/verifications?status=PENDING` | Operator queue |
| GET | `/v1/verifications/{id}` | Request details |
| POST | `/v1/verifications/{id}/complete` · `/reject` | `{"operator":"Jane","notes":"…"}` |
| GET | `/v1/verifications/{id}/screenshot` | Screenshot (only from the screenshot folder) |
| GET | `/v1/executions` · `/v1/reports/performance` | History and analytics |
| POST | `/v1/feedback` | `{"execution_id":12,"actual_type":"hcaptcha"}` |
| GET/POST | `/v1/models`, `/v1/models/{v}/approve`, `/v1/models/{v}/deploy` | Model governance |
| POST | `/v1/provider/verify` | Siteverify for **owned** sites |

**Authentication:** set `CAPTCHA_FW_API_KEY`; clients send it as `X-API-Key`. If the variable is
not set, only loopback clients are accepted.
**HTTPS:** set `CAPTCHA_FW_SSL_CERTFILE` and `CAPTCHA_FW_SSL_KEYFILE`. The server warns if it is
bound to a non-loopback interface without TLS.

## Learning, models & reports

```bat
run.bat dataset generate --per-label 20 --images 20 --mock-site   :: synthetic fixtures
run.bat model evaluate                                            :: accuracy, per-class P/R/F1, confusion
run.bat feedback --execution-id 42 --actual-type hcaptcha --reviewer "Jane"
run.bat model train                                               :: registers a CANDIDATE only
run.bat model approve 1.0.1 --approver "Jane Doe"                 :: named human required
run.bat model deploy 1.0.1                                        :: only APPROVED models
run.bat model deploy 1.0.0                                        :: roll back to an earlier approved model
run.bat report                                                    :: logs\reports\performance_report_*.json/.html
```

* **History:** every execution is stored with its status, type, confidence, method, duration,
  model version and the names of the signals that fired. Challenge content is never stored.
* **Analytics** (`learning/analytics.py`): volumes by type, status, method and platform;
  accuracy against operator labels; operator completion rate and time to complete; official API
  success rate; recurring failures (repeated misclassifications, hosts that keep failing, signals
  that show up in errors) with recommendations.
* **Training** adjusts signature weights using synthetic fixtures plus operator feedback about the
  *type* of challenge (as signal sets). It never trains on third-party challenge images or audio,
  and it never adds new signatures automatically; those go through code review.
* **Governance:** `CANDIDATE → APPROVED (named reviewer) → DEPLOYED`. Approval is refused if the
  candidate scores worse than the baseline, unless `--allow-regression` is passed explicitly. Only
  one model is deployed at a time, and rollback is supported.

## Image model training & continuous improvement

The signature model needs page metadata. When the robot can only *see* the screen (canvas
widgets, closed shadow DOM, Citrix/VDI), an **image model** can tell which *type* of challenge a
cropped screenshot shows: text, distorted text, maths, audio, image classification, tile grid,
reCAPTCHA-, hCaptcha- or Turnstile-style widget, or none. It never reads, segments or answers a
challenge. A detected type still goes through the same authorized routes: official API, a human
operator, or test mode.

**Dataset.** `training/generate_dataset.py` draws 10,000 synthetic images (1,000 per category,
split 8,000 / 1,000 / 1,000) with OpenCV. Each image varies in font, background, contrast,
resolution, rotation, warping and noise. The set includes hard negatives such as "Remember me"
checkbox rows, data tables and plain image galleries. The provider categories are generic
lookalike widgets with no logos. `datasets/manifest.json` records, for each sample, its id,
category, source, licence/authorisation reference, size, split, creation date, SHA-256 and
perceptual hash. It records only the category: the rendered characters and maths answers are
deliberately never stored. `training/dataset_validator.py` decodes and hashes every file and
checks the split, balance, duplicates and metadata.

**Training.** `training/train_classifier.py` extracts layout features (HOG, colour layout,
grid-line profiles) and compares nearest-centroid, softmax regression and a one-hidden-layer MLP.
It picks the best by validation macro-F1, then evaluates once on the fixed test set. Training runs
on CPU (any GPU is detected and reported; the numpy backend does not use one) and is
reproducible from the configured seed. It writes a checkpoint every epoch, so a run stopped by a
time or memory limit resumes exactly where it stopped. All settings are in
`training/training_config.yaml`.

**Promotion.** Every trained model is registered as a **CANDIDATE** in `models/image_registry.json`.
A candidate goes to production only when a *named person* approves it, and only if it meets the
criteria on the fixed test set: at least +0.2 points of accuracy, no category losing more than
0.02 F1, and no false-positive or false-negative rate rising more than 0.5 points. Promotion
archives the previous model and re-checks the new one; if that check fails, the previous
approved model is restored automatically. The running framework picks up a newly promoted model
without a restart.

```bat
scripts\train_10000.bat                         :: generate + validate + train/compare + report (skips work already done)
scripts\evaluate_model.bat [version] [split]    :: metrics on the fixed test set (default: production)
scripts\approve_model.bat img-1.0.0 "Jane Doe"  :: named approval + promotion (+ automatic rollback on failure)
scripts\rollback_model.bat [version]            :: restore a previously approved model
scripts\continuous_training.bat [iterations] [minutes]   :: bounded improvement run
scripts\continuous_training.bat status
scripts\schedule_training.bat 02:00             :: optional daily Windows task (you run it; /delete removes it)
```

**Continuous improvement.** Each iteration of `training/continuous_trainer.py` does the following:
1. checks the production model, rolling back automatically if the check fails
2. picks the weakest category, using validation F1
3. adds 400 new synthetic *training* samples for that category (the test set never changes)
4. validates the dataset
5. trains a candidate
6. compares the candidate with production (or, before anything is approved, with the best
   recommended candidate)
7. records the result

A candidate that meets the criteria is marked *recommended* and waits for approval. Nothing is
promoted automatically. A run stops at the iteration or time limit, at the augmentation budget
(4,000 extra samples), when disk space is low, or after 3 iterations with no validated
improvement. State is kept in `reports/training_history/state.json`, so an interrupted run
resumes. A lock file stops two controllers from running at once. Training that has already been
done on identical data and settings is skipped.

**Running it with Claude Code `/loop`.** In a Claude Code session, run:
`/loop run one bounded continuous-training iteration: scripts\continuous_training.bat 1 20, then summarise the result; stop the loop when the status reports a stop reason of patience, budget or disk`.
* **Stop it:** press Esc, or tell Claude "stop the loop". The current iteration finishes or
  leaves a checkpoint either way.
* **Restart it:** run the same `/loop` command again. The controller resumes from `state.json`
  and any checkpoint, and never repeats an iteration that already finished.
* **Without Claude:** use `scripts\continuous_training.bat` or the scheduled task.

**Using the image model from RPA.** Capture the suspected element (UiPath *Take Screenshot* on
the element, PAD *Take screenshot of UI element*, Blue Prism *Read → Get Image*), save it as a
PNG under the robot profile or project folder, and send `{"task_id":"…","image_file":"C:\\…\\crop.png"}`.
Send `html`/`url` as well whenever you have them: page metadata stays authoritative, and the
image model then only adds a second opinion. The image model decides on its own only above
`CAPTCHA_FW_IMAGE_MIN_CONFIDENCE` (default 0.80). Below that it returns `NO_CAPTCHA` with a hint.

Reports are written to `reports/performance/training_report_latest.md` (with a JSON copy),
evaluations to `reports/performance/eval_*.json`, and per-epoch logs to
`models/checkpoints/<version>/train_log.jsonl`. The audit is in `reports/audit/`.

## Configuration & security

| Environment variable | Default | Purpose |
|---|---|---|
| `CAPTCHA_FW_API_KEY` | – | REST API key |
| `CAPTCHA_FW_NOTIFY_WEBHOOK` | – | HTTPS webhook for operator notifications |
| `CAPTCHA_FW_SSL_CERTFILE` / `_KEYFILE` | – | TLS for the REST API |
| `CAPTCHA_FW_DB_PATH` | `database\captcha_events.db` | SQLite file |
| `CAPTCHA_FW_LOG_DIR`, `_SCREENSHOT_DIR`, `_TEST_DATA_DIR`, `_MODELS_DIR`, `_SITES_FILE` | project folders | Paths |
| `CAPTCHA_FW_SCREENSHOTS_ENABLED` | `1` | Global screenshot switch |
| `CAPTCHA_FW_SCREENSHOT_RETENTION_HOURS` | `24` | Screenshot retention |
| `CAPTCHA_FW_TEST_MODE_HOSTS` | `localhost,127.0.0.1,::1,mock.test,example.test` | Hosts where test mode is allowed |
| `CAPTCHA_FW_REQUIRE_HTTPS` | `1` | Outbound API and webhook calls must use HTTPS |
| `CAPTCHA_FW_MAX_TIMEOUT`, `_HTTP_TIMEOUT`, `_HTTP_RETRIES`, `_OPERATOR_POLL_INTERVAL` | 3600 / 15 / 2 / 2 | Limits |
| `CAPTCHA_FW_PYTHON`, `CAPTCHA_FW_VENV` | – | Interpreter or venv for `run.bat` and `scripts\*.bat` |
| `CAPTCHA_FW_IMAGE_MIN_CONFIDENCE` | `0.80` | Image model never declares a CAPTCHA on its own below this |
| `CAPTCHA_FW_DATASETS_DIR`, `_REPORTS_DIR`, `_TRAINING_CONFIG` | `datasets`, `reports`, `training\training_config.yaml` | Training paths |
| Names referenced in `sites.json` (`auth_env`, `provider_secret_env`) | – | Site API tokens and provider secrets |

Security measures built in:
* No secrets in code or config. `sites.json` holds only the *names* of environment variables.
* Pydantic validation with unknown fields rejected; task id and URL whitelists; no credentials
  embedded in URLs; `html_file` path containment.
* Parameterized SQL only. Only metadata is stored. Screenshots are opt-in, purged after the
  retention window, and served only from their own folder.
* Logs are JSON lines with redaction for API keys, bearer tokens and widget response values.
* No retry storms: 401/403/429 stop immediately, and other retries are bounded with backoff.
* Test mode is refused on production hosts. Nothing interacts with or bypasses CAPTCHA widgets.

## Testing

```bat
.venv\Scripts\python -m pytest -q tests
```
There are 209 tests covering detection, classification, human verification (blocking, async,
auto-clearance), the official API and provider verification, JSON I/O and the CLI, all four RPA
adapters, timeouts, network failures, the database, learning and governance, the REST API, a
live browser against a local mock site, the `install.bat`/`run.bat` launchers, and the image-model
pipeline (dataset validation, reproducible training, checkpoint resume, named-human approval,
regression blocking, automatic rollback, the continuous trainer's lock, budget and patience, and
framework/API/RPA/BAT integration). They use
synthetic fixtures (`test_data\fixtures`), mock HTTP transports and a local HTTP mock website.
Browser tests block all non-loopback traffic and are skipped if no browser can be launched.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python 3.12 or newer was not found` | Install Python from python.org and tick *Add to PATH*, or set `CAPTCHA_FW_PYTHON` |
| pip fails with `No such file or directory … build_tracker.py` | The project path is too deep for Windows' 260-character limit. Enable long paths (`LongPathsEnabled=1`) or `set CAPTCHA_FW_VENV=C:\venvs\captcha_fw` and run `install.bat` again |
| pip times out or hits SSL errors behind a proxy | `set HTTPS_PROXY=http://proxy:port`, and add `--trusted-host` / `PIP_CERT` if your company inspects TLS |
| `Playwright browser download failed` / not enough disk space | Use `install.bat /nobrowser` and `"browser":"edge"` (Edge is preinstalled), or free about 700 MB |
| `NETWORK_ERROR … ECONNREFUSED 127.0.0.1:9222` | The robot's browser was not started with `--remote-debugging-port=9222`, or another tool owns the port. Close all browser windows first |
| Result is always `POLICY_BLOCKED` | Test mode was used against a real host. Use `mode: production`, or add the host to `CAPTCHA_FW_TEST_MODE_HOSTS` if it really is your test environment |
| Always `HUMAN_VERIFICATION_REQUIRED` even though `sites.json` has an API | Check the `auth_env` variable is set **for the robot's user**, the URL is `https://`, and look in the log for `official API unavailable` |
| Operator completed but bot still sees `PENDING` | The bot is polling a different `request_id`, or a different database (`CAPTCHA_FW_DB_PATH` must match for the robot and the operator console) |
| `INVALID_INPUT: Path … outside the allowed directories` | Write `html_file` under the project, the robot's user profile, or the working directory |
| Garbled characters in PAD or Blue Prism | `run.bat` sets UTF-8 (`chcp 65001`). Read the result file as UTF-8 |
| `TIMEOUT` | Raise `timeout` (up to 3600) or switch to async polling (`wait_for_operator:false`) |
| Wrong type detected | Record feedback (`run.bat feedback …`), then `model train` → `approve` → `deploy` |
| `No production image model is deployed` | A named reviewer approves a candidate: `scripts\approve_model.bat <version> "Name"` (see `scripts\continuous_training.bat status`) |
| `promotion criteria not met` on approval | The candidate is not better than production on the fixed test set. Keep the current model, or approve with `--allow-regression` after a deliberate review |
| `LOCKED: another training controller is running` | Wait for it to finish, or if it crashed, run `python training\continuous_trainer.py unlock` |
| Training stops with `max_training_minutes reached` | Run the same command again: it resumes from the checkpoint. Raise the limit in `training_config.yaml` if needed |
| `only N MB free on disk` | Free disk space or lower `min_free_disk_mb`; the 10k dataset needs about 130 MB and each model about 2 MB |

Logs: `logs\captcha_framework.log`. Reports: `logs\reports\`. History DB: `database\captcha_events.db`.

## Project layout

```
captcha_rpa_framework/
├── main.py                CLI (run / operator / dataset / model / feedback / report / serve)
├── config.py              settings from CAPTCHA_FW_* environment variables
├── install.bat, run.bat   Windows installer and launcher
├── requirements.txt, sites.example.json, .env.example
├── core/        detector.py, classifier.py, resolution_engine.py, human_verification.py,
│                response_handler.py, framework.py (orchestrator), browser.py (Playwright), security.py
├── integrations/ uipath.py, automation_anywhere.py, blue_prism.py, power_automate.py, generic_rpa.py
├── learning/    analytics.py, dataset_manager.py, model_evaluation.py, performance_report.py
├── training/    generate_dataset.py, dataset_validator.py, dataset_manager.py, train_classifier.py,
│                evaluate_model.py, model_registry.py, continuous_trainer.py, training_report.py,
│                training_config.yaml, common.py
├── scripts/     train_10000.bat, continuous_training.bat, evaluate_model.bat, approve_model.bat,
│                rollback_model.bat, schedule_training.bat
├── datasets/    train/ validation/ test/ manifest.json (+ cache/ of extracted features)
├── database/    db_manager.py, schema.sql
├── api/         server.py (FastAPI + operator console)
├── models/      classifier_v1.0.0.json (signature model), image_registry.json,
│                checkpoints/<version>/, production/, archive/
├── reports/     audit/, training_history/, performance/
├── samples/     sample inputs and real outputs
├── test_data/   fixtures/ (static mock pages), test_responses.json, generated synthetic data
├── tests/       pytest suite
├── logs/  screenshots/
├── PROJECT_COMPLETION_REPORT.md
└── TRAINING_COMPLETION_REPORT.md
```

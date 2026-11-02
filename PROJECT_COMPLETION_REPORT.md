# Project Completion Report: Universal CAPTCHA Management Framework for RPA

**Date:** 2026-10-30  **Version:** 1.0.0  **Test result:** 175 / 175 passed (Windows 11, Python 3.14.7)

## 1. Summary
All eight phases are complete: architecture, detection, authorized resolution with human
verification, four RPA adapters plus a generic one, learning and analytics, BAT scripts and the
REST API, testing, and documentation. The framework was installed with `install.bat`, run from
the CLI, `run.bat`, the REST API (live under uvicorn) and a real browser (Microsoft Edge driven by
Playwright against a local mock website). Every defect found during testing was fixed (see §5).

## 2. Deliverables checklist

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Every required Python file | Done | Structure from the brief, plus `core/framework.py`, `core/browser.py`, `core/security.py` |
| 2 | BAT install and run scripts | Done | `install.bat` (6 steps plus a results table, `/nobrowser`, `/notest`), `run.bat` (file, stdin and pass-through modes; exit code = status) |
| 3 | Modules implemented | Done | See §3 |
| 4 | Dependencies installed and validated | Done | `install.bat` run: Python OK, venv OK, dependencies OK, verification OK, tests OK |
| 5 | Unit tests run | Done | 175 passed |
| 6 | Errors fixed | Done | §5 |
| 7 | Sample JSON input and output | Done | `samples/` (5 inputs; 7 outputs captured from real runs) |
| 8 | Integration instructions | Done | README: UiPath (3 options), A360, Blue Prism, PAD, Python/Selenium |
| 9 | README with install and troubleshooting | Done | `README.md` |
| 10 | Completion report | Done | This file |

## 3. What was built

**Detection and classification.** The detector reads metadata only: iframe and script URLs,
classes, ids, `data-*` attributes, field names and prompt text, plus live frame URLs from the
browser. It recognises reCAPTCHA (v2, v3/invisible, Enterprise, image and audio challenges),
hCaptcha, Turnstile and Cloudflare managed challenges, Arkose/FunCaptcha, GeeTest, AWS WAF, and
generic text, image-selection and audio challenges. Provider widgets take precedence over generic
hints. If a widget has already been completed, the detector sees that the response field is filled
without reading its value. Classification is a versioned, explainable JSON signature model, and
each response lists the signals that fired.

**Resolution.** A pluggable chain:
1. **Test mode** for mock pages or allow-listed test hosts only. Anything else returns `POLICY_BLOCKED`.
2. **Official API** (via `sites.json`). Returns `OFFICIAL_API_ROUTE` so the bot uses the site's API.
   Provider siteverify is available for sites marked as owned.
3. **Human-in-the-loop**: a verification queue, notification webhook, CLI, REST and web operator
   console, blocking or async polling, and automatic completion when the robot's browser shows the
   widget was completed.

**Integration.** One JSON contract (Pydantic) across the CLI, `run.bat`, REST and Python. 15
status codes map to process exit codes. A hard per-task timeout, atomic output files, and
platform-specific input coercion and output formats (compact JSON for UiPath, JSON or `k=v`
input for A360, flat `key=value` for Blue Prism, JSON for PAD).

**Learning.** SQLite history (metadata only), analytics, recurring-failure detection with
recommendations, synthetic HTML and image dataset generation (OpenCV), evaluation (accuracy,
P/R/F1, confusion matrix), perceptron-style weight training on synthetic fixtures and operator
type labels, a model registry with **mandatory named-human approval**, regression blocking,
deploy and rollback, and JSON plus HTML performance reports.

## 4. Test coverage (175 tests)
| Area | File | Highlights |
|---|---|---|
| Detection and classification | `test_detection.py` | All types, negatives, subtypes, frame-only detection, solved-widget handling without reading values, malformed HTML, model validation, deployed-model loading |
| Human verification | `test_human_verification.py` | Complete, reject, expire; blocking, async and cancel; timeout cancels the request; webhook HTTPS rule and failure tolerance; screenshot purge |
| Resolution and API handling | `test_resolution.py` | Test-mode policy, configurable responses, official API success, 401/429/network/HTTPS fallbacks, retry bounds, provider verify, custom resolver |
| JSON I/O and CLI | `test_json_io.py` | 11 invalid-input cases, BOM, spec shape, exit codes, redaction, stdin/file/inline, operator CLI flow, stdout carries only JSON |
| RPA integration | `test_integrations.py` | Adapter aliases, string coercion, A360 k=v, Blue Prism collection, PAD, Selenium-style usage |
| Timeouts and network | `test_timeout_network.py` | Hard timeout on a hung page, CDP refused, DNS failure, missing browser, path containment, history failure isolation |
| Database | `test_database.py` | Schema, cascade, single deployed model, SQL injection resistance, no content columns |
| Learning | `test_learning.py` | Datasets, determinism, OpenCV preprocessing, a weakened model repaired by training, approval rules, live model switch-over, analytics, reports, HTML escaping |
| REST API | `test_api.py` | Auth (key and loopback), operator flow, screenshot path safety, feedback, models, provider verify, console |
| Browser and install | `test_browser_and_install.py` | **Live Edge** against a local mock site (external traffic blocked), screenshots, unreachable pages; `run.bat` file, stdin and pass-through; installer content |

## 5. Issues found and fixed during testing
1. **pip failed in the project folder.** The project path is too deep for Windows' 260-character
   limit. `install.bat` now warns about long paths and supports `CAPTCHA_FW_VENV`; the README
   documents the fix.
2. **Disk filled during the Chromium download.** C: was nearly full. I removed my partial
   download, added an Edge fallback (`chrome → msedge → bundled Chromium`), and added `/nobrowser`.
3. **Secrets were not fully redacted.** In `Authorization: Bearer <token>`, the token leaked
   because two patterns overlapped. The redaction order is fixed and covered by a test.
4. **Race in blocking HITL.** The request could expire at the same moment the wait deadline hit,
   so callers got `EXPIRED` instead of `TIMEOUT`. A grace period now lets the waiter cancel cleanly.
5. **Noisy text-CAPTCHA signals** fired on provider field names such as `g-recaptcha-response`.
   The patterns now exclude provider names.
6. **Invalid input echoed a raw `task_id`.** It is now sanitized to `UNKNOWN`.
7. **`check_status` reported `captcha_detected:false`.** It now reports the verified type.
8. A CLI flag clash (`--verbose`) and relative `html_file` resolution from other working
   directories were also fixed.

## 6. External dependencies and their mocks
| Dependency | Needed for | Status / mock |
|---|---|---|
| Site's official API and token (`auth_env`) | Official API route | Configured per site; tested with an httpx `MockTransport` |
| Provider secret (`provider_secret_env`) | Siteverify for owned sites | Mocked siteverify endpoint in tests |
| Notification webhook (Teams/Slack/ITSM) | Operator push alerts | Optional; mocked. The CLI, console and REST work without it |
| Human operator | HITL completion | Simulated in tests with a background thread and auto-clearance |
| Robot browser with `--remote-debugging-port` | Live CDP attach | Fake sessions in tests, plus a live Edge run against the mock site |
| TLS certificate | HTTPS REST | Supported through environment variables; not provisioned here |
| Playwright Chromium | Bundled browser | Optional; Edge was used because of the disk constraint |

## 7. Deliberate scope decisions
* **No automated solving.** There is no OCR, audio transcription, image-grid answering, token
  farming or third-party solver integration. Tesseract and `pytesseract` were left out on purpose:
  the only CAPTCHA-related use for OCR would be reading challenge text, which is circumvention.
  OpenCV is used only to generate and preprocess *synthetic* test images.
* **Training is limited** to synthetic fixtures and operator labels of the challenge *type*.
  New signatures go through human code review; models are never auto-deployed.
* **Test mode** cannot be pointed at production hosts.

## 8. Known limitations and recommended next steps
* The signature model scores 100% on synthetic fixtures *by construction*. Real-world accuracy
  should be measured from operator feedback after a pilot (`run.bat report`).
* SQLite suits a single host. For many robots sharing one operator queue, point
  `CAPTCHA_FW_DB_PATH` to shared storage or swap `DatabaseManager` for a server database.
* The operator console is basic. Larger teams may prefer routing `HUMAN_VERIFICATION_REQUIRED`
  into UiPath Action Center or ServiceNow through the webhook.
* `fastapi.testclient` prints a Starlette deprecation warning about `httpx`. It is harmless for now.

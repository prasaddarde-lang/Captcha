"""Orchestrator shared by the CLI (main.py), the REST API and all RPA adapters.

    validate input -> load page -> detect/classify -> resolve (authorized) -> record history -> respond
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import httpx

from config import Settings, get_settings
from core.browser import BrowserSession, BrowserUnavailable
from core.classifier import CaptchaClassifier, Classification
from core.detector import CaptchaDetector, DetectionResult, PageSnapshot
from core.image_classifier import ImageCaptchaClassifier, ImagePrediction
from core.human_verification import Deadline, HumanVerificationManager
from core.resolution_engine import ResolutionContext, ResolutionEngine, ResolutionOutcome
from core.response_handler import Signal, Status, TaskRequest, TaskResponse, error_response
from core.security import host_of, safe_path, setup_logging
from database.db_manager import DatabaseManager

log = logging.getLogger("captcha_fw.framework")


class CaptchaFramework:
    def __init__(self, settings: Settings | None = None, db: DatabaseManager | None = None,
                 http_client: httpx.Client | None = None,
                 session_factory=BrowserSession):
        self.settings = settings or get_settings()
        setup_logging(self.settings.log_dir, self.settings.log_level)
        self.db = db or DatabaseManager(self.settings.db_path)
        self.hv = HumanVerificationManager(self.db, self.settings, http_client)
        self.engine = ResolutionEngine(self.settings, self.db, self.hv, http_client)
        self.http_client = http_client
        self.session_factory = session_factory
        self._classifier: CaptchaClassifier | None = None
        # Image-based type classifier: only the human-approved production model is ever loaded.
        self.image_classifier = ImageCaptchaClassifier(self.settings.models_dir, self.settings.image_min_confidence)

    @property
    def detector(self) -> CaptchaDetector:
        if self._classifier is None:
            self._classifier = CaptchaClassifier.load_active(self.db)
        return CaptchaDetector(self._classifier)

    def reload_model(self) -> None:
        self._classifier = None

    # ----------------------------------------------------------------- public
    def run(self, request: TaskRequest) -> TaskResponse:
        """Execute a task, enforcing the request timeout as a hard limit."""
        timeout = min(request.timeout, self.settings.max_timeout)
        result: dict = {}

        def target() -> None:
            try:
                result["response"] = self._execute(request, Deadline(timeout))
            except Exception as exc:  # noqa: BLE001 - convert everything into a structured response
                result["error"] = exc

        started = time.monotonic()
        worker = threading.Thread(target=target, daemon=True, name=f"task-{request.task_id}")
        worker.start()
        worker.join(timeout + 5)  # small grace period for clean-up inside the task
        if worker.is_alive():
            resp = error_response(request.task_id, Status.TIMEOUT, f"Task exceeded timeout of {timeout}s")
        elif "error" in result:
            resp = self._error_from_exception(request, result["error"])
        else:
            resp = result["response"]
        resp.duration_ms = int((time.monotonic() - started) * 1000)
        resp.task_id = request.task_id
        self._record(request, resp)
        return resp

    # --------------------------------------------------------------- internal
    def _execute(self, request: TaskRequest, deadline: Deadline) -> TaskResponse:
        if request.action in ("check_status", "cancel"):
            if request.action == "cancel":
                self.hv.cancel(request.request_id)  # type: ignore[arg-type]
            outcome = self.engine.check_status(request.request_id)  # type: ignore[arg-type]
            resp = self._response(request, None, outcome)
            vr = self.hv.get(request.request_id)  # type: ignore[arg-type]
            if vr:
                resp.captcha_detected, resp.captcha_type = True, vr["captcha_type"]
            return resp

        image = self._image_prediction(request)
        if request.html or request.html_file:
            snapshot = self._snapshot_from_input(request)
            return self._detect_and_resolve(request, snapshot, deadline, session=None, image=image)

        if not (request.cdp_endpoint or request.url):  # image-only request
            snapshot = PageSnapshot(html="", url=None, source="image_file")
            return self._detect_and_resolve(request, snapshot, deadline, session=None, image=image)

        # Live browser page (attach over CDP, or open the URL)
        with self.session_factory(url=request.url, cdp_endpoint=request.cdp_endpoint,
                                  browser=request.browser, headless=request.headless,
                                  timeout_s=max(1.0, min(deadline.remaining(), 60))) as session:
            shot_dir = self.settings.screenshot_dir if (request.screenshot and self.settings.screenshots_enabled) else None
            snapshot = session.snapshot(shot_dir, request.task_id)
            return self._detect_and_resolve(request, snapshot, deadline, session=session, image=image)

    def _allowed_roots(self) -> list[Path]:
        # RPA tools commonly write temp files under the user profile, so that is allowed too.
        return [self.settings.base_dir, self.settings.test_data_dir, self.settings.datasets_dir, Path.cwd(), Path.home()]

    def _resolve_input_path(self, value: str) -> Path:
        raw = Path(value)
        if not raw.is_absolute() and not raw.exists():
            raw = self.settings.base_dir / raw  # relative paths may be given from the project root
        return safe_path(raw, self._allowed_roots())

    def _image_prediction(self, request: TaskRequest) -> tuple[ImagePrediction, bool] | None:
        """Classify the challenge *type* from a cropped screenshot. Returns (prediction, synthetic)."""
        if not request.image_file:
            return None
        path = self._resolve_input_path(request.image_file)
        if not self.image_classifier.available:
            if request.html or request.html_file or request.cdp_endpoint or request.url:
                log.warning("image_file ignored: no production image model is deployed")
                return None
            raise RuntimeError("No production image model is deployed. A named reviewer approves a trained "
                               "candidate with scripts\\approve_model.bat <version> \"<name>\", or send html/url input.")
        pred = self.image_classifier.classify_file(path)
        synthetic_roots = [self.settings.test_data_dir, self.settings.datasets_dir]
        return pred, any(_is_under(path, r) for r in synthetic_roots)

    def _snapshot_from_input(self, request: TaskRequest) -> PageSnapshot:
        if request.html:
            return PageSnapshot(html=request.html, url=request.url, source="html")
        path = self._resolve_input_path(request.html_file)  # type: ignore[arg-type]
        if not path.exists():
            raise FileNotFoundError(f"html_file not found: {path.name}")
        html = path.read_text(encoding="utf-8", errors="replace")
        return PageSnapshot(html=html, url=request.url or path.as_uri(), source="html_file")

    def _detect_and_resolve(self, request: TaskRequest, snapshot: PageSnapshot,
                            deadline: Deadline, session, image=None) -> TaskResponse:
        detector = self.detector
        detection = detector.detect(snapshot)
        source = "metadata"
        pred = image[0] if image else None
        if pred is not None:
            if detection.detected:
                source = "metadata+image_model"  # page metadata is authoritative; image result is reported alongside
            elif pred.is_captcha:
                detection = self._detection_from_image(detection, pred, synthetic=image[1])
                source = "image_model"
            elif not snapshot.html:
                source = "image_model"
        extra = {"image": pred, "source": source}
        if not detection.detected:
            msg = "No CAPTCHA challenge detected"
            if pred is not None and pred.captcha_type and not pred.is_captcha:
                msg += (f" (image model suggested '{pred.category}' at {pred.confidence:.2f}, below the "
                        f"{self.image_classifier.min_confidence:.2f} threshold)")
            return self._response(request, detection, ResolutionOutcome(Status.NO_CAPTCHA, "none", msg), **extra)
        if detection.is_solved:
            return self._response(request, detection, ResolutionOutcome(
                Status.VERIFIED, "already_completed",
                "CAPTCHA widget present but already completed; workflow may continue"), **extra)

        ctx = ResolutionContext(request=request, detection=detection, deadline=deadline,
                                session=session, detector=detector)
        ctx.site_entry = self.engine.site_entry(request, ctx.host)
        if request.action == "detect":
            method = self.engine.planned_method(ctx)
            return self._response(request, detection, ResolutionOutcome(
                Status.CAPTCHA_DETECTED, method,
                f"{detection.classification.captcha_type} CAPTCHA detected; planned resolution: {method}"), **extra)
        return self._response(request, detection, self.engine.resolve(ctx), **extra)

    def _detection_from_image(self, base: DetectionResult, pred: ImagePrediction, synthetic: bool) -> DetectionResult:
        c = Classification(True, pred.captcha_type, pred.subtype, pred.confidence, {},
                           [{"captcha_type": pred.captcha_type, "signal": f"image_model:{pred.category}",
                             "weight": pred.confidence}], base.classification.model_version)
        if synthetic and base.features.mock_marker is None:
            base.features.mock_marker = "synthetic_image"  # allows test mode on our own generated images only
        return DetectionResult(c, base.features, base.snapshot)

    def _response(self, request: TaskRequest, detection: DetectionResult | None,
                  outcome: ResolutionOutcome, image: ImagePrediction | None = None,
                  source: str | None = None) -> TaskResponse:
        c = detection.classification if detection else None
        return TaskResponse(
            task_id=request.task_id,
            status=outcome.status,
            captcha_detected=bool(c and c.captcha_detected),
            captcha_type=c.captcha_type if c else None,
            captcha_subtype=c.subtype if c else None,
            confidence=c.confidence if c and c.captcha_detected else None,
            resolution_method=outcome.method,
            message=outcome.message,
            request_id=outcome.request_id,
            screenshot_path=outcome.screenshot_path or (detection.snapshot.screenshot_path if detection else None),
            next_action=outcome.next_action,
            official_api=outcome.official_api,
            signals=[Signal(**s) for s in c.signals] if c else [],
            model_version=c.model_version if c else None,
            detection_source=source if detection else None,
            image_model_version=image.model_version if image else None,
            image_category=image.category if image else None,
            image_confidence=image.confidence if image else None,
        )

    def _error_from_exception(self, request: TaskRequest, exc: Exception) -> TaskResponse:
        if isinstance(exc, (httpx.HTTPError, ConnectionError)) or "net::" in str(exc) or "ECONNREFUSED" in str(exc):
            status, msg = Status.NETWORK_ERROR, f"Network error: {type(exc).__name__}: {exc}"
        elif isinstance(exc, BrowserUnavailable):
            status, msg = Status.ERROR, str(exc)
        elif isinstance(exc, (FileNotFoundError, ValueError, PermissionError)):
            status, msg = Status.INVALID_INPUT, str(exc)
        elif type(exc).__name__ == "TimeoutError":
            status, msg = Status.TIMEOUT, f"Browser operation timed out: {exc}"
        else:
            status, msg = Status.ERROR, f"{type(exc).__name__}: {exc}"
        log.error("task %s failed: %s", request.task_id, msg, exc_info=exc)
        return error_response(request.task_id, status, msg)

    def _record(self, request: TaskRequest, resp: TaskResponse) -> None:
        try:
            self.db.record_execution(
                task_id=request.task_id, platform=request.platform, browser=request.browser,
                action=request.action, mode=request.mode, source_host=host_of(request.url),
                status=resp.status.value, captcha_detected=resp.captcha_detected,
                captcha_type=resp.captcha_type, confidence=resp.confidence,
                resolution_method=resp.resolution_method, model_version=resp.model_version,
                duration_ms=resp.duration_ms,
                error_code=resp.status.value if resp.exit_code >= 30 else None,
                message=resp.message[:500],
                signals=[s.model_dump() for s in resp.signals],
                detection_source=resp.detection_source, image_model_version=resp.image_model_version,
                image_category=resp.image_category,
            )
        except Exception as exc:  # history must never break the RPA workflow
            log.error("could not record execution: %s", exc)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

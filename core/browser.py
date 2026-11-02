"""Authorized browser interaction via Playwright (optional dependency).

Two ways to get a page:
  * ``cdp_endpoint`` - attach to the browser the RPA robot already drives
    (start Chrome/Edge with ``--remote-debugging-port=9222``). The framework
    only *reads* page metadata and takes screenshots; it never interacts with
    the CAPTCHA widget.
  * ``url`` - open the page in a Playwright-managed browser (detection,
    testing, or a headed window an operator can use).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from core.detector import RESPONSE_FIELDS, PageSnapshot

log = logging.getLogger("captcha_fw.browser")

_RESPONSE_JS = """
(fields) => {
  const out = {};
  for (const [name, provider] of Object.entries(fields)) {
    const els = document.querySelectorAll(`[name="${name}"]`);
    let filled = false;
    els.forEach(e => { if ((e.value || '').length > 0) filled = true; });
    if (els.length) out[provider] = filled;
  }
  return out;
}
"""


class BrowserUnavailable(RuntimeError):
    pass


def _local_only(route) -> None:
    from urllib.parse import urlparse
    host = urlparse(route.request.url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1") or route.request.url.startswith(("data:", "about:")):
        route.continue_()
    else:
        route.abort()


class BrowserSession:
    """Context manager wrapping a Playwright page."""

    def __init__(self, url: str | None = None, cdp_endpoint: str | None = None,
                 browser: str = "chrome", headless: bool = True, timeout_s: float = 30,
                 block_external: bool = False):
        self.url = url
        self.cdp_endpoint = cdp_endpoint
        self.browser_name = (browser or "chrome").lower()
        self.headless = headless
        self.timeout_ms = int(timeout_s * 1000)
        self._pw = None
        self._browser = None
        self.page = None
        self._owns_browser = False
        # For mock sites: abort every request that is not to a loopback host.
        self.block_external = block_external

    def __enter__(self) -> "BrowserSession":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on install
            raise BrowserUnavailable("Playwright is not installed. Run install.bat") from exc
        self._pw = sync_playwright().start()
        try:
            if self.cdp_endpoint:
                self._attach()
            else:
                self._launch()
        except Exception:
            self.close()
            raise
        return self

    def _attach(self) -> None:
        self._browser = self._pw.chromium.connect_over_cdp(self.cdp_endpoint, timeout=self.timeout_ms)
        pages = [p for ctx in self._browser.contexts for p in ctx.pages]
        if not pages:
            raise BrowserUnavailable("No open pages found on the CDP endpoint")
        if self.url:
            match = [p for p in pages if p.url.startswith(self.url)]
            self.page = match[0] if match else pages[-1]
        else:
            self.page = pages[-1]
        log.info("attached to existing browser page via CDP")

    def _launch(self) -> None:
        engine = self._pw.chromium
        kwargs: dict = {"headless": self.headless}
        if self.browser_name in ("firefox",):
            engine = self._pw.firefox
        elif self.browser_name in ("webkit", "safari"):
            engine = self._pw.webkit
        elif self.browser_name in ("edge", "msedge"):
            kwargs["channel"] = "msedge"
        elif self.browser_name == "chrome":
            kwargs["channel"] = "chrome"
        # Try the requested channel, then Edge (preinstalled on Windows), then bundled Chromium.
        attempts = [kwargs]
        if engine is self._pw.chromium:
            attempts += [{**kwargs, "channel": "msedge"}, {"headless": self.headless}]
        last_exc: Exception | None = None
        for opts in attempts:
            try:
                self._browser = engine.launch(**opts)
                break
            except Exception as exc:
                last_exc = exc
        else:
            raise BrowserUnavailable(f"No browser could be launched: {last_exc}")
        self._owns_browser = True
        self.page = self._browser.new_page()
        if self.block_external:
            self.page.route("**/*", _local_only)
        if self.url:
            self.page.goto(self.url, timeout=self.timeout_ms, wait_until="domcontentloaded")

    def snapshot(self, screenshot_dir: Path | None = None, task_id: str = "task") -> PageSnapshot:
        page = self.page
        html = page.content()
        frames = [f.url for f in page.frames if f.url and f.url != "about:blank"]
        try:
            filled = page.evaluate(_RESPONSE_JS, RESPONSE_FIELDS)
        except Exception:
            filled = {}
        shot = self.screenshot(screenshot_dir, task_id) if screenshot_dir else None
        return PageSnapshot(html=html, url=page.url, frame_urls=frames, response_filled=filled,
                            screenshot_path=shot, source="cdp" if self.cdp_endpoint else "url")

    def screenshot(self, directory: Path, task_id: str) -> str | None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{task_id}_{datetime.now():%Y%m%d_%H%M%S}.png"
            self.page.screenshot(path=str(path), full_page=False)
            return str(path)
        except Exception as exc:
            log.warning("screenshot failed: %s", exc)
            return None

    def bring_to_front(self) -> None:
        try:
            self.page.bring_to_front()
        except Exception:
            pass

    def wait(self, seconds: float) -> None:
        if self.page is not None:
            self.page.wait_for_timeout(int(seconds * 1000))
        else:
            time.sleep(seconds)

    def close(self) -> None:
        try:
            if self._browser is not None and self._owns_browser:
                self._browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()  # for CDP sessions this only disconnects; the RPA browser stays open
            self._pw = None

    def __exit__(self, *exc) -> None:
        self.close()

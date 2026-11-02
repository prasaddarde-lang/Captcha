"""Security helpers: input sanitisation, secret redaction and secure logging."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
from pathlib import Path
from urllib.parse import urlparse

# Values that must never reach a log file.
_REDACT_PATTERNS = [
    # bearer tokens (first, so "Authorization: Bearer <token>" redacts the token itself)
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]+)"),
    # key=value / "key": "value" style secrets
    re.compile(r'(?i)("?(?:api[_-]?key|secret|password|passwd|token|'
               r'g-recaptcha-response|h-captcha-response|cf-turnstile-response)"?\s*[:=]\s*"?)([^"\s,&}]+)'),
]
_LONG_TOKEN = re.compile(r"\b[A-Za-z0-9_\-]{60,}\b")

TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
ALLOWED_URL_SCHEMES = {"http", "https", "file"}


def redact(text: str) -> str:
    if not text:
        return text
    for pat in _REDACT_PATTERNS:
        text = pat.sub(lambda m: m.group(1) + "***REDACTED***", text)
    return _LONG_TOKEN.sub("***REDACTED***", text)


def sanitize_text(value: str, max_len: int = 500) -> str:
    """Strip control characters and truncate free-text fields."""
    value = "".join(ch for ch in value if ch.isprintable() or ch in "\t ")
    return value[:max_len]


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"URL scheme '{parsed.scheme}' is not allowed")
    if parsed.scheme in {"http", "https"} and not parsed.hostname:
        raise ValueError("URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("Credentials embedded in URLs are not allowed")
    return url


def host_of(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return "file"
    return (parsed.hostname or "").lower() or None


def safe_path(path: str | Path, allowed_roots: list[Path]) -> Path:
    """Resolve *path* and make sure it lives under one of *allowed_roots* (no traversal)."""
    p = Path(path).expanduser().resolve()
    for root in allowed_roots:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise ValueError(f"Path '{path}' is outside the allowed directories")


def get_secret(env_name: str | None) -> str | None:
    """Secrets are only ever read from environment variables."""
    if not env_name:
        return None
    value = os.environ.get(env_name)
    return value or None


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        record.msg = redact(msg)
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("task_id", "status", "captcha_type"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def setup_logging(log_dir: Path, level: str = "INFO", console: bool = False) -> logging.Logger:
    """Configure the framework logger. Logs go to a rotating JSON-lines file; stdout is
    reserved for the JSON result consumed by the RPA tool, so console logging goes to stderr."""
    global _configured
    logger = logging.getLogger("captcha_fw")
    if _configured:
        return logger
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "captcha_framework.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(JsonFormatter())
    fh.addFilter(RedactingFilter())
    logger.addHandler(fh)
    if console:
        sh = logging.StreamHandler()  # stderr
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        sh.addFilter(RedactingFilter())
        logger.addHandler(sh)
    _configured = True
    return logger

"""Central configuration.

All values can be overridden with environment variables prefixed ``CAPTCHA_FW_``.
Secrets (API keys, provider secrets) are NEVER stored here: config only stores
the *name* of the environment variable that holds a secret.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"CAPTCHA_FW_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    base_dir: Path = BASE_DIR
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(BASE_DIR / "database" / "captcha_events.db"))))
    log_dir: Path = field(default_factory=lambda: Path(_env("LOG_DIR", str(BASE_DIR / "logs"))))
    screenshot_dir: Path = field(default_factory=lambda: Path(_env("SCREENSHOT_DIR", str(BASE_DIR / "screenshots"))))
    test_data_dir: Path = field(default_factory=lambda: Path(_env("TEST_DATA_DIR", str(BASE_DIR / "test_data"))))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", str(BASE_DIR / "models"))))
    datasets_dir: Path = field(default_factory=lambda: Path(_env("DATASETS_DIR", str(BASE_DIR / "datasets"))))
    sites_file: Path = field(default_factory=lambda: Path(_env("SITES_FILE", str(BASE_DIR / "sites.json"))))

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    default_timeout: int = field(default_factory=lambda: _env_int("DEFAULT_TIMEOUT", 120))
    max_timeout: int = field(default_factory=lambda: _env_int("MAX_TIMEOUT", 3600))
    operator_poll_interval: float = field(default_factory=lambda: float(_env("OPERATOR_POLL_INTERVAL", "2")))
    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "15")))
    http_retries: int = field(default_factory=lambda: _env_int("HTTP_RETRIES", 2))

    # Screenshots of challenges are only captured when explicitly allowed.
    screenshots_enabled: bool = field(default_factory=lambda: _env_bool("SCREENSHOTS_ENABLED", True))
    # Retention for screenshots (hours). Sensitive challenge content is not kept longer.
    screenshot_retention_hours: int = field(default_factory=lambda: _env_int("SCREENSHOT_RETENTION_HOURS", 24))
    # Optional deployment pin for the signature model, for hosts that start with an empty registry
    # (e.g. a cloud service whose disk is reset on redeploy). Both values are required: the version
    # must already have been approved by the named human, whose name is recorded with the pin.
    signature_model: str | None = field(default_factory=lambda: os.environ.get("CAPTCHA_FW_SIGNATURE_MODEL") or None)
    signature_model_approved_by: str | None = field(
        default_factory=lambda: os.environ.get("CAPTCHA_FW_SIGNATURE_MODEL_APPROVED_BY") or None)
    # Image model: below this confidence it never declares a CAPTCHA on its own.
    image_min_confidence: float = field(default_factory=lambda: float(_env("IMAGE_MIN_CONFIDENCE", "0.80")))
    # Never send traffic to non-HTTPS endpoints for official APIs unless explicitly allowed (tests).
    require_https: bool = field(default_factory=lambda: _env_bool("REQUIRE_HTTPS", True))

    # Hosts on which the "test" mode (mock challenges, configurable responses) may run.
    test_mode_hosts: tuple[str, ...] = field(default_factory=lambda: tuple(
        h.strip() for h in _env("TEST_MODE_HOSTS", "localhost,127.0.0.1,::1,mock.test,example.test").split(",") if h.strip()
    ))

    # REST API
    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8765))
    api_key_env: str = "CAPTCHA_FW_API_KEY"          # name of env var holding the REST API key
    ssl_certfile: str | None = field(default_factory=lambda: os.environ.get("CAPTCHA_FW_SSL_CERTFILE"))
    ssl_keyfile: str | None = field(default_factory=lambda: os.environ.get("CAPTCHA_FW_SSL_KEYFILE"))

    def ensure_dirs(self) -> None:
        for p in (self.log_dir, self.screenshot_dir, self.test_data_dir, self.models_dir, self.db_path.parent):
            p.mkdir(parents=True, exist_ok=True)

    def load_sites(self) -> dict:
        """Load the authorized-site registry (see sites.example.json)."""
        if not self.sites_file.exists():
            return {}
        with open(self.sites_file, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("sites", {})


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> None:
    """Re-read settings from the environment (used by tests)."""
    global _settings
    _settings = None

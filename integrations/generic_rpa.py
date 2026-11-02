"""Generic RPA adapter - the common interface every platform adapter builds on.

Any tool that can run a command line, a Python function or an HTTP request can
use this:

    python main.py run --input-file task.json --output-file result.json
    run.bat task.json result.json
    from integrations.generic_rpa import handle_json; handle_json('{"task_id": ...}')
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.response_handler import Status, TaskResponse, error_response, parse_request

_framework = None


def get_framework():
    global _framework
    if _framework is None:
        from core.framework import CaptchaFramework
        _framework = CaptchaFramework()
    return _framework


def set_framework(fw) -> None:
    """Inject a framework instance (tests, custom wiring)."""
    global _framework
    _framework = fw


class GenericRPAAdapter:
    platform = "generic"
    aliases: tuple[str, ...] = ()

    def normalize_input(self, data: dict[str, Any]) -> dict[str, Any]:
        """Hook for platform-specific input conventions."""
        data = dict(data)
        data.setdefault("platform", self.platform)
        return data

    def format_output(self, response: TaskResponse) -> str:
        """Hook for platform-specific output conventions. Default: pretty JSON."""
        return response.to_json()

    def handle(self, raw: str | dict) -> TaskResponse:
        task_id = None
        try:
            if isinstance(raw, str):
                raw = raw.strip().lstrip("﻿")
                data = json.loads(raw) if raw else {}
            else:
                data = raw
            if isinstance(data, dict):
                task_id = data.get("task_id") if isinstance(data.get("task_id"), str) else None
                data = self.normalize_input(data)
            request = parse_request(data)
        except json.JSONDecodeError as exc:
            return error_response(task_id, Status.INVALID_INPUT, f"Input is not valid JSON: {exc.msg}")
        except ValueError as exc:
            return error_response(task_id, Status.INVALID_INPUT, str(exc))
        return get_framework().run(request)

    def handle_json(self, raw: str) -> str:
        return self.format_output(self.handle(raw))

    def handle_file(self, input_path: str | Path, output_path: str | Path | None = None) -> TaskResponse:
        try:
            raw = Path(input_path).read_text(encoding="utf-8-sig")
        except OSError as exc:
            return error_response(None, Status.INVALID_INPUT, f"Cannot read input file: {exc.strerror}")
        resp = self.handle(raw)
        if output_path:
            write_output(output_path, self.format_output(resp))
        return resp


def write_output(path: str | Path, text: str) -> None:
    """Atomic write so an RPA tool polling for the file never reads a partial result."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


_BOOL_KEYS = ("screenshot", "headless", "wait_for_operator")


def coerce_types(data: dict[str, Any]) -> dict[str, Any]:
    """Convert text booleans / numbers that RPA tools commonly send."""
    for key in _BOOL_KEYS:
        if isinstance(data.get(key), str):
            data[key] = data[key].strip().lower() in ("true", "1", "yes")
    if isinstance(data.get("timeout"), str) and data["timeout"].strip().isdigit():
        data["timeout"] = int(data["timeout"])
    return data


def flatten(response: TaskResponse) -> dict[str, str]:
    """Flat string key/value view (no nested objects) for tools with weak JSON support."""
    d = response.model_dump(mode="json", exclude={"signals"})
    out: dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}_{k2}"] = "" if v2 is None else str(v2)
        else:
            out[k] = "" if v is None else (str(v).lower() if isinstance(v, bool) else str(v))
    out["signal_count"] = str(len(response.signals))
    return out


_ADAPTERS: dict[str, type[GenericRPAAdapter]] = {}


def register(cls: type[GenericRPAAdapter]) -> type[GenericRPAAdapter]:
    _ADAPTERS[cls.platform] = cls
    for a in cls.aliases:
        _ADAPTERS[a] = cls
    return cls


register(GenericRPAAdapter)


def get_adapter(platform: str | None) -> GenericRPAAdapter:
    # Import platform modules so they self-register.
    from integrations import automation_anywhere, blue_prism, power_automate, uipath  # noqa: F401
    key = (platform or "generic").strip().lower().replace(" ", "_")
    return _ADAPTERS.get(key, GenericRPAAdapter)()


def handle_json(raw: str, platform: str | None = None) -> str:
    """Single-function entry point: JSON string in, JSON string out."""
    return get_adapter(platform).handle_json(raw)

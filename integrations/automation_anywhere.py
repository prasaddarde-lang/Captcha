"""Automation Anywhere (A360) integration.

Python Script package:
    Python script: Open             -> file = <project>\\integrations\\automation_anywhere.py
    Python script: Execute function -> Function name = "aa_handle",
                                       Input = $jsonInput$ (String), Output = $jsonResult$
    JSON package (Start session / Get node value) to read "status", "request_id", ...

Alternatively use "Open program/file" with run.bat and an output file.
A360 passes a single string argument, so this adapter also accepts a simple
"key=value;key=value" string for bots that do not build JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.response_handler import TaskResponse  # noqa: E402
from integrations.generic_rpa import GenericRPAAdapter, coerce_types, register  # noqa: E402


def parse_kv(text: str) -> dict:
    data: dict = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            data[k.strip()] = v.strip()
    return coerce_types(data)


@register
class AutomationAnywhereAdapter(GenericRPAAdapter):
    platform = "automation_anywhere"
    aliases = ("aa", "a360", "automationanywhere")

    def normalize_input(self, data):
        data = super().normalize_input(data)
        data["platform"] = "AutomationAnywhere"
        return coerce_types(data)

    def handle(self, raw):
        if isinstance(raw, str) and raw.strip() and not raw.strip().lstrip("﻿").startswith("{"):
            raw = parse_kv(raw)
        return super().handle(raw)

    def format_output(self, response: TaskResponse) -> str:
        return response.to_json(indent=None, include_signals=False)


def aa_handle(arg: str) -> str:
    """Entry point for A360 'Python script: Execute function'."""
    return AutomationAnywhereAdapter().handle_json(arg)


def aa_status(arg: str) -> str:
    """Convenience: return only the status string (for simple If conditions)."""
    return json.loads(aa_handle(arg))["status"]

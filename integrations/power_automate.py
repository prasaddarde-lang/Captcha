"""Microsoft Power Automate Desktop integration.

Option 1 - "Run DOS command":
    Command: "<project>\\run.bat" "%TaskJsonFile%" "%ResultJsonFile%" powerautomate
    Then:    Read text from file -> Convert JSON to custom object -> %Result['status']%
             (the process exit code is the status code; stderr carries diagnostics)
Option 2 - PAD's built-in "Run Python script" uses IronPython, which cannot load this
    framework (CPython packages) - call the venv Python via "Run DOS command" instead.
Option 3 - "Invoke web service" against the REST API: POST /v1/tasks.
Cloud flows can reach the REST API through an on-premises data gateway / custom connector.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.response_handler import TaskResponse  # noqa: E402
from integrations.generic_rpa import GenericRPAAdapter, coerce_types, register  # noqa: E402


@register
class PowerAutomateAdapter(GenericRPAAdapter):
    platform = "power_automate"
    aliases = ("powerautomate", "pad", "power_automate_desktop")

    def normalize_input(self, data):
        data = super().normalize_input(data)
        data["platform"] = "PowerAutomateDesktop"
        # Custom objects built from text variables serialise numbers/booleans as strings.
        return coerce_types(data)

    def format_output(self, response: TaskResponse) -> str:
        return response.to_json(indent=2)


def pad_handle(json_input: str) -> str:
    return PowerAutomateAdapter().handle_json(json_input)

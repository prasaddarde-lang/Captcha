"""UiPath integration.

Option 1 - Python Activities package (UiPath.Python.Activities):
    Python Scope (Path = <project>\\.venv\\Scripts, Target = x64)
      Load Python Script   -> File = <project>\\integrations\\uipath.py
      Invoke Python Method -> Name = "handle", Input parameters = {jsonString}
      Get Python Object    -> TypeArgument String -> Deserialize JSON

Option 2 - Start Process / Invoke Process:
    run.bat "C:\\path\\task.json" "C:\\path\\result.json" uipath   (exit code = status code)

Option 3 - HTTP Request activity against the REST API (POST /v1/tasks).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # allow loading this file directly from UiPath
    sys.path.insert(0, str(_ROOT))

from core.response_handler import TaskResponse  # noqa: E402
from integrations.generic_rpa import GenericRPAAdapter, coerce_types, register  # noqa: E402


@register
class UiPathAdapter(GenericRPAAdapter):
    platform = "uipath"
    aliases = ("uipath_studio", "uipath_robot")

    def normalize_input(self, data):
        data = super().normalize_input(data)
        data["platform"] = "UiPath"
        # Dictionaries built in VB/C# workflows often arrive with string booleans / numbers.
        return coerce_types(data)

    def format_output(self, response: TaskResponse) -> str:
        # Compact JSON parses cleanly with "Deserialize JSON" into a JObject.
        return response.to_json(indent=None)


def handle(json_input: str) -> str:
    """Entry point for UiPath 'Invoke Python Method'."""
    return UiPathAdapter().handle_json(json_input)

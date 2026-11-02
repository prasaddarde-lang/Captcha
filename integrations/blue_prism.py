"""Blue Prism integration.

Blue Prism works best with flat data, so this adapter emits ``key=value`` lines
(or a flat JSON object) that map directly to a single-row Collection.

Option 1 - Utility - Environment :: Start Process (wait) + read the output file:
    run.bat "<task.json>" "<result.txt>" blueprism
    Utility - File Management :: Read All Text From File -> split lines on the first '='
Option 2 - Web API (Blue Prism 6.4+): POST http://127.0.0.1:8765/v1/tasks and map
    the JSON response with the JSON VBO (JSON to Collection).
Option 3 - Code stage (C#/VB) starting run.bat via System.Diagnostics.Process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.response_handler import TaskResponse  # noqa: E402
from integrations.generic_rpa import GenericRPAAdapter, coerce_types, flatten, register  # noqa: E402


@register
class BluePrismAdapter(GenericRPAAdapter):
    platform = "blue_prism"
    aliases = ("blueprism", "bp")

    def __init__(self, output_format: str = "kv"):
        self.output_format = output_format

    def normalize_input(self, data):
        data = super().normalize_input(data)
        data["platform"] = "BluePrism"
        # Collections frequently carry "True"/"False" text flags and numeric text.
        return coerce_types(data)

    def format_output(self, response: TaskResponse) -> str:
        flat = flatten(response)
        if self.output_format == "json":
            return json.dumps(flat, indent=2)
        return "\n".join(f"{k}={v.replace(chr(13), ' ').replace(chr(10), ' ')}" for k, v in flat.items())


def to_collection(response: TaskResponse) -> list[dict[str, str]]:
    """Single-row collection representation."""
    return [flatten(response)]

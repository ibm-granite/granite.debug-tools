from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from runtimes_validator.domain.models import Report
from runtimes_validator.reporting.base import AbstractReporter


class JsonReporter(AbstractReporter):
    """Writes the validation report as a JSON file."""

    def __init__(self, output_dir: str | Path = ".") -> None:
        self._output_dir = Path(output_dir)

    def report(self, report: Report) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"report_{report.engine_info.engine_id}_{report.model}_{report.timestamp}.json"
        # Sanitize filename
        filename = filename.replace("/", "_").replace(":", "-").replace(" ", "_")
        path = self._output_dir / filename

        data = asdict(report)
        # Add computed fields
        data["all_passed"] = report.all_passed
        data["skipped_count"] = sum(1 for r in report.results if r.skipped)
        for i, result in enumerate(report.results):
            data["results"][i]["passed"] = result.passed
            data["results"][i]["status"] = result.status

        path.write_text(json.dumps(data, indent=2, default=str))

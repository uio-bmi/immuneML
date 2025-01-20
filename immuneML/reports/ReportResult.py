from dataclasses import dataclass, field
from typing import List

from immuneML.reports.ReportOutput import ReportOutput


@dataclass
class ReportResult:
    name: str = None
    info: str = field(default_factory=lambda: '')  # optional extra info about this report to display to the user
    output_figures: List[ReportOutput] = field(default_factory=lambda: [])
    output_tables: List[ReportOutput] = field(default_factory=lambda: [])
    output_text: List[ReportOutput] = field(default_factory=lambda: [])


from dataclasses import dataclass
from typing import List

from source.reports.ReportOutput import ReportOutput


@dataclass
class ReportResult:
    name: str = None
    output_figures: List[ReportOutput] = None
    output_tables: List[ReportOutput] = None
    output_text: List[ReportOutput] = None
    other_output: List[ReportOutput] = None

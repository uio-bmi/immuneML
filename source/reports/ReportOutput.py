from dataclasses import dataclass


@dataclass
class ReportOutput:
    path: str
    name: str = None

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReportOutput:
    def __init__(self, path: Path, name: str = None):
        self.path = Path(path)
        self.name = name

    def __repr__(self):
        return f"ReportOutput(name={self.name}, path={self.path})"

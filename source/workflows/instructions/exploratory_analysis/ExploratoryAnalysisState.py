from dataclasses import dataclass


@dataclass
class ExploratoryAnalysisState:
    exploratory_analysis_units: dict
    result_path: str = None
    name: str = None


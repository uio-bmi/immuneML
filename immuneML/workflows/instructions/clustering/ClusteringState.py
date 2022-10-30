from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClusteringState:
    clustering_units: dict
    result_path: Path = None
    name: str = None
    clustering_scores: dict = None


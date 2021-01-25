from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.simulation.Simulation import Simulation


@dataclass
class SimulationState:
    signals: list
    simulation: Simulation
    dataset: Dataset
    resulting_dataset: Dataset = None
    result_path: Path = None
    name: str = None

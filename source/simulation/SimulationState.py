from dataclasses import dataclass

from source.data_model.dataset.Dataset import Dataset
from source.simulation.Simulation import Simulation


@dataclass
class SimulationState:
    signals: list
    simulation: Simulation
    dataset: Dataset
    resulting_dataset: Dataset = None
    result_path: str = None
    batch_size: int = 1
    name: str = None

from dataclasses import dataclass

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.simulation.Simulation import Simulation


@dataclass
class SimulationState:
    signals: list
    simulation: Simulation
    dataset: RepertoireDataset
    resulting_dataset: RepertoireDataset = None
    path: str = None
    result_path: str = None
    batch_size: int = 1
    name: str = None

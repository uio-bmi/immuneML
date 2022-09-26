from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.implants.Signal import Signal


@dataclass
class LIgOSimulationState:
    simulation: Simulation = None
    result_path: Path = None
    signals: List[Signal] = None
    name: str = None
    dataset: Dataset = None
    formats: list = None
    paths: list = None
    store_signal_in_receptors: bool = None
    sequence_batch_size: int = None
    max_iterations: int = None
    number_of_processes: int = None


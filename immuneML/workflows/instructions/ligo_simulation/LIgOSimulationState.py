from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationStrategy import SimulationStrategy
from immuneML.simulation.implants.Signal import Signal


@dataclass
class LIgOSimulationState:
    is_repertoire: bool = True
    paired: bool = False
    sequence_type: SequenceType = None
    use_generation_probabilities: bool = False
    simulation_strategy: SimulationStrategy = None
    simulation: Simulation = None
    result_path: Path = None
    signals: List[Signal] = None
    name: str = None
    dataset: Dataset = None
    formats: list = None
    paths: list = None
    store_signal_in_receptors: bool = None
    sequence_batch_size: int = None


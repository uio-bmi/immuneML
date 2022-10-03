from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.Signal import Signal


@dataclass
class LigoImplanter:
    sim_item: LIgOSimulationItem = None
    sequence_type: SequenceType = None
    all_signals: List[Signal] = None
    sequence_batch_size: int = None
    seed: int = None
    export_p_gens: bool = None

    def make_repertoires(self, path: Path):
        pass

    def make_receptors(self, path: Path):
        raise NotImplementedError

    def make_sequences(self, path: Path):
        raise NotImplementedError

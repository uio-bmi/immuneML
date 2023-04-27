from dataclasses import dataclass
from typing import List

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy


@dataclass
class SimConfig:
    sim_items: List[SimConfigItem] = None
    identifier: str = None
    is_repertoire: bool = None
    paired: bool = None
    sequence_type: SequenceType = None
    simulation_strategy: SimulationStrategy = None
    p_gen_bin_count: int = None
    keep_p_gen_dist: bool = None
    remove_seqs_with_signals: bool = None

    def __str__(self):
        return ",\n".join(str(simulation_item) for simulation_item in self.sim_items)

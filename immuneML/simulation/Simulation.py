from dataclasses import dataclass

from immuneML.environment.SequenceType import SequenceType


@dataclass
class Simulation:
    sim_items: list = None
    identifier: str = None
    is_repertoire: bool = None
    paired: bool = None
    sequence_type: SequenceType = None
    simulation_strategy: str = None
    p_gen_bin_count: int = None
    keep_p_gen_dist: bool = None
    remove_seqs_with_signals: bool = None

    def __str__(self):
        return ",\n".join(str(simulation_item) for simulation_item in self.sim_items)

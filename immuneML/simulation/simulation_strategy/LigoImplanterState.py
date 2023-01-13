from dataclasses import dataclass
from typing import List

from immuneML.simulation.LIgOSimulationItem import SimConfigItem

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Signal import Signal


@dataclass
class LigoImplanterState:
    sim_item: SimConfigItem
    sequence_type: SequenceType
    all_signals: List[Signal]
    sequence_batch_size: int
    seed: int
    export_p_gens: bool
    keep_p_gen_dist: bool
    remove_seqs_with_signals: bool
    max_iterations: int
    p_gen_bin_count: int


    # not implemented yet
    keep_low_p_gen_proba: float = None

    # defined at runtime
    target_p_gen_histogram = None
    sequence_paths: dict = None
    p_gen_bins = None

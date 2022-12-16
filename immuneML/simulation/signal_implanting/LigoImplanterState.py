from dataclasses import dataclass
from typing import List

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.implants.Signal import Signal


@dataclass
class LigoImplanterState:
    sim_item: LIgOSimulationItem = None
    sequence_type: SequenceType = None
    all_signals: List[Signal] = None
    sequence_batch_size: int = None
    seed: int = None
    export_p_gens: bool = None
    keep_p_gen_dist: bool = None
    remove_seqs_with_signals: bool = None
    max_iterations: int = None
    p_gen_threshold: float = None
    target_p_gen_histogram = None
    current_p_gen_histogram = None
    p_gen_bin_count: int = None
    keep_low_p_gen_proba: float = None
    sequence_paths: dict = None
    p_gen_bins = None

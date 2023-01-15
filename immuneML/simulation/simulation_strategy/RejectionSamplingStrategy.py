from typing import List

from bionumpy.bnpdataclass import BNPDataClass

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy
from immuneML.simulation.util.util import filter_out_illegal_sequences


class RejectionSamplingStrategy(SimulationStrategy):

    MAX_SIGNALS_PER_SEQUENCE = 1
    MAX_MOTIF_POSITION_LENGTH = 10

    def process_sequences(self, sequences: BNPDataClass, seqs_per_signal_count: dict, use_p_gens: bool, sequence_type: SequenceType,
                          sim_item: SimConfigItem, all_signals: List[Signal], remove_positives_first: bool) -> BNPDataClass:
        filtered_sequences = filter_out_illegal_sequences(sequences, sim_item, all_signals, 1)
        return filtered_sequences

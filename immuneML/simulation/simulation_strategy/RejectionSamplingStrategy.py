from bionumpy.bnpdataclass import BNPDataClass

from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy


class RejectionSamplingStrategy(SimulationStrategy):

    MAX_SIGNALS_PER_SEQUENCE = 1
    MAX_MOTIF_POSITION_LENGTH = 10

    def process_sequences(self, sequences: BNPDataClass, seqs_per_signal_count: dict, max_signals: int, use_p_gens: bool) -> BNPDataClass:
        return sequences

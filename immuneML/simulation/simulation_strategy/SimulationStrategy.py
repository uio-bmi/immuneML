import abc

from bionumpy.bnpdataclass import BNPDataClass


class SimulationStrategy:

    @abc.abstractmethod
    def process_sequences(self, sequences: BNPDataClass, seqs_per_signal_count: dict, max_signals: int, use_p_gens: bool) -> BNPDataClass:
        pass

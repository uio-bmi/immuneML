# quality: gold

from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.simulation.implants.Motif import Motif
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class Signal:
    """
    Class representing the signal that will be implanted;
    Used in simulation setting;
    It contains information about:
        - motifs that can be instantiated and implanted into a receptor_sequence / repertoire
        - implanting strategy: the way the motif instances will be implanted in sequences/repertoires
    """
    def __init__(self, identifier, motifs: list, implanting_strategy: SignalImplantingStrategy):

        assert all([isinstance(m, Motif) for m in motifs])

        self.id = identifier
        self.motifs = motifs
        self.implanting_strategy = implanting_strategy

    def implant_to_repertoire(self, repertoire: SequenceRepertoire, repertoire_implanting_rate: float) \
            -> SequenceRepertoire:
        processed_repertoire = self.implanting_strategy\
                                .implant_in_repertoire(repertoire=repertoire,
                                                       repertoire_implanting_rate=repertoire_implanting_rate,
                                                       signal=self)
        return processed_repertoire

    def __str__(self):
        return self.id + "; " + ",".join([str(motif) for motif in self.motifs])

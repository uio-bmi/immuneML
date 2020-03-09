# quality: gold

from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.simulation.implants.Motif import Motif
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class Signal:
    """
    This class represents the signal that will be implanted during a Simulation.
    A signal is represented by a list of motifs, and an implanting strategy.

    An signal is associated with a metadata label, which is assigned to a receptor or repertoire.
    For example antigen-specific/disease-associated (receptor) or diseased (repertoire).


    Arguments:
        motifs (list): A list of the motifs associated with this signal.
        implanting_strategy (:py:obj:`~source.simulation.signal_implanting_strategy.SignalImplantingStrategy.SignalImplantingStrategy`):
            The strategy that is used to decide in which sequences the motifs should be implanted, and how.
            Currently, the only avaible implanting_strategy is :py:obj:`~source.simulation.signal_implanting_strategy.HealthySequenceImplanting.HealthySequenceImplanting`.


    Specification:

        motifs:
            my_motif:
                ...

        signals:
            my_signal:
                motifs:
                    - my_motif
                    - ...
                implanting: HealthySequence
                ...
    """
    def __init__(self, identifier, motifs: list, implanting_strategy: SignalImplantingStrategy):

        assert all([isinstance(m, Motif) for m in motifs])

        self.id = identifier
        self.motifs = motifs
        self.implanting_strategy = implanting_strategy

    def implant_to_repertoire(self, repertoire: SequenceRepertoire, repertoire_implanting_rate: float, path: str) \
            -> SequenceRepertoire:
        processed_repertoire = self.implanting_strategy\
                                .implant_in_repertoire(repertoire=repertoire,
                                                       repertoire_implanting_rate=repertoire_implanting_rate,
                                                       signal=self, path=path)
        return processed_repertoire

    def __str__(self):
        return self.id + "; " + ",".join([str(motif) for motif in self.motifs])

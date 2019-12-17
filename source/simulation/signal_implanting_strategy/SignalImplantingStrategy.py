# quality: gold

import abc

from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire


class SignalImplantingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def implant_in_repertoire(self, repertoire: SequenceRepertoire, repertoire_implanting_rate: float, signal, path):
        pass

    @abc.abstractmethod
    def implant_in_sequence(self, sequence, signal):
        pass

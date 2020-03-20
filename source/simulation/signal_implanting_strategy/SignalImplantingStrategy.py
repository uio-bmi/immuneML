# quality: gold

import abc

from source.data_model.repertoire.Repertoire import Repertoire


class SignalImplantingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def implant_in_repertoire(self, repertoire: Repertoire, repertoire_implanting_rate: float, signal, path):
        pass

    @abc.abstractmethod
    def implant_in_sequence(self, sequence, signal):
        pass

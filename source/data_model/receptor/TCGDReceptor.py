from uuid import uuid4

from source.data_model.receptor.Receptor import Receptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class TCGDReceptor(Receptor):

    def __init__(self, gamma: ReceptorSequence = None, delta: ReceptorSequence = None, metadata: dict = None):

        self.gamma = gamma
        self.delta = delta
        self.metadata = metadata
        self.identifier = uuid4().hex

    def get_chains(self):
        return ["gamma", "delta"]

    def get_attribute(self, name: str):
        raise NotImplementedError

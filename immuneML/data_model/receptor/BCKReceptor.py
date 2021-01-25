from uuid import uuid4

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class BCKReceptor(Receptor):

    def __init__(self, heavy: ReceptorSequence = None, kappa: ReceptorSequence = None, metadata: dict = None,
                 identifier: str = None):
        self.heavy = heavy
        self.kappa = kappa
        self.metadata = metadata
        self.identifier = uuid4().hex if identifier is None else identifier

    def get_chains(self):
        return ["heavy", "kappa"]

    def get_attribute(self, name: str):
        raise NotImplementedError

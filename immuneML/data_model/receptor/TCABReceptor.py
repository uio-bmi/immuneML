from uuid import uuid4

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class TCABReceptor(Receptor):

    def __init__(self, alpha: ReceptorSequence = None, beta: ReceptorSequence = None, metadata: dict = None, identifier: str = None):

        self.alpha = alpha
        self.beta = beta
        self.metadata = metadata
        self.identifier = uuid4().hex if identifier is None else identifier

    def get_chains(self):
        return ["alpha", "beta"]

    def get_attribute(self, name: str):
        raise NotImplementedError

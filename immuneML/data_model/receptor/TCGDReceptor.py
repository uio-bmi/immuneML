from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class TCGDReceptor(Receptor):
    FIELDS = {'gamma': object, 'delta': object, 'identifier': str, 'metadata': dict}

    @classmethod
    def get_record_names(cls):
        return ['gamma_' + name for name in ReceptorSequence.get_record_names()] \
               + ['delta_' + name for name in ReceptorSequence.get_record_names()] \
               + [name for name in cls.FIELDS if name not in ['gamma', 'delta']]

    def __init__(self, gamma: ReceptorSequence = None, delta: ReceptorSequence = None, metadata: dict = None, identifier: str = None):
        super().__init__(metadata=metadata, identifier=identifier)
        self.gamma = gamma
        self.delta = delta

    def get_chains(self):
        return ["gamma", "delta"]

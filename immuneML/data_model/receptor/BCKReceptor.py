from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class BCKReceptor(Receptor):
    FIELDS = {'heavy': object, 'kappa': object, 'identifier': str, 'metadata': dict}

    @classmethod
    def get_record_names(cls):
        return ['heavy_' + name for name in ReceptorSequence.get_record_names()] \
               + ['kappa_' + name for name in ReceptorSequence.get_record_names()] \
               + [name for name in cls.FIELDS if name not in ['heavy', 'kappa']]

    def __init__(self, heavy: ReceptorSequence = None, kappa: ReceptorSequence = None, metadata: dict = None, identifier: str = None):
        super().__init__(metadata=metadata, identifier=identifier)
        self.heavy = heavy
        self.kappa = kappa

    def get_chains(self):
        return ["heavy", "kappa"]

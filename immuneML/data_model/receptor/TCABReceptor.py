from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class TCABReceptor(Receptor):
    FIELDS = {'alpha': object, 'beta': object, 'identifier': str, 'metadata': dict, 'version': str}

    def __init__(self, alpha: ReceptorSequence = None, beta: ReceptorSequence = None, metadata: dict = None,
                 identifier: str = None):
        super().__init__(metadata=metadata, identifier=identifier)
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def get_record_names(cls):
        return ['alpha_' + name for name in ReceptorSequence.get_record_names()] \
            + ['beta_' + name for name in ReceptorSequence.get_record_names()] \
            + [name for name in cls.FIELDS if name not in ['alpha', 'beta']]

    def get_chains(self):
        return ["alpha", "beta"]

from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class BCReceptor(Receptor):
    FIELDS = {'heavy': object, 'light': object, 'identifier': str, 'metadata': dict}

    @classmethod
    def get_record_names(cls):
        return ['heavy_' + name for name in ReceptorSequence.get_record_names()] \
               + ['light_' + name for name in ReceptorSequence.get_record_names()] \
               + [name for name in cls.FIELDS if name not in ['heavy', 'light']]

    def __init__(self, heavy: ReceptorSequence = None, light: ReceptorSequence = None, metadata: dict = None,
                 identifier: str = None):
        super().__init__(metadata=metadata, identifier=identifier)
        self.heavy = heavy
        self.light = light

    def get_chains(self):
        return ["heavy", "light"]
